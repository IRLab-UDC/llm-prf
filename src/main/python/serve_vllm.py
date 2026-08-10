import asyncio
import json
import math
from typing import Annotated, List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


class RelevanceCheck(BaseModel):
    relevance: bool = Field(..., description="Relevance. Boolean (true/false).")


class RelevantSpans(BaseModel):
    spans: List[Annotated[str, Field(max_length=300)]] = Field(
        ...,
        description="Most relevant passages or sentences from the document for the query.",
        max_length=10,
    )


class JudgeSpans(BaseModel):
    relevant: bool = Field(..., description="Whether the document is relevant to the query.")
    spans: List[Annotated[str, Field(max_length=300)]] = Field(
        ...,
        description=(
            "Most relevant passages or sentences from the document for the query. "
            "Must be empty if relevant is false."
        ),
        max_length=10,
    )


json_schema = json.dumps(RelevanceCheck.model_json_schema())
spans_json_schema = json.dumps(RelevantSpans.model_json_schema())
judge_spans_json_schema = json.dumps(JudgeSpans.model_json_schema())

app = FastAPI(title="vLLM Logit Server")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_MODEL_LEN = 8192
MAX_PROMPT_TOKENS = 7900
llm = LLM(model=MODEL_NAME, max_logprobs=1000, max_model_len=MAX_MODEL_LEN)
tokenizer = llm.get_tokenizer()

_inference_lock = asyncio.Lock()


class LogitRequest(BaseModel):
    prompt: str


@app.post("/prob")
async def get_true_false_probs(request: LogitRequest):
    async with _inference_lock:
        guided_decoding_params = StructuredOutputsParams(json=json_schema)

        user_prompt = request.prompt

        print(f"Original user prompt: {user_prompt}")

        tokens = tokenizer.encode(user_prompt)
        if len(tokens) > MAX_PROMPT_TOKENS:
            tokens = tokens[:MAX_PROMPT_TOKENS]
            user_prompt = tokenizer.decode(tokens)

        print(f"Truncated user prompt: {user_prompt}")

        messages = [
            {
                "role": "system",
                "content": """You are a strict TREC assessor that outputs machine-readable judgments. Return only a JSON object in the format: { "relevance": true } or { "relevance": false } Do not include any other text, explanation, or reasoning.""",
            },
            {"role": "user", "content": user_prompt},
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,
            logprobs=1000,
            structured_outputs=guided_decoding_params,
        )

        response = llm.chat([messages], sampling_params)
        logprobs = response[0].outputs[0].logprobs

        target_pos = None
        for pos in range(len(logprobs)):
            for token_id, entry in logprobs[pos].items():
                if entry.rank == 1:
                    if (
                        entry.decoded_token.strip() == "false"
                        or entry.decoded_token.strip() == "true"
                    ):
                        print(f"Found exact match for true/false at position {pos}")
                        target_pos = pos
                        break

        if target_pos is None:
            print("Could not automatically detect true/false token position")
            return {
                "p_true": 0.5,
                "p_false": 0.5,
                "error": "Could not detect true/false token position",
            }

        logprob_false = None
        logprob_true = None

        for token_id, entry in logprobs[target_pos].items():
            decoded = entry.decoded_token
            if "false" in decoded and logprob_false is None:
                logprob_false = entry.logprob
                print(f"Found false token: '{decoded}' with logprob {logprob_false}")
            elif "true" in decoded and logprob_true is None:
                logprob_true = entry.logprob
                print(f"Found true token: '{decoded}' with logprob {logprob_true}")

        if logprob_false is None or logprob_true is None:
            print("Could not find true/false tokens in logprobs at detected position")
            return {
                "p_true": 0.5,
                "p_false": 0.5,
                "error": "Could not find true/false tokens in logprobs",
            }

        if math.isinf(logprob_false) and math.isinf(logprob_true):
            print("Both true and false tokens have -inf logprob")
            p_true = 0.5
            p_false = 0.5
        elif math.isinf(logprob_false):
            print("False token has -inf logprob")
            p_true = 1.0
            p_false = 0.0
        elif math.isinf(logprob_true):
            print("True token has -inf logprob")
            p_true = 0.0
            p_false = 1.0
        else:
            max_logprob = max(logprob_false, logprob_true)
            exp_false = math.exp(logprob_false - max_logprob)
            exp_true = math.exp(logprob_true - max_logprob)

            p_true = exp_true / (exp_true + exp_false)
            p_false = exp_false / (exp_true + exp_false)

            print(f"prob true: {p_true}, prob false: {p_false}")

        return {
            "p_true": p_true,
            "p_false": p_false,
        }

@app.post("/spans")
async def get_relevant_spans(request: LogitRequest):
    async with _inference_lock:
        guided_decoding_params = StructuredOutputsParams(json=spans_json_schema)

        user_prompt = request.prompt

        tokens = tokenizer.encode(user_prompt)
        if len(tokens) > MAX_PROMPT_TOKENS:
            tokens = tokens[:MAX_PROMPT_TOKENS]
            user_prompt = tokenizer.decode(tokens)

        messages = [
            {
                "role": "system",
                "content": "You are a search quality assessor identifying the most relevant passages in a document for a given query. Return only text that appears in the document.",
            },
            {"role": "user", "content": user_prompt},
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            structured_outputs=guided_decoding_params,
        )

        response = llm.chat([messages], sampling_params)
        output = response[0].outputs[0]
        output_text = output.text
        finish_reason = output.finish_reason

        print(f"[/spans] finish_reason={finish_reason} len={len(output_text)}")
        print(f"[/spans] raw output: {repr(output_text[:500])}")

        try:
            result = json.loads(output_text)
            raw_spans = result.get("spans", [])
            seen = set()
            spans = [s for s in raw_spans if s and not (s in seen or seen.add(s))]
            print(f"[/spans] OK: {len(spans)} spans (from {len(raw_spans)} raw): {[s[:60] for s in spans[:3]]}")
            return {"spans": spans}
        except Exception as e:
            print(f"[/spans] parse error: {e}")
            print(f"[/spans] full output: {repr(output_text)}")
            return {"spans": []}


@app.post("/judge_spans")
async def judge_and_extract_spans(request: LogitRequest):
    async with _inference_lock:
        guided_decoding_params = StructuredOutputsParams(json=judge_spans_json_schema)

        user_prompt = request.prompt

        tokens = tokenizer.encode(user_prompt)
        if len(tokens) > MAX_PROMPT_TOKENS:
            tokens = tokens[:MAX_PROMPT_TOKENS]
            user_prompt = tokenizer.decode(tokens)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict TREC assessor. First judge whether the document is "
                    "relevant to the query. If it is relevant, identify the passages or "
                    "sentences that most directly address the query; return only text that "
                    "appears in the document. If it is not relevant, return an empty list of "
                    "spans."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            structured_outputs=guided_decoding_params,
        )

        response = llm.chat([messages], sampling_params)
        output = response[0].outputs[0]
        output_text = output.text
        finish_reason = output.finish_reason

        print(f"[/judge_spans] finish_reason={finish_reason} len={len(output_text)}")
        print(f"[/judge_spans] raw output: {repr(output_text[:500])}")

        try:
            result = json.loads(output_text)
            relevant = bool(result.get("relevant", False))
            raw_spans = result.get("spans", []) if relevant else []
            seen = set()
            spans = [s for s in raw_spans if s and not (s in seen or seen.add(s))]
            print(f"[/judge_spans] OK: relevant={relevant}, {len(spans)} spans (from {len(raw_spans)} raw)")
            return {"relevant": relevant, "spans": spans}
        except Exception as e:
            print(f"[/judge_spans] parse error: {e}")
            print(f"[/judge_spans] full output: {repr(output_text)}")
            return {"relevant": False, "spans": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=False,
    )