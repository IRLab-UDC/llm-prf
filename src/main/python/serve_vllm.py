import json
import math

from fastapi import FastAPI
from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


class RelevanceCheck(BaseModel):
    relevance: bool = Field(..., description="Relevance. Boolean (true/false).")


json_schema = json.dumps(RelevanceCheck.model_json_schema())

app = FastAPI(title="vLLM Logit Server")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_MODEL_LEN = 8192  # Context window for Llama 3.1 8B
MAX_PROMPT_TOKENS = 7900  # Leave room for response and system message
llm = LLM(model=MODEL_NAME, max_logprobs=1000, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=.4')
tokenizer = llm.get_tokenizer()


class LogitRequest(BaseModel):
    prompt: str


@app.post("/prob")
def get_true_false_probs(request: LogitRequest):
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
        guided_decoding=guided_decoding_params,
    )

    response = llm.chat([messages], sampling_params)
    logprobs = response[0].outputs[0].logprobss

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

    # Handle missing tokens
    if logprob_false is None or logprob_true is None:
        print("Could not find true/false tokens in logprobs at detected position")
        return {
            "p_true": 0.5,
            "p_false": 0.5,
            "error": "Could not find true/false tokens in logprobs",
        }

    # Handle -inf logprobs (zero probability tokens)
    if math.isinf(logprob_false) and math.isinf(logprob_true):
        print("Both true and false tokens have -inf logprob")
        # Both are -inf, return equal probabilities
        p_true = 0.5
        p_false = 0.5
    elif math.isinf(logprob_false):
        print("False token has -inf logprob")
        # Only false is -inf, true has probability 1
        p_true = 1.0
        p_false = 0.0
    elif math.isinf(logprob_true):
        print("True token has -inf logprob")
        # Only true is -inf, false has probability 1
        p_true = 0.0
        p_false = 1.0
    else:
        # Normal case: compute probabilities
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