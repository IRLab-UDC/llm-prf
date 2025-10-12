from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load model and tokenizer once
model_name = "castorini/monot5-base-msmarco"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route("/eval", methods=["POST"])
def eval():
    data = request.json
    query = data["query"]
    doc = data["document"]
    # Replace all line breaks with spaces in the document text
    doc = doc.replace("\n", " ").replace("\r", " ")
    # Input format for MonoT5
    prompt = f"Query: {query} Document: {doc} Relevant:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generate output with scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=2,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Decoded sequence ("true" or "false")
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Get logits from the first generated token
    logits = outputs.scores[0][0]  # scores for the first generated token
    true_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_id = tokenizer.encode("false", add_special_tokens=False)[0]
    
    # Calculate probabilities
    probs = torch.softmax(logits, dim=-1)
    prob_true = probs[true_id].item()
    prob_false = probs[false_id].item()
    
    # Individual logits
    logit_true = logits[true_id].item()
    logit_false = logits[false_id].item()
    
    # Score (probability of "true")
    score = prob_true

    result = {
        "prediction": decoded.strip(),
        "score": score,
        "logit_true": logit_true,
        "logit_false": logit_false,
        "prob_true": prob_true,
        "prob_false": prob_false
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)