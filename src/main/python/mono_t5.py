from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Cargar modelo y tokenizer una sola vez
model_name = "castorini/monot5-base-msmarco"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route("/eval", methods=["POST"])
def eval():
    data = request.json
    query = data["query"]
    doc = data["document"]

    # Formato de entrada para MonoT5
    prompt = f"Query: {query} Document: {doc} Relevant:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generar salida con scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=2,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Secuencia decodificada ("true" o "false")
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Obtener logits del primer token generado
    logits = outputs.scores[0][0]  # scores para el primer token generado
    true_id = tokenizer.encode("true", add_special_tokens=False)[0]
    false_id = tokenizer.encode("false", add_special_tokens=False)[0]
    
    # Calcular probabilidades
    probs = torch.softmax(logits, dim=-1)
    prob_true = probs[true_id].item()
    prob_false = probs[false_id].item()
    
    # Logits individuales
    logit_true = logits[true_id].item()
    logit_false = logits[false_id].item()
    
    # Score (probabilidad de "true")
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