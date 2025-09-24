from flask import Flask, render_template, request, jsonify
import os

from inference import (
    load_letter_model, predict_letter,
    load_word_model,   predict_word,
    predict_words
)

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your models once at startup
letter_model = load_letter_model("models/lettercnn_best.pth")
word_model   = load_word_model("word_model.pth")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    mode = request.form.get("mode")
    if not file or not mode:
        return jsonify({"error": "Missing file or mode"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    abs_path = os.path.abspath(save_path)
    print("📂 Absolute path:", abs_path)

    try:
        if mode == "letter":
            # single‐letter predict
            pred = predict_letter(letter_model, abs_path)

        elif mode == "word":
            # single‐word predict
            pred = predict_word(word_model, abs_path)

        elif mode == "text":
            # full‐text pipeline
            pred = predict_words(abs_path)
            # use_gpt = request.form.get("use_gpt") == "true"
            # print(f"[APP DEBUG] use_gpt flag = {use_gpt}")
            # pred = predict_words(abs_path, use_gpt=use_gpt)

        else:
            return jsonify({"error": "Invalid mode"}), 400

        return jsonify({
            "prediction": pred,
            "image_url": "/" + save_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
