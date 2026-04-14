
from flask import Flask, jsonify, request, send_from_directory
import os

from classifier import classify_email, get_evaluation_report, get_model_info

app = Flask(__name__, static_folder=".")

# Frontend routes
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# API GET and POST routes
@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = classify_email(text)
    return jsonify(result)


@app.route("/evaluate")
def evaluate():
    report = get_evaluation_report()
    return jsonify(report)


@app.route("/info")
def info():
    return jsonify(get_model_info())


if __name__ == "__main__":
    print("\n🪼  SPAM DRIFT server starting...")
    print("    Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)