from flask import Flask, request, jsonify
import logging
from semantic_similarity import compute_similarity
import os
from flask_cors import CORS
# Set up structured logging for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Enable CORS for all endpoints
CORS(app)

# Health check endpoint
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "API is running"}), 200


@app.route('/predict_similarity', methods=['POST'])
def predict_similarity():
    try:
        # Parse JSON data
        data = request.get_json()
        text1 = data.get('text1', '').strip()
        text2 = data.get('text2', '').strip()

        # Input validation
        if not text1 or not text2:
            logging.warning("Empty input detected")
            return jsonify({"error": "Both 'text1' and 'text2' must be non-empty strings."}), 400

        logging.info(f"Received texts: {text1[:50]}..., {text2[:50]}...")

        # Compute similarity
        similarity_score = compute_similarity(text1, text2)
        logging.info(f"Similarity score computed: {similarity_score}")

        return jsonify({"similarity score": similarity_score}), 200

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal Server Error. Please try again later."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
