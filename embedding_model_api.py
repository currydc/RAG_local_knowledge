from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('./models/BAAI/bge-base-zh-v1___5')


@app.route('/embed', methods=['POST'])
def get_embeddings():
    data = request.json
    texts = data.get('texts', [])

    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    embeddings = model.encode(texts).tolist()

    return embeddings


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)