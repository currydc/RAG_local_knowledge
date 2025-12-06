from flask import Flask, request, jsonify
from build_promt import query_system

app = Flask(__name__)


@app.route('/query', methods=['POST'])
def query_knowledge_base():
    data = request.get_json()

    question = data.get("question", "").strip()
    n_results = data.get("n_results", 3)

    if not question:
        return jsonify({
            "answer": "问题不能为空",
            "sources": [],
            "context": "",
            "success": False
        }), 400

    try:
        result = query_system.query(question, n_results)
        return jsonify({
            "answer": result["full_prompt"],
            "sources": result["sources"],
            "context": result["context"],
            "success": result["success"]
        }), 200
    except Exception as e:
        return jsonify({
            "answer": f"查询失败: {str(e)}",
            "sources": [],
            "context": "",
            "success": False
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)