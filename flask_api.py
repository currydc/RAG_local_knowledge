from flask import Flask, request, send_from_directory, render_template, jsonify, Response
from sentence_transformers import SentenceTransformer
from build_promt import OptimizedKnowledgeBaseQuery,query_system
from DocumentProcessor import process_documents_incremental
from define_api import LocalEmbeddings
import os
import json
import requests
from openai import OpenAI
import time


app = Flask(__name__)
#使用本地嵌入模型
model = SentenceTransformer('./models/BAAI/bge-large-zh-v1___5')
#向量化本地文档
process_documents_incremental('./data')

@app.route('/embed', methods=['POST'])
def get_embeddings():
    data = request.json
    texts = data.get('texts', [])

    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    embeddings = model.encode(texts).tolist()

    return embeddings


@app.route('/query', methods=['POST'])
def query_knowledge_base():
    data = request.get_json()

    question = data.get("question", "").strip()
    n_results = data.get("n_results", 5)

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


@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files
    for k, v in files.items():
        file = v
        file.save(os.path.join('./data', file.filename))
    vector_db = process_documents_incremental('./data')
    return '文件上传成功'


# 设置文件下载目录
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), 'data')
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER


@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_from_directory(
            app.config['DOWNLOAD_FOLDER'],
            filename,
            as_attachment=True  # 强制下载
        )
    except FileNotFoundError:
        return "文件未找到", 404



# #配置ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","ollama")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")

# #配置api
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","XXXXXXXXX")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://XXXXXXXX/v1")

# 允许的文件扩展名（保持与原逻辑一致）
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'pptx', 'ppt', 'xlsx', 'xls', 'csv', 'html', 'htm', 'md', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/ai_generate_chat', methods=['GET', 'POST'])
def ai_generate_chat():
    start_time = time.time()
    if request.method == 'GET':
        manage_info = {'account': ""}
        return render_template('ai_generate_chat.html', **manage_info)

    elif request.method == 'POST':
        # 处理文件上传
        if request.files:
            files = request.files
            for k, file in files.items():
                if file and file.filename:
                    filename = file.filename
                    if not allowed_file(filename):
                        return f'不支持的文件格式：{filename}，仅支持：{", ".join(ALLOWED_EXTENSIONS)}', 400
                    # 转发文件到后端服务
                    files_dict = {'file': (filename, file.stream, file.mimetype)}
                    try:
                        res = requests.post('http://127.0.0.1:5000/upload', files=files_dict, timeout=30)
                        res.raise_for_status()
                    except requests.RequestException as e:
                        return f'文件上传失败: {str(e)}', 500
            return 'ok'

        # 处理聊天请求（无文件上传）
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({"error": "无效的 JSON 请求体"}), 400
        model_name = data.get("model", "qwen3:0.6b")
        # 设置固定生成参数
        data['options'] = {
            "temperature": 0.3,
            "top_p": 0.5,
            "top_k": 20,
            "repeat_penalty": 1.1
        }

        source = ''
        # 处理 think_tag（思考模式）
        think = '' if data.get('think_tag') is True else ' /no_think'

        # 获取最新用户消息
        messages = data.get('messages', [])
        if not messages:
            return jsonify({"error": "缺少 messages"}), 400
        prompt = messages[-1].get('content', '')

        # 知识库检索（tag == True）
        if data.get('tag') is True:
            try:
                # res = requests.post(
                #     'http://127.0.0.1:5000/query',
                #     json={"question": prompt, "n_results": 3},
                #     timeout=30
                # )
                # res.raise_for_status()
                # result = res.json()
                # prompt = result.get('answer', prompt)
                result = query_system.query(prompt, n_results=3)
                prompt = result.get('full_prompt', prompt)

                # 构建 source 信息（仅取前3个）
                for doc in result.get('sources', [])[:3]:
                    source += f"- 《{os.path.basename(doc.get('source', '未知文件'))}》 第{doc.get('page', 'N')}页\n"
            except Exception as e:
                return jsonify({"error": f"知识库查询失败: {str(e)}"}), 500

        # 更新最后一条消息（添加 think 指令）
        data['messages'][-1] = {'role': 'user', 'content': prompt + think}


        # ✅ 根据 OPENAI_API_KEY 选择后端
        def stream_response():
            # 1. 先返回 source_info（如有）
            if source:
                yield json.dumps({
                    "type": "source_info",
                    "source": source.rstrip("\n")
                }, ensure_ascii=False) + "\n"

            # 2. 调用模型
            try:
                # —— OpenAI 流式调用 ——
                client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL
                )

                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    temperature=0.3,
                    top_p=0.5
                )

                count = 0
                thinking_content = ""  # 缓存思考内容
                in_thinking = False  # 标记是否在思考过程中

                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    think = chunk.choices[0].delta.model_extra.get('reasoning_content','')
                    if think:
                        if not in_thinking:
                            # 思考开始，发送开始标签
                            yield f"{json.dumps({'message': {'content': '<think>\n'}, 'done': False})}\n\n"
                            in_thinking = True
                        # 发送思考内容片段
                        yield f"{json.dumps({'message': {'content': think}, 'done': False})}\n\n"
                        thinking_content += think
                    elif in_thinking and think=='':
                        # 思考结束，发送结束标签
                        yield f"{json.dumps({'message': {'content': '\n</think>\n'}, 'done': False})}\n\n"
                        in_thinking = False
                        thinking_content = ""
                    if content:
                        yield f"{json.dumps({'message': {'content': content}, 'done': False})}\n\n"
                    count = count + 1
                else:
                    # 流结束时，如果还在思考中，发送结束标签
                    if in_thinking:
                        yield f"{json.dumps({'message': {'content': '</think>'}, 'done': False})}\n\n"
                    end_time = time.time()
                    if 'ollama' in OPENAI_API_KEY:
                        pass
                    else:
                        count = count * 4
                    yield f"{json.dumps({'message': {'content': content}, 'done': True, 'eval_count': count, 'total_duration': (end_time - start_time) * 1e9})}\n\n"

            except Exception as e:
                yield json.dumps({"error": f"内部错误: {str(e)}"}) + "\n"

        return Response(stream_response(), mimetype="application/json")

if __name__ == '__main__':
    # 调试
    # app.run(host='0.0.0.0', port=5000,debug=True,use_reloader=True)
    app.run(host='0.0.0.0', port=5000)
