from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from build_promt import query_system
from DocumentProcessor import process_documents_incremental
import os
import json
import requests
from openai import OpenAI
import time
from typing import List, Optional
import uvicorn
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
from define_api import LocalEmbeddings
from build_promt import OptimizedKnowledgeBaseQuery
from openai import AsyncOpenAI

# 定义数据模型
class EmbeddingRequest(BaseModel):
    texts: List[str]


class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 5


class ChatRequest(BaseModel):
    model: Optional[str] = "qwen3:0.6b"
    messages: List[dict]
    tag: Optional[bool] = False
    think_tag: Optional[bool] = False


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    print("正在加载嵌入模型...")
    app.state.model = SentenceTransformer('./models/BAAI/bge-large-zh-v1___5')
    print("正在向量化本地文档...")
    app.state.vector_db = process_documents_incremental('./data')
    print("初始化完成！")
    yield
    # 关闭时清理资源
    print("正在关闭应用...")


# 创建FastAPI应用
app = FastAPI(title="知识库问答系统", lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件和模板
templates = Jinja2Templates(directory="templates")
os.makedirs("data", exist_ok=True)

# 配置API
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "xxxxxxxxxxx")
# OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://xxxxxxx/v1")

# 配置ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")


# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'pptx', 'ppt', 'xlsx', 'xls', 'csv', 'html', 'htm', 'md', 'docx', 'doc'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 路由定义
@app.post("/embed")
async def get_embeddings(request: EmbeddingRequest):
    """获取文本嵌入向量"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    embeddings = app.state.model.encode(request.texts).tolist()
    return embeddings


@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    """查询知识库"""
    if not request.question.strip():
        return JSONResponse(
            status_code=400,
            content={
                "answer": "问题不能为空",
                "sources": [],
                "context": "",
                "success": False
            }
        )

    try:
        result = query_system.query(request.question, request.n_results)
        # result = OptimizedKnowledgeBaseQuery(LocalEmbeddings(app.state.model)).query(request.question, request.n_results)
        return {
            "answer": result["full_prompt"],
            "sources": result["sources"],
            "context": result["context"],
            "success": result["success"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    """上传文件到知识库"""
    saved_files = []
    for file in files:
        if file.filename and not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f'不支持的文件格式：{file.filename}，仅支持：{", ".join(ALLOWED_EXTENSIONS)}'
            )

        # 保存文件
        file_path = os.path.join('./data', file.filename)
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        saved_files.append(file.filename)

    # 处理文档
    app.state.vector_db = process_documents_incremental('./data')

    return {
        "message": "文件上传成功",
        "saved_files": saved_files,
        "total_files": len(saved_files)
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """下载文件"""
    file_path = os.path.join('data', filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件未找到")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/ai_generate_chat")
async def ai_generate_chat_page(request: Request):
    """返回聊天页面"""
    return templates.TemplateResponse(
        "ai_generate_chat.html",
        {"request": request, "account": ""}
    )



@app.post("/ai_generate_chat")
async def ai_generate_chat(request: ChatRequest):
    """AI聊天接口"""
    start_time = time.time()

    # 处理聊天请求
    model_name = request.model
    messages = request.messages

    if not messages:
        raise HTTPException(status_code=400, detail="缺少 messages")

    # 准备生成参数
    options = {
        "temperature": 0.3,
        "top_p": 0.5,
        "top_k": 20,
        "repeat_penalty": 1.1
    }

    source = ''
    # 处理 think_tag（思考模式）
    think = '' if request.think_tag is True else ' /no_think'

    # 获取最新用户消息
    prompt = messages[-1].get('content', '')

    # 知识库检索（tag == True）
    if request.tag is True:
        try:
            result = query_system.query(prompt, n_results=3)
            # result = OptimizedKnowledgeBaseQuery(LocalEmbeddings(app.state.model)).query(prompt, n_results=3)
            prompt = result.get('full_prompt', prompt)

            # 构建 source 信息（仅取前3个）
            for doc in result.get('sources', [])[:3]:
                source += f"- 《{os.path.basename(doc.get('source', '未知文件'))}》 第{doc.get('page', 'N')}页\n"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"知识库查询失败: {str(e)}")

    # 更新最后一条消息（添加 think 指令）
    messages[-1] = {'role': 'user', 'content': prompt + think}

    async def stream_response():
        # 1. 先返回 source_info（如有）
        if source:
            yield json.dumps({
                "type": "source_info",
                "source": source.rstrip("\n")
            }, ensure_ascii=False).encode() + b"\n\n"

        # 2. 使用 AsyncOpenAI！
        try:
            client = AsyncOpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL.strip()
            )

            # 创建流式响应（真正异步！）
            stream = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                temperature=0.3,
                top_p=0.5
            )

            thinking_content = ""
            in_thinking = False

            # 异步迭代 stream，及时 yield
            async for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                think_content = chunk.choices[0].delta.model_extra.get('reasoning_content', "")

                # 处理思考内容
                if think_content:
                    if not in_thinking:
                        yield json.dumps({'message': {'content': '<think>\n'}, 'done': False}).encode() + b"\n\n"
                        in_thinking = True
                    yield json.dumps({'message': {'content': think_content}, 'done': False}).encode() + b"\n\n"
                elif in_thinking and not think_content:
                    yield json.dumps({'message': {'content': '\n</think>\n'}, 'done': False}).encode() + b"\n\n"
                    in_thinking = False

                # 处理回答内容
                if content:
                    yield json.dumps({'message': {'content': content}, 'done': False}).encode() + b"\n\n"

            # 结束
            end_time = time.time()
            yield json.dumps({
                'message': {'content': ''},
                'done': True,
                'eval_count': 0,  # 你可从 token_usage 统计
                'total_duration': (end_time - start_time) * 1e9
            }).encode() + b"\n\n"

        except Exception as e:
            yield json.dumps({"error": f"内部错误: {str(e)}"}).encode() + b"\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson"  # 或 text/event-stream
    )


@app.post("/upload_with_form")
async def upload_with_form(files: List[UploadFile] = File(...)):
    """兼容表单上传的接口"""
    saved_files = []
    for file in files:
        if not allowed_file(file.filename):
            continue

        file_path = os.path.join('./data', file.filename)
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        saved_files.append(file.filename)

    # 处理文档
    app.state.vector_db = process_documents_incremental('./data')

    return "文件上传成功"


@app.get("/")
async def root():
    """根路由"""
    return {
        "message": "知识库问答系统 API",
        "version": "1.0.0",
        "endpoints": {
            "/embed": "获取文本嵌入向量",
            "/query": "查询知识库",
            "/upload": "上传文件",
            "/download/{filename}": "下载文件",
            "/ai_generate_chat": "AI聊天接口"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        reload=False  # 生产环境建议设为False
    )