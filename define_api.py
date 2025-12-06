from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import requests


class LocalEmbeddings(Embeddings):
    """直接使用本地 SentenceTransformer 模型，无 HTTP 开销"""
    def __init__(self, model):
        self.model = model  # SentenceTransformer 实例

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

class CustomAPIEmbeddings(Embeddings):
    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key

    def embed_documents(self, texts):
        """为多个文档生成嵌入向量"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "texts": [texts]
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

    def embed_query(self, text):
        """为单个查询生成嵌入向量"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "texts": [text]
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            # 如果返回的是列表，取第一个元素
            result = response.json()
            if isinstance(result, list):
                return result[0] if len(result) > 0 else result
            return result
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")


# 使用自定义API嵌入模型
embedding_model = CustomAPIEmbeddings(
    api_url="http://127.0.0.1:5000/embed",  # 替换为实际的API端点
    api_key=""  # 如果需要API密钥
)

# 从已存在的向量数据库加载
vector_db = Chroma(
    persist_directory="./my_knowledge_db",
    embedding_function=embedding_model
)

