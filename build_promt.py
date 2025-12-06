from langchain_chroma import Chroma
from define_api import CustomAPIEmbeddings,LocalEmbeddings
import logging
from typing import Dict, Any, List
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedKnowledgeBaseQuery:
    def __init__(self, embedding_model=None,persist_directory: str = "./my_knowledge_db"):
        self.embedding_model = embedding_model or CustomAPIEmbeddings(
            api_url="http://127.0.0.1:5000/embed",
            api_key = ""
        )

        try:
            self.vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_model
            )
            logger.info("向量数据库加载成功")
        except Exception as e:
            logger.error(f"加载向量数据库失败: {e}")
            raise

    def _preprocess_question(self, question: str) -> str:
        """预处理问题"""
        # 移除多余空格和特殊字符
        question = re.sub(r'\s+', ' ', question.strip())
        return question

    def _create_dynamic_prompt(self, context: str, question: str) -> str:
        """根据上下文质量动态调整提示"""
        if "未找到相关上下文" in context or len(context.strip()) < 50:
            template = """问题：{question}

很抱歉，我在知识库中没有找到相关的信息来回答这个问题。请尝试换一种方式提问或咨询其他信息来源。"""
        else:
            template = """
基于以下提供的上下文信息，请回答用户的问题。如果上下文信息不足以回答问题，请说明无法基于提供的信息回答。

上下文信息：
{context}

用户问题：
{question}

请用中文回答：
"""

        return template.format(context=context, question=question)



    def _semantic_filter(self, question: str, documents: List, threshold: float = 0.3) -> List:
        """基于语义相似度过滤文档"""
        try:
            # 获取问题向量
            question_vec = np.array(self.embedding_model.embed_query(question)).reshape(1, -1)

            filtered_docs = []
            for doc in documents:
                doc_vec = np.array(self.embedding_model.embed_query(doc.page_content)).reshape(1, -1)
                sim = cosine_similarity(question_vec, doc_vec)[0][0]
                if sim >= threshold:
                    doc.metadata["similarity_score"] = float(sim)  # 记录分数用于排序或调试
                    filtered_docs.append(doc)

            # 按相似度降序排序
            filtered_docs.sort(key=lambda x: x.metadata.get("similarity_score", 0), reverse=True)
            return filtered_docs
        except Exception as e:
            logger.warning(f"语义过滤失败，回退到关键词过滤: {e}")
            return documents  # 降级回原逻辑
    def _retrieve_with_fallback(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """带降级策略的检索"""
        try:
            # 尝试主要检索方法
            documents = self.vector_db.similarity_search(question, k=n_results * 2)

            if not documents or len(documents) < n_results:
                # 降级策略：尝试更宽松的检索
                logger.warning("主要检索结果不足，尝试降级检索")
                documents = self.vector_db.similarity_search(question, k=n_results * 3)


            #轻量语义相似度过滤
            # filtered_docs = self._semantic_filter(question, documents, threshold=0.35)
            # 简单的相关性过滤
            filtered_docs = []
            question_lower = question.lower()
            for doc in documents:
                content_lower = doc.page_content.lower()
                # 简单的关键词匹配过滤
                if any(keyword in content_lower for keyword in question_lower.split()):
                    filtered_docs.append(doc)

            # 如果过滤后结果太少，保留原始结果
            if len(filtered_docs) < n_results // 2:
                filtered_docs = documents[:n_results]

            search_results = {
                'documents': [[doc.page_content for doc in filtered_docs[:n_results]]],
                'metadatas': [[doc.metadata for doc in filtered_docs[:n_results]]]
            }

            return search_results

        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            raise

    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """优化的查询方法"""
        if not question or not question.strip():
            return {
                "answer": "问题不能为空",
                "sources": [],
                "context": "",
                "success": False
            }

        try:
            # 预处理问题
            processed_question = self._preprocess_question(question)

            # 检索相关文档
            search_results = self._retrieve_with_fallback(processed_question, n_results)

            # 构建上下文
            context = self._build_context(search_results)

            # 动态生成提示
            full_prompt = self._create_dynamic_prompt(context, processed_question)

            return {
                "full_prompt": full_prompt,
                "sources": search_results.get('metadatas', [[]])[0],
                "context": context,
                "success": True
            }

        except Exception as e:
            logger.error(f"查询过程中发生错误: {e}")
            return {
                "full_prompt": f"查询失败: {str(e)}",
                "sources": [],
                "context": "",
                "success": False
            }

    def _build_context(self, search_results: Dict[str, Any]) -> str:
        """构建上下文字符串"""
        if not search_results or 'documents' not in search_results:
            return "未找到相关上下文信息"

        documents = search_results['documents'][0]
        if not documents:
            return "未找到相关上下文信息"

        # 添加文档来源标记
        context_parts = []
        for i, doc_content in enumerate(documents):
            context_parts.append(f"[文档{i + 1}]: {doc_content}")

        return "\n\n".join(context_parts)
model = SentenceTransformer('./models/BAAI/bge-large-zh-v1___5')
model = LocalEmbeddings(model)
query_system=OptimizedKnowledgeBaseQuery(model)
# 调用嵌入模型api
# query_system=OptimizedKnowledgeBaseQuery()

# res=query_system.query('测试', n_results=4)
# print(res)