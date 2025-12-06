from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载文档（以PDF为例）
loader = DirectoryLoader(r'./data', glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# 2. 切分文档成小块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个块的大小
    chunk_overlap=50  # 块之间的重叠部分，避免语义断裂
)
chunks = text_splitter.split_documents(documents)
pass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 使用本地嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name=r"./models/BAAI/bge-small-zh" # 或者 "moka-ai/m3e-base"
)

# 将文本块转换为向量并存入ChromaDB
# persist_directory 指定持久化目录，否则数据只在内存中
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./my_knowledge_db"
)
pass