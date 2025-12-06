from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import re
import hashlib
from typing import List, Dict, Any, Optional
import json
import glob
from pathlib import Path
import nltk
nltk.data.path.append('./nltk_data')
# 动态导入Word加载器
try:
    from langchain.document_loaders import Docx2txtLoader

    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("警告: 未安装 docx2txt，将跳过Word文档处理")

# 尝试导入代码分割器
try:
    from langchain_experimental.text_splitter import SemanticChunker

    SEMANTIC_CHUNKING_SUPPORT = True
except ImportError:
    SEMANTIC_CHUNKING_SUPPORT = False
    print("注意: 语义分块不可用，将使用基础分块方法")


def get_document_loaders():
    """返回支持的文档类型和对应的加载器"""
    loaders = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,
        '.csv': CSVLoader,
        '.html': UnstructuredHTMLLoader,
        '.htm': UnstructuredHTMLLoader,
        '.md': TextLoader,
    }

    # 代码文件支持
    code_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs', '.ts','.sql']
    for ext in code_extensions:
        loaders[ext] = TextLoader

    if DOCX_SUPPORT:
        loaders['.docx'] = Docx2txtLoader
        loaders['.doc'] = Docx2txtLoader
    else:
        print("Word文档支持已禁用 (缺少 docx2txt 依赖)")

    return loaders


def get_file_hash(file_path: str) -> str:
    """计算文件的哈希值用于检测变更"""
    try:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        print(f"计算文件哈希时出错 {file_path}: {e}")
        return "error"


def safe_load_document(loader_cls, file_path: str):
    """安全加载文档，处理编码和文件名问题"""
    try:
        # 对于文本文件，指定UTF-8编码
        if loader_cls == TextLoader:
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            loader = loader_cls(file_path)

        documents = loader.load()
        return documents
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            loader = TextLoader(file_path, encoding='gbk')
            documents = loader.load()
            return documents
        except:
            print(f"编码问题无法加载文件: {file_path}")
            return []
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return []


def load_documents_from_directory(directory_path: str, processed_files: Dict[str, str] = None) -> tuple:
    """从目录加载所有支持的文档类型，支持中文文件名"""
    loaders = get_document_loaders()
    all_documents = []
    new_processed_files = processed_files.copy() if processed_files else {}
    loaded_stats = {}
    error_files = []

    # 使用glob手动查找文件，避免DirectoryLoader的中文路径问题
    for ext, loader_cls in loaders.items():
        try:
            # 使用Path处理中文路径
            pattern = os.path.join(directory_path, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)

            for file_path in files:
                try:
                    # 跳过隐藏文件
                    if os.path.basename(file_path).startswith('.'):
                        continue

                    file_hash = get_file_hash(file_path)

                    # 检查文件是否已处理且未更改
                    if processed_files and file_path in processed_files:
                        if processed_files[file_path] == file_hash:
                            continue

                    # 加载文档
                    documents = safe_load_document(loader_cls, file_path)

                    if documents:
                        for doc in documents:
                            # 确保metadata中有正确的source
                            doc.metadata['source'] = file_path
                            doc.metadata['file_name'] = os.path.basename(file_path)

                        all_documents.extend(documents)
                        new_processed_files[file_path] = file_hash
                        loaded_stats[ext] = loaded_stats.get(ext, 0) + 1

                except Exception as e:
                    error_files.append((file_path, str(e)))
                    continue

        except Exception as e:
            print(f"处理 {ext} 文件时出错: {e}")
            continue

    # 打印统计信息
    if loaded_stats:
        print("\n文档加载统计:")
        for ext, count in loaded_stats.items():
            print(f"  {ext}: {count} 个新文件")

    if error_files:
        print(f"\n警告: {len(error_files)} 个文件加载失败")
        for file_path, error in error_files[:5]:  # 只显示前5个错误
            print(f"  失败: {os.path.basename(file_path)} - {error}")
        if len(error_files) > 5:
            print(f"  还有 {len(error_files) - 5} 个错误未显示...")

    return all_documents, new_processed_files
def create_optimal_splitter(file_extension=None):
    """根据文件类型创建最优化的文本分割器"""
    base_config = {
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'length_function': len,
    }

    if file_extension in ['.html', '.htm']:
        return RecursiveCharacterTextSplitter(
            **base_config,
            separators=["\n\n", "\n", "。", "！", "？", "；", "\\. ", "! ", "\\? ", "; ", " ", ""]
        )

    elif file_extension in ['.py', '.js', '.java', '.cpp', '.c', '.cs']:
        language = None
        if file_extension == '.py':
            language = Language.PYTHON
        elif file_extension == '.js':
            language = Language.JS
        elif file_extension in ['.java', '.cpp', '.c', '.cs']:
            language = Language.CPP

        if language:
            return RecursiveCharacterTextSplitter.from_language(language=language, **base_config)

    elif file_extension == '.md':
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        return MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    return RecursiveCharacterTextSplitter(
        **base_config,
        separators=["\n\n", "\n", "。", "！", "？", "；", "\\. ", "! ", "\\? ", "; ", " ", ""]
    )


def split_documents_by_type(documents):
    """根据文档类型分别进行分割处理"""
    documents_by_type = {}
    for doc in documents:
        source = doc.metadata.get('source', '')
        _, ext = os.path.splitext(source)
        ext = ext.lower()

        if ext not in documents_by_type:
            documents_by_type[ext] = []
        documents_by_type[ext].append(doc)

    all_chunks = []
    for ext, docs in documents_by_type.items():
        print(f"处理 {ext} 文件: {len(docs)} 个文档")

        splitter = create_optimal_splitter(ext)

        try:
            if ext == '.md':
                for doc in docs:
                    try:
                        with open(doc.metadata['source'], 'r', encoding='utf-8') as f:
                            md_content = f.read()
                        chunks = splitter.split_text(md_content)
                        for chunk in chunks:
                            chunk.metadata = doc.metadata.copy()
                        all_chunks.extend(chunks)
                    except UnicodeDecodeError:
                        with open(doc.metadata['source'], 'r', encoding='gbk') as f:
                            md_content = f.read()
                        chunks = splitter.split_text(md_content)
                        for chunk in chunks:
                            chunk.metadata = doc.metadata.copy()
                        all_chunks.extend(chunks)
            else:
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)

            print(f"  -> 分割为 {len(chunks)} 个块")
        except Exception as e:
            print(f"  处理 {ext} 文件时出错: {e}")
            default_splitter = create_optimal_splitter()
            chunks = default_splitter.split_documents(docs)
            all_chunks.extend(chunks)

    return all_chunks


def enhance_chunk_metadata(chunks):
    """增强块的元数据"""
    for chunk in chunks:
        source = chunk.metadata.get('source', '')
        content = chunk.page_content

        if 'title' not in chunk.metadata:
            title_match = re.search(r'^(#+ |【|标题[：:]|title[：:])(.+?)(\n|$)', content, re.IGNORECASE)
            if title_match:
                chunk.metadata['title'] = title_match.group(2).strip()
            else:
                chunk.metadata['title'] = os.path.splitext(os.path.basename(source))[0]

        _, ext = os.path.splitext(source)
        chunk.metadata['document_type'] = ext.lower()

        if len(content) > 50:
            chunk.metadata['summary'] = content[:50] + "..."
        else:
            chunk.metadata['summary'] = content

    return chunks


def load_processed_files(metadata_file: str) -> Dict[str, str]:
    """加载已处理文件的元数据"""
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_processed_files(metadata_file: str, processed_files: Dict[str, str]):
    """保存已处理文件的元数据"""
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(processed_files, f, ensure_ascii=False, indent=2)


def get_existing_vector_db(persist_directory: str = "./my_knowledge_db"):
    """获取现有的向量数据库"""
    embedding_model = HuggingFaceEmbeddings(
        model_name="./models/BAAI/bge-large-zh-v1___5"
    )

    if os.path.exists(persist_directory):
        try:
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
        except Exception as e:
            print(f"加载现有向量数据库时出错: {e}")
            return None
    return None


def process_documents_incremental(data_directory="./data", persist_directory="./my_knowledge_db"):
    """增量处理目录中的文档"""
    if not os.path.exists(data_directory):
        raise ValueError(f"目录 {data_directory} 不存在")

    # 加载已处理文件记录
    metadata_file = os.path.join(persist_directory, "processed_files.json")
    processed_files = load_processed_files(metadata_file)

    print("开始增量加载文档...")
    documents, new_processed_files = load_documents_from_directory(data_directory, processed_files)

    if not documents:
        print("没有需要处理的新文件或更改的文件")
        return get_existing_vector_db(persist_directory)

    print(f"\n总共加载了 {len(documents)} 个新文档")

    # 切分文档成小块
    print("开始切分文档...")
    chunks = split_documents_by_type(documents)
    chunks = enhance_chunk_metadata(chunks)

    print(f"文档被切分成 {len(chunks)} 个文本块")

    # 获取或创建向量数据库
    vector_db = get_existing_vector_db(persist_directory)

    if vector_db:
        print("向现有向量数据库添加新文档...")
        vector_db.add_documents(chunks)
        print("新文档已添加到向量数据库")
    else:
        print("创建新的向量数据库...")
        embedding_model = HuggingFaceEmbeddings(
            model_name="./models/BAAI/bge-large-zh-v1___5"
        )
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        print(f"向量数据库已创建并保存到 {persist_directory}")

    # 更新已处理文件记录
    save_processed_files(metadata_file, new_processed_files)

    if vector_db:
        collection = vector_db._client.get_collection(vector_db._collection.name)
        print(f"向量数据库中总共有 {collection.count()} 个文档块")

    return vector_db


def main():
    """主函数"""
    try:
        # 增量处理文档
        vector_db = process_documents_incremental("./data")

        if vector_db:
            # 测试查询
            query = "你的查询内容"
            results = vector_db.similarity_search(query, k=3)
            print(f"\n找到 {len(results)} 个相关结果")

            for i, result in enumerate(results, 1):
                source = result.metadata.get('source', '未知来源')
                title = result.metadata.get('title', '无标题')
                filename = os.path.basename(source)
                print(f"结果 {i}: {title} (来自 {filename})")
        else:
            print("没有可用的向量数据库")

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()