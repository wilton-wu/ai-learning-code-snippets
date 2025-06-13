from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader


def retrieve_documents(query):
    """从向量数据库检索与查询相关的文档

    Args:
        query: 用户查询字符串

    Returns:
        检索到的相关文档列表
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    index_path = Path("faiss_index.db")

    if index_path.exists():
        # 加载已保存的向量库
        db = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        # 加载文档并创建向量库
        loader = PyMuPDFLoader("knowledge_base.pdf")
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )

        texts = text_splitter.create_documents([page.page_content for page in pages])

        db = FAISS.from_documents(texts, embeddings)
        # 保存向量库到本地
        db.save_local(index_path)

    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever.invoke(str(query))
