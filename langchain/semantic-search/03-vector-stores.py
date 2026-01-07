# 语义检索

## 1. 读取PDF，按照页来管理，Document，List[Document]。
from langchain_community.document_loaders import PyPDFLoader
file_path="./data/langchain.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load();

## 1. 分割文本，TextSplitter，文本段（chunk），Document，List[Document]。
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

## 2. 向量化：文本段 <=> 向量，需要嵌入模型来辅助。
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

## 3. 向量库：把多个文本段/向量存到向量库。
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

ids = vector_store.add_documents(documents=all_splits)

print(len(ids))
print(ids)
# 4
# ['368a164e-52f4-4f77-811d-c8de5f9c73f8', '20d671d1-0bbc-49cc-a9bd-1ab73002d264', '960e9b11-6f82-4d26-8dd2-4ce371d40fef', 'c3273730-d436-4ed0-9495-6fea2f8c102a']