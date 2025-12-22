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

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

# Generated vectors of length 768
#
# [-0.015581737, 0.060439598, -0.14651589, -0.08556373, 0.030379577, 0.030669384, 0.048987467, -0.02043358, -0.0006320369, -0.044006452]