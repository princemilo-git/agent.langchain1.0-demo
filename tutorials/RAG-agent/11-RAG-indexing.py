
## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\11-RAG-indexing.py

## 1. 读取网页，Document，List[Document]。
from langchain_community.document_loaders import WebBaseLoader
import bs4
import os
import shutil

if os.path.exists("./chroma_rag_db"):
    shutil.rmtree("./chroma_rag_db")

page_url=""

bs4_strainer = bs4.SoupStrainer()

loader = WebBaseLoader(
    web_paths=(page_url, ),
    bs_kwargs={"parse_only": bs4_strainer}
)

docs = loader.load()

print(len(docs)) #
print(type(docs[0])) # <class 'langchain_core.documents.base.Document'>
print(docs[0])

## 1. 分割文本，TextSplitter，文本段（chunk），Document，List[Document]。
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

print(len(all_splits)) #
print(all_splits[0])

## 2. 向量化：文本段 <=> 向量，需要嵌入模型来辅助。
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

## 3. 向量库：把多个文本段/向量存到向量库。
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_rag_db"
)

ids = vector_store.add_documents(documents=all_splits)

print(len(ids))
print(ids) #