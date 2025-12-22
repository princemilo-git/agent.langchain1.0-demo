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

## 4. 向量库：检索器（Retrievers）VectorStore 本身不是 Runnable，而 Retriever 是，因此更适合集成到 LangChain 链（Chain）或 Agent 中。
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain

# @chain
# def retriever(query: str) -> List[Document]:
#     return vector_store.similarity_search(query, k=1)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

queries = [
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
]

results = retriever.batch(queries)

# 打印结果
for i, (query, docs) in enumerate(zip(queries, results)):
    print(f"\n🔍 Query {i + 1}: {query}")
    if not docs:
        print("   ❌ No documents retrieved.")
        continue
    doc = docs[0]  # 因为 k=1，只取第一个
    print(f"   📄 Content (first 300 chars):\n      {doc.page_content[:300]}...")
    print(f"   🏷️  Metadata: {doc.metadata}")

# [[Document(metadata={'page': 4, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 3125}, page_content='direct to consumer operations sell products through the following number of retail stores in the United States:\nU.S. RETAIL STORES NUMBER\nNIKE Brand factory stores 213 \nNIKE Brand in-line stores (including employee-only stores) 74 \nConverse stores (including factory stores) 82 \nTOTAL 369 \nIn the United States, NIKE has eight significant distribution centers. Refer to Item 2. Properties for further information.\n2023 FORM 10-K 2')],
#  [Document(metadata={'page': 3, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 0}, page_content='Table of Contents\nPART I\nITEM 1. BUSINESS\nGENERAL\nNIKE, Inc. was incorporated in 1967 under the laws of the State of Oregon. As used in this Annual Report on Form 10-K (this "Annual Report"), the terms "we," "us," "our,"\n"NIKE" and the "Company" refer to NIKE, Inc. and its predecessors, subsidiaries and affiliates, collectively, unless the context indicates otherwise.\nOur principal business activity is the design, development and worldwide marketing and selling of athletic footwear, apparel, equipment, accessories and services. NIKE is\nthe largest seller of athletic footwear and apparel in the world. We sell our products through NIKE Direct operations, which are comprised of both NIKE-owned retail stores\nand sales through our digital platforms (also referred to as "NIKE Brand Digital"), to retail accounts and to a mix of independent distributors, licensees and sales')]]

