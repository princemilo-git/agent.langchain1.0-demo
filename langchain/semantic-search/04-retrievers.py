# 语义检索
# 用检索器进行相似度查询

from langchain_ollama import OllamaEmbeddings

# 嵌入模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")

from langchain_chroma import Chroma

# 向量库（文本和向量的映射，发挥例如知识库）
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# 相似度查询
# 通过向量，量化的概念表达语义，找到语义的相似度。余弦相似度。

# 1.基础查询
results = vector_store.similarity_search(
    "how to learn langchain1.0?"
)

for index, result in enumerate(results):
    # print(f"{index + 1}. {result.metadata}")
    print(index)
    print(f"{result.page_content[:100]}\n")

# 2.带分数的查询
results = vector_store.similarity_search_with_score(
    "如何学习 LangChain1.0?"
)

for doc, score in results: # unpacking
    print(score)
    print(f"{doc.page_content[:100]}\n")

# 3.用向量进行相似度查询
vector = embeddings.embed_query(
    "如何学习 LangChain1.0?"
)

results = vector_store.similarity_search_by_vector(vector)

for index, result in enumerate(results):
    print(index)
    print(f"{result.page_content[:100]}\n")

# chain: langchain: 大模型，提示词模板，tools，output，Runnable :runnables::chain
# 4.用检索器进行相似度查询
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

results = retriever.invoke("如何学习 LangChain1.0?")

for index, result in enumerate(results):
    print(index)
    print(f"{result.page_content[:100]}\n")
