from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import vector_stores

# 嵌入模型
embedding = OllamaEmbeddings(model="qwen3-embedding:4b")

# 评分方式
score_measures = [
    "default", # default：'l2'
    "cosine", # 用两个向量的夹角度量相似度， 余弦相似度，1-cos（角度），数值越小相似度越高
    "l2", # 用两个向量的欧式距离，
    "ip", # 用两个向量的内积，和cosine相近
    # "dot_product", # 用两个向量的点积，
]

# 创建向量库和4个集合
persist_dir = "./chroma_score_db"
vector_stores = []
for score_measure in score_measures:
    collection_metadata = {
        "score_measure": score_measure,
    }

    if score_measure == "default":
        collection_metadata = None

    collection_name = f"my_collection_{score_measure}"
    vector_store = Chroma()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_dir,
        collection_metadata=collection_metadata
    )
    vector_stores.append(vector_store)

def indexing(docs):
    print("\n加入文档：")
    for vector_store in vector_stores:
        ids = vector_store.add_documents(docs)
        print(f"\n向量库：{vector_store._collection.name}，\n{ids}")

# docs=[
#     Document(page_content="小米手机很好用"),
#     Document(page_content="我国山西地区盛产小米")
# ]
#
# indexing(docs)

def query_with_score(query):
    for i in range(len(score_measures)):
        result = vector_stores[i].similarity_search_with_score(query)
        print(f"\n搜索：{query}")
        for doc, score in result:
            print(doc.page_content, end='')
            print(f"{score_measures[i]}：{score}")

query_with_score("小米手机怎么样")