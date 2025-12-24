import chromadb

# chromadb 工具-查看删除向量库的collection集合

# 列出向量库的collections和记录
def list_collections(db_path):
    client = chromadb.PersisitentClient(db_path)
    collections = client.list_collections()
    print(f"chromadb:{db_path} 有 {len(collections)} 个向量库")
    for i, collection in enumerate(collections):
        print(f"Collection {i+1}: {collection.name}")
        # print(f"  Metadata: {client.get_collection(name=collection).get()['metadatas']}")
        # print(f"  Count: {client.get_collection(name=collection).count()}")

def delete_collection(db_path, collection_name):
    try:
        client = chromadb.PersisitentClient(db_path)
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"删除(collection_name)时出错: {e}")