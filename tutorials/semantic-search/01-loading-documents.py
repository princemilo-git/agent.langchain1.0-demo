# 语义检索
## 从PDF到向量库，四种语义搜索方法

## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\semantic-search\semantic-search-1-Loading documents.py

## 1. 读取PDF，按照页来管理，Document，List[Document]。
from langchain_community.document_loaders import PyPDFLoader
file_path="./data/langchain.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load();

print(len(docs)) # 4
print(type(docs[0])) # <class 'langchain_core.documents.base.Document'>
# print(docs[0])
print(f"{docs[0].page_content[:200]}\n") # page_content=' ... '
print(docs[0].metadata)
# metadata={
# 'producer': 'Skia/PDF m94',
# 'creator': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) marktext/0.17.1 Chrome/94.0.4606.81 Electron/15.4.0 Safari/537.36',
# 'creationdate': '2025-12-22T02:32:34+00:00',
# 'moddate': '2025-12-22T02:32:34+00:00',
# 'source': './data/langchain.pdf',
# 'total_pages': 4,
# 'page': 0,
# 'page_label': '1'
# }

