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

print(len(all_splits)) # 4
# print(all_splits[0])
print(f"{all_splits[0].page_content[:200]}\n") # page_content=' ... '
print(all_splits[0].metadata)
# {
# 'producer': 'Skia/PDF m94',
# 'creator': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) marktext/0.17.1 Chrome/94.0.4606.81 Electron/15.4.0 Safari/537.36',
# 'creationdate': '2025-12-22T02:32:34+00:00',
# 'moddate': '2025-12-22T02:32:34+00:00',
# 'source': './data/langchain.pdf',
# 'total_pages': 4,
# 'page': 0,
# 'page_label': '1',
# 'start_index': 0
# }