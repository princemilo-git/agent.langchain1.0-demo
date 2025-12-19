# 索引
## 1. 读取PDF，按照页来管理，Document，List[Document]。
## 2. 分割文本，TextSplitter，文本段（chunk），Document，List[Document]。
## 3. 向量化：文本段 <=> 向量，需要嵌入模型来辅助。
## 4. 向量库：把多个文本段/向量存到向量库。

# pip install pypdf

## 1. 读取PDF，按照页来管理，Document，List[Document]。