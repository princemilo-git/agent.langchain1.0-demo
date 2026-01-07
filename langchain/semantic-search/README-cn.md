# 使用 LangChain 构建语义搜索引擎

> 原文链接：Build a semantic search engine with LangChain

## 概述

本教程将帮助你熟悉 LangChain 的以下核心抽象组件：

- 文档加载器（Document Loaders）
- 嵌入模型（Embedding Models）
- 向量存储（Vector Stores）

这些抽象组件旨在支持从（向量）数据库及其他数据源中**检索数据**，以便集成到大语言模型（LLM）的工作流中。它们对于需要在模型推理过程中获取外部数据进行推理的应用至关重要，例如**检索增强生成**（Retrieval-Augmented Generation, RAG）。

在本教程中，我们将围绕一个 PDF 文档构建一个搜索引擎，使其能够根据输入查询检索出 PDF 中语义相似的段落。此外，我们还会在此搜索引擎基础上实现一个最简版的 RAG 应用。

## 核心概念

本指南聚焦于**文本数据的检索**，涵盖以下关键概念：

- 文档与文档加载器
- 文本分割器
- 嵌入（Embeddings）
- 向量存储与检索器（Vector Stores 和 Retrievers）

## 环境准备

### 安装

本教程需要安装 `langchain-community` 和 `pypdf` 包：

<CodeGroup>

```bash pip theme={null}
  pip install langchain-community pypdf
```

```bash
conda install langchain-community pypdf -c conda-forge
```

</CodeGroup>

更多详情请参阅我们的 [安装指南](/oss/python/langchain/install)。

### LangSmith（可选但推荐）

当你使用 LangChain 构建包含多个 LLM 调用步骤的复杂应用时，调试和追踪变得尤为重要。 [**LangSmith**](https://smith.langchain.com) 是官方推荐的调试与监控平台。

1. 访问 LangSmith 并注册账号。
  
2. 设置环境变量以启用追踪：
  
  ```shell
  export LANGSMITH_TRACING="true"
  export LANGSMITH_API_KEY="你的API密钥"
  ```
  
  或在 Notebook 中如下设置：
  
  ```python
  import getpass 
  import os 
  
  os.environ["LANGSMITH_TRACING"] = "true"; 
  os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
  ```
  

## 1. 文档与文档加载器

LangChain 定义了 `Document` 抽象类，用于表示一段文本及其元数据，包含三个属性：

- `page_content`：字符串，表示文本内容；
- `metadata`：字典，包含任意元数据（如来源、页码等）；
- `id`（可选）：文档的唯一标识符。

> 注意：一个 `Document` 对象通常代表原始文档的一个**片段**（chunk）。

你可以手动创建示例文档：

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
```

但更常见的是使用 LangChain 提供的[**文档加载器**](/oss/python/langchain/retrieval#document-loaders)，它支持数百种数据源（如 PDF、网页、数据库等）。

### 加载 PDF 文档

我们将加载一份 Nike 2023 年 10-K 财报 PDF。

```python
from langchain_community.document_loaders import PyPDFLoader  

file_path = "../example_data/nke-10k-2023.pdf"  
loader = PyPDFLoader(file_path)  

docs = loader.load()  

print(len(docs))  plits))  # 107  
```

`PyPDFLoader` 会为 PDF 的每一页生成一个 `Document` 对象。我们可以查看其内容和元数据：

````PyPDFLoader`
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)
```

输出示例：

```
目录
美国证券交易委员会
华盛顿特区 20549
表格 10-K
（勾选一项）
☑ 根据 1934 年证券交易法第 13 或 15(d) 条提交的年度报告
...
{'source': '../example_data/nke-10k-2023.pdf', 'page': 0}
```

### 文本分割（Splitting）

一页 PDF 可能包含太多信息，不利于精确检索。为了提高检索精度，我们需要将长文本**切分为更小的块**。

使用 `RecursiveCharacterTextSplitter` 按字符递归分割：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))  # 输出：514
```

重叠部分（`chunk_overlap`）有助于保留上下文，避免关键信息被切断。

---

## 2. 嵌入（Embeddings）

语义搜索的核心是将文本转换为**数值向量**，然后通过向量相似度（如余弦相似度）进行检索。

LangChain 支持来自 数十家提供商 的嵌入模型。以下是几种常用选择：

### OpenAI（推荐）

```bash
pip install -U "langchain-openai"
```

```python
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### 其他选项（按需安装）

| 提供商 | 安装命令 | 示例代码 |
| --- | --- | --- |
| Google Gemini | `pip install -qU langchain-google-genai` | `GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")` |
| Hugging Face | `pip install -qU langchain-huggingface` | `HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")` |
| Ollama（本地） | `pip install -qU langchain-ollama` | `OllamaEmbeddings(model="llama3")` |
| Cohere | `pip install -qU langchain-cohere` | `CohereEmbeddings(model="embed-english-v3.0")` |
| Fake（测试用） | `pip install -qU langchain-core` | `DeterministicFakeEmbedding(size=4096)` |

### 测试嵌入效果

```python
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])
```

输出示例：

```text
生成的向量长度为 1536

[-0.00858657, -0.03341241, -0.00893678, ...]
```

---

## 3. 向量存储（Vector Stores）

接下来，我们将嵌入后的向量存入**向量数据库**，以支持高效相似性搜索。

LangChain 提供多种 向量存储集成，包括内存型、本地型和云服务型。

### 内存向量存储（适合演示）

```bash
1pip install -U "langchain-core"
```

```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
```

### Chroma（轻量级本地向量库）

```bash
1pip install -qU langchain-chroma
```

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
```

### FAISS（Facebook 开源）

```bash
pip install -qU langchain-community faiss-cpu
```

```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

> 其他支持：Pinecone、Qdrant、Milvus、MongoDB Atlas、PGVector、AstraDB、OpenSearch 等。

### 添加文档到向量库

```python
ids = vector_store.add_documents(documents=all_splits)
```

### 查询示例

#### 基于文本的相似性搜索

```python
1results = vector_store.similarity_search("耐克在美国有多少个配送中心？")
2print(results[0])
```

输出：

```text
page_content='...在美国，NIKE 拥有八个重要的配送中心...'
metadata={'page': 4, 'source': '../example_data/nke-10k-2023.pdf', 'start_index': 3125}
```

#### 异步查询

```python
results = await vector_store.asimilarity_search("耐克是哪年成立的？")
```

#### 返回相似度分数

编辑

```python
results = vector_store.similarity_search_with_score("耐克 2023 年的收入是多少？")
doc, score = results[0]
print(f"相似度得分: {score}")
print(doc.page_content)
```

#### 基于向量的搜索

```python
embedding = embeddings.embed_query("2023 年耐克的利润率受何影响？")
results = vector_store.similarity_search_by_vector(embedding)
```

---

## 4. 检索器（Retrievers）

`VectorStore` 本身不是 `Runnable`，而 `Retriever` 是，因此更适合集成到 LangChain 链（Chain）或 Agent 中。

### 手动构建简易 Retriever

```python
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
    "耐克在美国有多少个配送中心？",
    "耐克是哪年成立的？"
])
```

### 使用内置 `as_retriever()`

更推荐的方式：

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

retriever.batch([...])  # 效果相同
```

支持的 `search_type`：

- `"similarity"`（默认）
- `"mmr"`（最大边际相关性，兼顾多样性）
- `"similarity_score_threshold"`（按阈值过滤）

> Retriever 可轻松集成到 RAG 应用中。