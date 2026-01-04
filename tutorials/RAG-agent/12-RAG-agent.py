
## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\12-RAG-agent.py

from langchain_core.tools import tool

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv("./env/.env")

from langchain_ollama import OllamaEmbeddings
# 嵌入模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")

from langchain_chroma import Chroma

# 向量库
vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_rag_db"
)

@tool(response_format="content_and_artifact")
def retrive_context(query: str) -> str:
    """Retrieve information to help answer a query"""
    retriveded_docs = vector_store.similarity_search(query, k=2)
    content = '\n\n'.join(
        (f"Source:{doc.metadata}\nContent:{doc.page_content}") for doc in retriveded_docs
    )
    return content, retriveded_docs

model = init_chat_model(
    model_provider="openai",
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.7
)

tools=[retrive_context, ]

SYSTEM_PROMPT = """
    你可以使用信息检索工具，回答用户的问题
"""

# 创建Agent
from langchain.agents import create_agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)

results = agent.invoke(
    {"messages":[{"role":"user", "content":"讲一下Meta官宣收购智能体初创公司Manus"}]}
)

messages = results['messages']
print(f"历史消息： {len(messages)} 条")
for message in messages:
    message.pretty_print()
