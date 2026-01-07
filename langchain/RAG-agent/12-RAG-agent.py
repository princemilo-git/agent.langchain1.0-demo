
## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\langchain\RAG-agent\12-RAG-agent.py

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

# 2. Retrieval and Generation

# RAG applications commonly work as follows:
# Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
# Generate: A model produces an answer using a prompt that includes both the question with the retrieved data

# 检索与生成
#
# RAG（检索增强生成）应用通常按以下方式工作：
#
# 检索（Retrieve）：给定用户输入，使用检索器（Retriever）从存储中获取相关的文本片段（splits）。
# 生成（Generate）：模型基于一个包含用户问题和检索到的数据的提示（prompt），生成最终的回答。

# 现在，我们来编写实际的应用逻辑。我们希望创建一个简单的应用程序：接收用户的问题，搜索与该问题相关的文档，将检索到的文档和原始问题一起传递给模型，并返回答案。
#
# 我们将演示以下两种方法：
#
# 一个使用简单工具执行搜索的 RAG 智能体（RAG agent），这是一种通用性较强的标准实现方式。
# 一个 两步 RAG 链（two-step RAG chain），每次查询仅调用一次大语言模型（LLM），这种方法对于简单查询而言快速且高效。

# RAG agents

# One formulation of a RAG application is as a simple agent with a tool that retrieves information. We can assemble a minimal RAG agent by implementing a tool that wraps our vector store:
# RAG 应用的一种实现方式是将其构建为一个带有信息检索工具的简单智能体（agent）。我们可以通过实现一个封装了向量数据库的工具，来组装一个极简的 RAG 智能体。

from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

model = init_chat_model(
    model_provider="openai",
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.7
)

tools=[retrieve_context]

# SYSTEM_PROMPT = """
#     你可以使用信息检索工具，回答用户的问题
# """

# 创建Agent
# from langchain.agents import create_agent
# agent = create_agent(
#     model=model,
#     tools=tools,
#     system_prompt=SYSTEM_PROMPT,
# )
from langchain.agents import create_agent

# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)

# results = agent.invoke(
#     {"messages":[{"role":"user", "content":"讲一下Meta官宣收购智能体初创公司Manus"}]}
# )
#
# messages = results['messages']
# print(f"历史消息： {len(messages)} 条")
# for message in messages:
#     message.pretty_print()

query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
