# Your deep agent automatically:
#
# 1. **Planned its approach**: Used the built-in `write_todos` tool to break down the research task.计划其方法：使用内置的write_todos工具分解研究任务。
# 2. **Conducted research**: Called the `internet_search` tool to gather information.进行研究：调用internet_search工具收集信息。
# 3. **Managed context**: Used file system tools (`write_file`, `read_file`) to offload large search results.管理上下文：使用文件系统工具（write_file、read_file）转移大型搜索结果。
# 4. **Spawned subagents** (if needed): Delegated complex subtasks to specialized subagents.共享subagents（如果需要）：将复杂的子任务委托给专门的subagents。
# 5. **Synthesized a report**: Compiled findings into a coherent response.合成报告：将结果编译成连贯的响应。

## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\deep-agents\01-deepagents.py

import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

## Create a search tool
from dotenv import load_dotenv
import os
load_dotenv("./env/.env")

tavily_client = TavilyClient() # api_key="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" #os.getenv("TAVILY_API_KEY")) # (api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

## Create a deep agent
from langchain.chat_models import init_chat_model
model = init_chat_model(
    model_provider="openai",
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.7
)

# System prompt to steer the agent to be an expert researcher
# research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.
#
# You have access to an internet search tool as your primary means of gathering information.
#
# ## `internet_search`
#
# Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
# """
research_instructions = """
你是一个专业的研究员。你的任务是进行彻底的研究并撰写一份完整的报告。

你可以使用以下工具：
- internet_search：用于搜索互联网信息

请确保：
1. 进行全面的搜索来手机信息
2. 验证信息的准确性
3. 组织信息并撰写结构化的报告
"""

agent = create_deep_agent(
    model,
    tools=[internet_search],
    system_prompt=research_instructions
)

## Run the agent
# result = agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})
# # Print the agent's response
# print(result["messages"][-1].content)

for result in agent.stream(
{"messages": [
        {"role": "user",
         "content": "什么是langchain?详细介绍它的功能、用途和主要特点"}
        ]
    },
    stream_mode="values"
):
    result["messages"][-1].pretty_print()