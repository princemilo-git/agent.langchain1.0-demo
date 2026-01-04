## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\create-agent\01-agent-demo.py

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# 定义工具
@tool
def search_web(query: str) -> str:
    """搜索网络信息。

    输入：
        query: 搜索内容关键字

    输出：
        搜索结果
    """
    return f"搜索‘{query}’的结果：[这里是模拟的搜索结果]"

@tool
def save_note(content: str) -> str:
    """保存笔记到文件。

    :param content: 笔记内容
    :return: 保存状态
    """
    return f"笔记已保存：{content[:50]}..."

@tool
def get_time() -> str:
    """获取当前时间。

    :param location:
    :return: 当前时间字符串
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S")

tools = [get_time, search_web, save_note]

# 创建Agent
# import os
# model = ChatOpenAI(
#     model="qwen-plus",  # DashScope 支持的模型名
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),  # 必须显式传入或设环境变量
#     temperature=0.7
# )

# 1.基本用法
agent_base = create_agent(
    model="ollama:qwen:1.8b", # 本地模型
    tools=tools,
    system_prompt="创建助手。"
)

print(f"Agent 类型：{type(agent_base).__name__}\n")
#输出：Agent类型：CompiledStateGraph

# 2.system_prompt - 系统提示词
# 详细的系统提示词示例
system_prompt="""你是一个专业的研究助手，名叫 ResearchBot。

你的职责：
1. 帮助用户搜索和整理信息
2. 保存重要的笔记和发现
3. 提供准确的时间信息

工作原则：
- 始终保持专业和准确
- 在搜索后总结关键信息
- 主动询问是否需要保存重要信息
- 使用清晰的中文表达

风格：友好但专业，简洁但完整。
"""

agent_with_prompt = create_agent(
    model="ollama:llama3.2:3b", # qwen:1.8b", # 本地模型
    tools=tools,
    system_prompt=system_prompt
)

# 测试
response = agent_with_prompt.invoke(
    {"messages":[{
        "role":"user",
        "content":"what is the time now"
    }]}
)
print("Agent 响应：")
print(response['messages'][-1].content)
print("Agent 响应完成。\n")

# 3.使用 stream 方式进行流式输出
agent_stream = create_agent(
    model="ollama:llama3.2:3b", # qwen:1.8b", # 本地模型
    tools=tools,
    system_prompt="你是一个助手，请提供详细的回答。"
)

print("开始流式输出：\n")

# for chunk in agent_stream.stream(
#         {"message": ["what is the time now"]},
#         stream_mode="messages"
# ):
for chunk in agent_stream.stream(
    {"messages":[{
        "role":"user",
        "content":"what is the time now"}]
    },
    stream_mode="messages" # token by token
):
    print(chunk[0].content, end="")
print("\n结束流式输出。")

# # 4.持久化和恢复
# from langgraph.checkpoint.memory import MemorySaver
#
# # 创建内存检查点保存器
# checkpointer = MemorySaver
#
# agent_persistent = create_agent(
#     model="ollama:qwen:1.8b", # 本地模型
#     tools=tools,
#     system_prompt="你是一个有记忆的助手，可以记住之前的对话。",
#     checkpointer=checkpointer
# )
#
# # 使用线程 ID 来标识不同的对话会话
# thread_id = "user_123_session_1"
# config = {"configurable": {"thread_id": thread_id}}
#
# # 第一轮对话
# print("=== 开始第一轮对话 ===")
# response1 = agent_persistent.invoke(
#     {"message": ["你好，我的名字是张三"]},
#     config=config
# )
# print(f"User：你好，我的名字是张三\n")
# print(f"Agent 响应：{response1['message'][-1].content}\n")
#
# # 第二轮对话 - Agent 应该记住用户的名字
# print("\n=== 开始第二轮对话 ===")
# response2 = agent_persistent.invoke(
#     {"message": ["你还记得我的名字吗？"]},
#     config=config
# )
# print(f"User：你还记得我的名字吗？\n")
# print(f"Agent 响应：{response2['message'][-1].content}")