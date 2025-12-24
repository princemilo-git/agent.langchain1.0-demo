## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\01-agent-basic.py

# agent基本用法-工具调用

from langchain.agents import create_agent

# 定义工具
def get_weather(city: str) -> str:
    """Get the weather in a given location."""
    return f"It's sunny in {city}~"

tools=[get_weather]

# 创建Agent
agent = create_agent(
    model="ollama:qwen:1.8b", # 本地模型
    tools=tools
)

# print(agent_base)
# <langgraph.graph.state.CompiledStateGraph object at 0x000001A4B2E6AD90>
print(agent_base.nodes)
# {
#   '__start__': <langgraph.pregel._read.PregelNode object at 0x000001A4B344C350>,
#   'model': <langgraph.pregel._read.PregelNode object at 0x000001A4B344C5D0>,
#   'tools': <langgraph.pregel._read.PregelNode object at 0x000001A4B3475F10>
# }

results = agent_base.invoke({
    "messages":[{
        "role":"user",
        "content":"what is the weather in Beijing"
    }]
})

messages = results['messages']
print(f"历史消息： {len(messages)} 条")
for message in messages:
    message.pretty_print()

# "what is the weather in beijing"
# 历史消息： 4 条
# ================================ Human Message =================================
#
# what is the weather in beijing
# ================================== Ai Message ==================================
# Tool Calls:
#   get_weather (call_37ad2ea4e7894b40ae4d2c)
#  Call ID: call_37ad2ea4e7894b40ae4d2c
#   Args:
#     city: beijing
# ================================= Tool Message =================================
# Name: get_weather
#
# It's sunny in beijing~
# ================================== Ai Message ==================================
#
# The weather in Beijing is sunny! 🌞

# "how many people in beijing"
# 历史消息： 2 条
# ================================ Human Message =================================
#
# how many people in beijing
# ================================== Ai Message ==================================
#
# 我无法提供北京当前的人口数量。建议查阅最新的官方统计数据或相关政府网站获取准确信息。