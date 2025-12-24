## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\01-agent-basic.py

# agent基本用法-流式输出，message by message 和 token by token两种方式

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

# 1. values
for event in agent.stream(
    {"messages":[{
        "role":"user",
        "content":"what is the weather in Beijing"}]
    },
    stream_mode="values" # 消息输出
):
    messages = event["messages"]
    print(f"历史消息： {len(messages)} 条")
    for message in messages:
        #     message.pretty_print()
        messages[len(messages)-1].pretty_print()

# 2. messages
for chunk in agent.stream(
    {"messages":[{
        "role":"user",
        "content":"what is the weather in Beijing"}]
    },
    stream_mode="messages" # token by token
):
    print( chunk[0].content, end="\n")


#
# It's sunny in Beijing~
# The
#  weather in
#  Beijing is sunny
# !
#  😊