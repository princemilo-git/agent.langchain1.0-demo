from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from pyreadline3.console import event

load_dotenv("./env/.env")
# agent基本用法-流式输出，message by message 和 token by token两种方式

# 指定模型
import os
model = ChatOpenAI(
    model="qwen-plus",  # DashScope 支持的模型名
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 必须显式传入或设环境变量
    temperature=0.7
)

# 定义工具
def get_weather(city: str) -> str:
    """Get the weather in a given location."""
    return f"It's sunny in {city}~"

tools=[get_weather]

# 创建Agent
agent_base = create_agent(
    model=model,
    tools=tools,
)

# 1. values
# for event in agent_base.stream(
#     {"messages":[{
#         "role":"user",
#         "content":"what is the weather in Beijing"}]
#     },
#     stream_mode="values" # 消息输出
# ):
#     messages = event["messages"]
#     print(f"历史消息： {len(messages)} 条")
#     # for message in messages:
#     #     message.pretty_print()
#     messages[len(messages)-1].pretty_print()

# 2. messages
for chunk in agent_base.stream(
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