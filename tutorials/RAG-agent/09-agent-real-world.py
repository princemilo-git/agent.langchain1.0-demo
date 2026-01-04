# Agent 结构化输出、运行时参数、系统提示词，综合应用开发

## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\09-agent-real-world.py

## Build a real-world agent

# Next, build a practical weather forecasting agent that demonstrates key production concepts:

# 1. **Detailed system prompts** for better agent behavior
# 2. **Create tools** that integrate with external data
# 3. **Model configuration** for consistent responses
# 4. **Structured output** for predictable results
# 5. **Conversational memory** for chat-like interactions
# 6. **Create and run the agent** create a fully functional agent

# 1. Define the system prompt 定义系统提示词
SYSTEM_PROMPT = """
You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.

请用中文回答
"""

# 2. Create tools 创建工具
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# 3. Configure your model 配置模型
from langchain.chat_models import init_chat_model

# model = init_chat_model(
#     # "claude-sonnet-4-5-20250929",
#     model="ollama:llama3.2:3b", # qwen:1.8b", # "ollama:deepseek-r1:1.5b", # 本地模型
#     temperature=0.5,
#     timeout=10,
#     max_tokens=1000
# )

from dotenv import load_dotenv
import os

load_dotenv("./env/.env")

model = init_chat_model(
    model_provider="openai",
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 必须显式传入或设环境变量
    temperature=0.7
)

tools=[get_weather_for_location, get_user_location]

# 4. Define response format 结构化输出
from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# 5. Add memory 记忆管理
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# 6. Create and run the agent
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation. 配置 `thread_id`
config = {"configurable": {"thread_id": "1"}}

# 第一轮问答
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )