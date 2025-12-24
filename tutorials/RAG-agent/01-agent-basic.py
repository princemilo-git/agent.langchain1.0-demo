## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\01-agent-basic.py

# agent基本用法

from langchain.agents import create_agent

# 创建Agent
agent = create_agent(
    model="ollama:qwen:1.8b" # 本地模型
)

# print(agent)
# <langgraph.graph.state.CompiledStateGraph object at 0x00000212EAFC3450
# Graph: nodes - edges 网状

# print(agent.nodes)
# {
# '__start__': <langgraph.pregel._read.PregelNode object at 0x000002742992F750>,
# 'model': <langgraph.pregel._read.PregelNode object at 0x000002742992FAD0>
# }

## pregel: Google 2010 发布的技术

results = agent.invoke({
    "messages":[{
        "role":"user",
        "content":"what is the weather in Beijing" # "请搜索 LangChain v1.0 的新特性"
    }]
})

# print( results)
# {'messages': [
#   HumanMessage(content='what is the weather in beijing', additional_kwargs={}, response_metadata={}, id='e2085da5-ed6f-406c-bc43-bca1a01ff1ba'),
#   AIMessage(content="I can't provide real-time information such as current weather. For the most accurate and up-to-date weather in Beijing, I recommend checking a trusted weather website or app like:\n\n- [weather.com](https://www.weather.com)\n- AccuWeather\n- BBC Weather\n- Your local weather service\n\nLet me know if you'd like help understanding weather forecasts or typical climate patterns in Beijing!", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 79, 'prompt_tokens': 23, 'total_tokens': 102, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'qwen-plus', 'system_fingerprint': None, 'id': 'chatcmpl-854a98a7-0e6a-4495-8558-d7fdc87f1c82', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b4a7c-0145-71e2-b8cd-42ba0d2df8b4-0', usage_metadata={'input_tokens': 23, 'output_tokens': 79, 'total_tokens': 102, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})
# ]}

messages = results['messages']
print(f"历史消息： {len(messages)} 条")
for message in messages:
    message.pretty_print()
    # print(message.content)

# 历史消息： 2 条
# ================================ Human Message =================================
#
# 请搜索 LangChain v1.0 的新特性
# ================================== Ai Message ==================================
#
# 截至目前（2024年6月），LangChain 并没有发布名为“v1.0”的传统意义上的版本。实际上，LangChain 团队在 2023 年底至 2024 年初经历了一次重大的架构演进，推出了 **LangChain Next Generation (LangChain NG)**，这通常被视为向“v1.0”理念迈进的重要里程碑。虽然官方并未正式标注为 “v1.0”，但社区和开发者普遍将这一系列更新视为 LangChain 的成熟版本，具备了更稳定、模块化和生产就绪的特性。
#
# 以下是 LangChain 在其新一代架构中引入的关键新特性和改进，可视为“类 v1.0”的核心更新：
#
# ---
#
# ......
#
# ---