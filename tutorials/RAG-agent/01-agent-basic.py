from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv("./env/.env")
# agent基本用法

# 指定模型
import os
model = ChatOpenAI(
    model="qwen-plus",  # DashScope 支持的模型名
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 必须显式传入或设环境变量
    temperature=0.7
)

# 创建Agent
agent_base = create_agent(
    model=model,
)

# print(agent_base)
# <langgraph.graph.state.CompiledStateGraph object at 0x00000212EAFC3450
# Graph: nodes - edges 网状

# print(agent_base.nodes)
# {
# '__start__': <langgraph.pregel._read.PregelNode object at 0x000002742992F750>,
# 'model': <langgraph.pregel._read.PregelNode object at 0x000002742992FAD0>
# }

## pregel: Google 2010 发布的技术

results = agent_base.invoke({
    "messages":[{
        "role":"user",
        "content":"what is the weather in beijing" # "请搜索 LangChain v1.0 的新特性"
    }]
})

print( results)

# {'messages': [
#   HumanMessage(content='请搜索 LangChain v1.0 的新特性', additional_kwargs={}, response_metadata={}, id='1e4e8281-3f3e-412a-b630-9bc76de0d706'),
#   AIMessage(content='截至目前，并没有官方发布的 LangChain v1.0 版本。LangChain 是一个持续快速迭代的开源项目，由 Harrison Chase 及其团队维护，主要用于构建基于语言模型的应用程序（如聊天机器人、智能代理、检索增强生成系统等）。截至 2024 年初，LangChain 的版本仍在 0.x 系列（例如 0.1.x 或 0.2.x），尚未正式发布语义化版本为 1.0.0 的里程碑版本。\n\n不过，社区和开发者普遍预期未来的 LangChain v1.0 将带来以下潜在“新特性”或重大改进方向（基于当前发展路线图和 GitHub 上的讨论）：\n\n### 预期中的 LangChain v1.0 新特性（预测与趋势）\n\n1. **更清晰的模块化架构**\n   - 将核心组件（如 Chains、Agents、Retrievers、Callbacks）进一步解耦。\n   - 提供更一致的接口设计，减少冗余类和命名混乱。\n\n2. **标准化的接口与抽象层**\n   - 统一 LLM、ChatModel、Embedding Model 的调用方 式。\n   - 引入更强大的 `Runnable` 接口（已在 0.1+ 版本中引入），支持链式调用 `.bind()`、`.pipe()` 等函数式编程范式。\n\n3. **增强的可观察性与调试能力**\n   - 内建对 tracing、日志记录、性能监控的支持。\n   - 更好地集成 LangSmith（LangChain 团队推出的调试与评估平台）。\n\n4. **Agent 系统的重大升级**\n   - 改进的 ReAct 框架、Plan-and-Execute Agent、Tool Calling 机制。\n   - 支持多智能体协作（Multi-agent workflows）。\n   - 更自然的工具描述与动态工具发现 。\n\n5. **更好的流式输出与异步支持**\n   - 全面支持 async/await 和 streaming response，提升用户体验。\n   - 在 Web 应用中实现逐字输出（token by token）。\n\n6. **向后兼容性承诺**\n   - 作为 1.0 版本的核心意义：API 稳定化，承诺 不再频繁 breaking change。\n   - 提供迁移指南和长期支持（LTS）路径。\n\n7. **文档与学习资源全面更新**\n   - 官方文档重构，增加教程、最佳实践、架构图示。\n   - 更多示例项目（starter templates）和部署指南。\n\n8. **生态系统整合加 强**\n   - 与 LangServe（用于部署 LangChain 应用为 REST API）深度集成。\n   - 支持更多第三方 LLM 平台（如 Anthropic、Google Gemini、Ollama 等）。\n\n---\n\n### 当前推荐使用版本\n\n虽然没有 v1.0，但你可以安装最新的稳定版 LangChain：\n\n```bash\npip install langchain\n```\n\n查看当前版本：\n```python\nimport langchain\nprint(langchain.__version__)\n```\n\n建议关注官方渠道获取未来 v1.0 发布信息：\n\n- 官网: https://www.langchain.com\n- GitHub: https://github.com/langchain-ai/langchain\n- 文档: https://docs.langchain.com/docs/\n- 博客: https://blog.langchain.dev\n\n---\n\n### 总结\n\n尽管 LangChain 尚未发布 v1.0，但其发展方向已明确：**构建一个更稳定、模块化、可观测且易于部署的 LLM 应用开发框架**。一旦 v1.0 正式发布，预计将标志着该项目进入生产就绪（production-ready）的新阶段。\n\n如果你希望了解最新进展，建议订阅其 GitHub 仓库并关注 @hwchase17（创始人）的技术分享。\n\n需要我为你总结当前最新版本（如 0.1.x 或 0.2.x）的重要更新吗？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 840, 'prompt_tokens': 27, 'total_tokens': 867, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'qwen-plus', 'system_fingerprint': None, 'id': 'chatcmpl-f6dcc73a-a9c4-9609-9070-a469a718bc07', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b4a1e-a6a9-7ec0-9723-a471c3d7cd84-0', usage_metadata={'input_tokens': 27, 'output_tokens': 840, 'total_tokens': 867, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})
# ]}

# {'messages': [
#   HumanMessage(content='what is the weather in beijing', additional_kwargs={}, response_metadata={}, id='e2085da5-ed6f-406c-bc43-bca1a01ff1ba'),
#   AIMessage(content="I can't provide real-time information such as current weather. For the most accurate and up-to-date weather in Beijing, I recommend checking a trusted weather website or app like:\n\n- [weather.com](https://www.weather.com)\n- AccuWeather\n- BBC Weather\n- Your local weather service\n\nLet me know if you'd like help understanding weather forecasts or typical climate patterns in Beijing!", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 79, 'prompt_tokens': 23, 'total_tokens': 102, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'qwen-plus', 'system_fingerprint': None, 'id': 'chatcmpl-854a98a7-0e6a-4495-8558-d7fdc87f1c82', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019b4a7c-0145-71e2-b8cd-42ba0d2df8b4-0', usage_metadata={'input_tokens': 23, 'output_tokens': 79, 'total_tokens': 102, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})
# ]}

messages = results['messages']
print(f"历史消息： {len(messages)} 条")
# for message in messages:
    # message.pretty_print()
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
# ### 🔧 1. 模块化与清晰的架构设计
# LangChain NG 将代码库彻底重构，采用更清晰的模块划分：
# - `langchain-core`：包含抽象接口（如 LLM、ChatModel、Tool、Retriever 等）和基础数据结构。
# - `langchain-community`：集成第三方工具（如各种 LLM 提供商、向量数据库等）。
# - `langchain`：高层封装，提供开箱即用的链（Chains）、代理（Agents）等。
# - `langchain-experimental`：实验性功能。
#
# > ✅ 好处：更易维护、减少依赖冲突、支持按需安装。
#
# ---
#
# ### 🤖 2. 统一的接口抽象（Standardized Abstractions）
# 引入更一致的接口设计，例如：
# - 所有语言模型统一通过 `BaseModel` 接口调用。
# - 支持 `stream()`、`ainvoke()`、`abatch()` 等异步和流式方法。
# - 使用 Pydantic v2 进行数据验证和序列化。
#
# > ✅ 好处：提升可组合性，便于构建复杂应用。
#
# ---
#
# ### ⚙️ 3. LangChain Expression Language (LCEL)
# LCEL 是新一代的核心编程范式，允许以声明式方式构建可组合的链：
# ```python
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
#
# prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
# model = ChatOpenAI(model="gpt-3.5-turbo")
# parser = StrOutputParser()
#
# # 使用 LCEL 构建链（自动异步、流式支持）
# chain = prompt | model | parser
#
# # 直接调用
# result = chain.invoke({"topic": "cats"})
# ```
#
# > ✅ 特性：
# > - 自动支持同步/异步/流式输出。
# > - 易于调试和测试。
# > - 天然支持中间步骤的监控和日志记录。
#
# ---
#
# ### 🧠 4. 更强大的 Agent 架构
# - 引入基于函数调用（function calling）的新一代 Agent。
# - 支持 ReAct、Plan-and-Execute、AutoGPT 等多种策略。
# - 工具定义更简单，使用 `@tool` 装饰器即可注册。
# - 支持多智能体协作（multi-agent）场景。
#
# ---
#
# ### 🌐 5. 生产级支持增强
# - 内置对 tracing、monitoring 的支持（集成 LangSmith）。
# - 支持缓存（LLM Cache）、重试机制、超时控制。
# - 提供更完善的文档和类型提示（Type Hints）。
#
# ---
#
# ### ☁️ 6. 与 LangSmith 深度集成
# LangSmith 是 LangChain 官方推出的开发与运维平台，用于：
# - 调试 Chains 和 Agents。
# - 记录调用日志（traces）。
# - 评估模型输出质量。
# - A/B 测试不同提示词或模型。
#
# > ✅ 开发者可在 [https://smith.langchain.com](https://smith.langchain.com) 上使用。
#
# ---
#
# ### 📦 7. 包名变更与安装方式更新
# 由于模块拆分，安装方式有所变化：
# ```bash
# pip install langchain                    # 高层接口
# pip install langchain-openai            # OpenAI 集成
# pip install langchain-community         # 社区集成（如 Pinecone、Weaviate）
# pip install langchain-core              # 核心抽象
# ```
#
# ---
#
# ### 总结：LangChain “v1.0” 级别的关键升级
# | 特性 | 描述 |
# |------|------|
# | ✅ 模块化架构 | 清晰分离 core、community、experimental |
# | ✅ LCEL | 声明式链构建，支持流式/异步 |
# | ✅ 统一接口 | 所有组件遵循标准协议 |
# | ✅ 生产就绪 | 缓存、重试、tracing、监控 |
# | ✅ Agent 增强 | 函数调用、多智能体、策略丰富 |
# | ✅ LangSmith 集成 | 全链路可观测性 |
#
# ---
#
# ### 参考资料
# - 官方文档：[https://python.langchain.com](https://python.langchain.com)
# - LangSmith：[https://smith.langchain.com](https://smith.langchain.com)
# - GitHub 仓库：[https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
#
# 如果你正在开始新项目，建议直接使用最新版 LangChain（>=0.1.0，注意版本号规则已变），并采用 LCEL 和 LangSmith 进行开发，以获得最佳体验。
