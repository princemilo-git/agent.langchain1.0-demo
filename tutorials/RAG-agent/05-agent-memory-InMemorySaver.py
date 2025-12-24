## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\05-agent-memory-InMemorySaver.py

# 消息列表的内存管理
# 通过 config 实现多会话管理

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# 创建Agent
agent = create_agent(
    model="ollama:deepseek-r1:1.5b",
    checkpointer=checkpointer
)

config = {"configurable":{"thread_id":"1"}}

# 第一轮问答
# 问
results = agent.invoke(
    {"messages":[{"role":"user", "content":"来一首苏轼的宋词"}]},
    config=config
)

# 答
messages = results['messages']
print(f"历史消息： {len(messages)} 条")
for message in messages:
    message.pretty_print()

# 第二轮问答
# 问

results = agent.invoke({
    "messages":[{"role":"user", "content":"再来"}]},
    config=config
)

# 答
messages = results['messages']
print(f"历史消息： {len(messages)} 条")
for message in messages:
    message.pretty_print()