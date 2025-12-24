## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\04-agent-memory.py

# 短期记忆管理-手动保存历史消息，并先行调用

from langchain.agents import create_agent

# 创建Agent
agent = create_agent(
    model="ollama:deepseek-r1:1.5b"
)

# 第一轮问答
# 问
results = agent.invoke({
    "messages":[{
        "role":"user",
        "content":"来一首苏轼的宋词"
    }]
})

# 答
messages = results['messages']
print(f"历史消息： {len(messages)} 条")
for message in messages:
    message.pretty_print()

# 短期记忆
his_messages = messages

# 第二轮问答
# 问
message={"role":"user", "content":"再来"}
his_messages.append(message)
results = agent.invoke({
    "messages":his_messages
})

# 答
messages = results['messages']
print(f"历史消息： {len(messages)} 条")
for message in messages:
    message.pretty_print()