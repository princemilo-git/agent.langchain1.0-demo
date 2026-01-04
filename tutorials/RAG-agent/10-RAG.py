# 1. 理解 RAG
# - 大模型交互：问题 → 大模型 → 回答（生成式）
# - 基于 RAG 的大模型交互：增强：[问题 + （检索：根据问题找到相似的文本）] → 大模型 → 回答
# - 从大模型的角度看：没有区别。RAG 不是大模型技术，而是如何更好使用大模型的技术。
# - RAG 可以看作是一种提示词工程技术。
#
# 2. 扒一下历史，了解 RAG 的历史地位
# 起源：2020 年 Facebook（Meta）论文
# 《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》
# 时间线回顾：
# 2018年 Google，基于注意力机制的 Transformer 架构，改进 CNN/RNN
# 2018年 OpenAI，GPT-1.0，Generative Pre-Trained Transformer
# 2019年 OpenAI，GPT-2.0
# 2020年 OpenAI，GPT-3.0 —— 对照 RAG 的起源
# 2022年 OpenAI，GPT-3.5，ChatGPT-3.5
# ...
## RAG，近几年热度不减，深度应用场景结合

# 3. 大模型局限：
# 1）公开数据，大模型幻觉
# 2）时效问题，
#
# 4. 解决方案
# 1）大模型不断实时更新， 不可能方案
# 2）微调：用私有数据，新的数据，（标注）局部的训练，垂直行业大模型
# 3）RAG：成本最低，时间短

## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\10-RAG.py

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv("./env/.env")

model = init_chat_model(
    model_provider="openai",
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.7
)

# 创建Agent
from langchain.agents import create_agent
agent = create_agent(
    model=model
)

results = agent.invoke(
    {"messages":[{"role":"user", "content":"讲一下3i/Atlas"}]}
)

messages = results['messages']
print(f"历史消息： {len(messages)} 条")
for message in messages:
    message.pretty_print()
