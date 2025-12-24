from langchain.chat_models import init_chat_model

modelDS = init_chat_model(
    model="ollama:deepseek-r1:1.5b",
    # base_url="http://localhost:11434",
    # temperature=0.7
)

# 流式输出完整句子（逐字显示，但连贯）
print("AI 正在生成回答 流式输出：", end="", flush=True)
print("\n")  # 换行
for chunk in modelDS.stream("来一首李白以自然景观为背景的唐诗"):
    print(chunk.content, end="", flush=True)
print("\n" + "AI 完成回答")
print("\n")  # 换行

modelQwen = init_chat_model(
    model="ollama:qwen:1.8b",
    # base_url="http://localhost:11434",
    # temperature=0.7
)

# 直接获取完整回答（非流式）
print("AI 正在生成回答 非流式输出：", end="", flush=True)
print("\n")  # 换行
response = modelQwen.invoke("来一首李白以描述感情为背景的唐诗")
print(response.content)
print("\n" + "AI 完成回答")  # 换行

## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\hello\01-chatmodel-ollama-local-V1.0.py