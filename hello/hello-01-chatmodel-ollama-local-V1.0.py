from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="ollama:deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# 流式输出完整句子（逐字显示，但连贯）
print("AI 正在生成回答 流式输出：", end="", flush=True)
print("\n")  # 换行
for chunk in model.stream("来一首李白的唐诗"):
    print(chunk.content, end="", flush=True)
print("\n" + "AI 完成回答")
print("\n")  # 换行

# 直接获取完整回答（非流式）
print("AI 正在生成回答 非流式输出：", end="", flush=True)
print("\n")  # 换行
response = model.invoke("来一首李白的唐诗")
print(response.content)
print("\n" + "AI 完成回答")  # 换行