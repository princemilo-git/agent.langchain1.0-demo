from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0.7
)

# for chunk in model.stream(
#     "来一首苏轼的宋词"
# ):
#     print(chunk)

# 流式输出完整句子（逐字显示，但连贯）
print("AI 正在生成回答 流式输出：", end="", flush=True)
print("\n")  # 换行
for chunk in model.stream("来一首苏轼的宋词"):
    print(chunk.content, end="", flush=True)
print("\n" + "AI 完成回答")
print("\n")  # 换行

# 预期效果：像聊天一样逐字打出“出自《赋·春阳》。这首词描写了人……”

# 直接获取完整回答（非流式）
print("AI 正在生成回答 非流式输出：", end="", flush=True)
print("\n")  # 换行
response = model.invoke("来一首苏轼的宋词")
print(response.content)
print("\n" + "AI 完成回答")  # 换行

# 预期效果：出自《水调歌头·明月几时有》。这首词描写了……

## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\hello\01-chatmodel-ollama-local-V0.2.py