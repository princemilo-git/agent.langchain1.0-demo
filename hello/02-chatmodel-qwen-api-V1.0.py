from langchain.chat_models import init_chat_model
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

print("AI 正在生成回答 非流式输出：")
response = model.invoke("来一首杜甫的唐诗")
print(response.content)
print("\nAI 完成回答")

# python .\hello\hello-02-chatmodel-qwen-api-V1.0.py