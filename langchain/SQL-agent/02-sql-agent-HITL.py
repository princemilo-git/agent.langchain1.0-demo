
## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\langchain\SQL-agent\02-sql-agent-HITL.py

# 1. Select an LLM
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

# 2. Configure the database
import requests, pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
local_path = SCRIPT_DIR / "Chinook.db"

from langchain_community.utilities import SQLDatabase

# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
db = SQLDatabase.from_uri(f"sqlite:///{local_path.resolve().as_posix()}")

# 3. Add tools for database interactions
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

# 4. Use create_agent
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)

# 5. Run the agent
question = "Which genre on average has the longest tracks?"
config = {"configurable": {"thread_id": "1"}}

decisions = [{"type": "reject"}]

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
        # 获取用户输入
        user_input = input("\n请输入您的决定（approve/reject/edit）：").strip().lower()
        if user_input == "approve":
            decisions = [{"type": "approve"}]
        elif user_input == "reject":
            decisions = [{"type": "reject"}]
        elif user_input == "edit":
            decisions = [{"type": "edit"}]
        else:
            print("无效的输入，默认批准") # 请重新输入。")
            decisions = [{"type": "approve"}]
            # continue
    else:
        pass

# 6. Implement human-in-the-loop review
from langgraph.types import Command

for step in agent.stream(
    Command(resume={"decisions": decisions}),
    config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    else:
        pass