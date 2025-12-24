# langgraph 是 LangChain 团队推出的用于构建 状态化、可中断、可持久化 Agent 工作流 的库。
# langgraph-checkpoint-postgres 是其官方提供的 PostgreSQL 持久化检查点（checkpoint）后端，用于将对话状态、中间步骤等保存到 PostgreSQL 数据库。
# psycopg 是 Python 连接 PostgreSQL 的官方驱动（v3 版本），langgraph-checkpoint-postgres 依赖它。

# PostgreSQL-本地启动停止，连接退出客户端
# net start postgresql-x64-18
# net stop postgresql-x64-18
# psql -U postgres -h localhost -p 5432
# \q

# postgres=# \dt
# Did not find any tables.
# postgres=# \dt
#                     List of tables
#  架构模式 |         名称          |  类型  |  拥有者
# ----------+-----------------------+--------+----------
#  public   | checkpoint_blobs      | 数据表 | postgres
#  public   | checkpoint_migrations | 数据表 | postgres
#  public   | checkpoint_writes     | 数据表 | postgres
#  public   | checkpoints           | 数据表 | postgres

# 本地环境
# pip install psycopg langgraph-checkpoint-postgre

## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\06-agent-memory-PostgreSaver.py

# 通过 中间件-PostgreSQL 实现多会话管理

from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI="postgresql://postgres:root@localhost:5432/postgres"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # PostgresSaver.create_table(DB_URI)
    # PostgresSaver.drop_table(DB_URI)
    checkpointer.setup() # 创建数据表，创建表结构，只在初次运行

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
        "messages":[{"role":"user", "content":"再来一首"}]},
        config=config
    )

    # 答
    messages = results['messages']
    print(f"历史消息： {len(messages)} 条")
    for message in messages:
        message.pretty_print()