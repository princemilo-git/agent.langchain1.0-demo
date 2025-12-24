
## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\07-agent-memory-PostgreSaver-view.py

# 通过 中间件-PostgreSQL 查看会话消息

from langgraph.checkpoint.postgres import PostgresSaver

DB_URI="postgresql://postgres:root@localhost:5432/postgres"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # 获取所有checkpoint
    checkpointers = checkpointer.list(
        {"configurable":{"thread_id":"1"}}
    )

    for checkpointer in checkpointers:
        messages = checkpointer[1]["channel_values"]["messages"]
        for message in messages:
            message.pretty_print()
        break
        # print(f"{checkpointer.id}")
        # print(f"{checkpointer.config}")
        # print(f"{checkpointer.created_at}")
        # print(f"{checkpointer.updated_at}")
        # print(f"{checkpointer.deleted_at}")
        # print(f"{checkpointer.configurable}")
        # print(f"{checkpointer.configurable_id}")
        # print(f"{checkpointer.configurable_type}")
