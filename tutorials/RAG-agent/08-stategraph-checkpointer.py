# langgraph 状态管理器 checkpointer 详细解析

## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\RAG-agent\08-stategraph-checkpointer.py

# checkpointer：检查点管理器。
# checkpoint：检查点，保存对话状态。作用：状态图的总体状态快照。
# thread_id：会话 ID，用于标识不同的对话会话。作用：记忆管理（对话回溯），时间旅行（time travel）、人工在环审批（human-in-the-loop）
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver # checkpointer
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

# 定义状态图的结构
# 表达状态：是整个状态图的状态
class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

# 定义节点
def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}

# 构建状态图
# workflow = StateGraph(
#     [
#         (START, "a", node_a),
#         ("a", "b", node_b),
#         ("b", END, None),
#     ],
#     checkpointer=InMemorySaver(),
# )
workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# 添加检查点管理器
# workflow.add_checkpointer(InMemorySaver())
checkpointer = InMemorySaver()

# 编译
graph = workflow.compile(checkpointer)

# 配置
config: RunnableConfig = {
    "configurable": {"thread_id": "1"},
}

# 调用
results = graph.invoke({"foo":""}, config)
print(results)
# {'foo': 'b', 'bar': ['a', 'b']}

## 状态查看
# print(graph.get_state(config))
# {'foo': 'b', 'bar': ['a', 'b']}
# StateSnapshot(
#     values={'foo': 'b', 'bar': ['a', 'b']},
#     next=(),
#     config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0e9181-7b52-6da1-8002-e9487faf2aa6'}},
#     metadata={'source': 'loop', 'step': 2, 'parents': {}},
#     created_at='2026-01-04T02:50:13.472195+00:00',
#     parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0e9181-7b4a-6328-8001-76dcfc6cb049'}},
#     tasks=(),
#     interrupts=()
# )

## 状态查看
for checkpoint_tuple in checkpointer.list(config):
    print("\n")
    print(checkpoint_tuple[2]["step"])
    print(checkpoint_tuple[2]["source"])
    print(checkpoint_tuple[1]["channel_values"])
# 2
# loop
# {'foo': 'b', 'bar': ['a', 'b']}
#
#
# 1
# loop
# {'foo': 'a', 'bar': ['a'], 'branch:to:node_b': None}
#
#
# 0
# loop
# {'foo': '', 'branch:to:node_a': None}
#
#
# -1
# input
# {'__start__': {'foo': ''}}

# print(checkpoint_tuple)
# CheckpointTuple(
# config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0e918e-5cd5-600b-8002-22ea3130d441'}},
# checkpoint={
#   'v': 4,
#   'ts': '2026-01-04T02:55:59.241012+00:00',
#   'id': '1f0e918e-5cd5-600b-8002-22ea3130d441',
#   'channel_versions': {
#       '__start__': '00000000000000000000000000000002.0.10107370784247527',
#       'foo': '00000000000000000000000000000004.0.7050544488448334',
#       'branch:to:node_a': '00000000000000000000000000000003.0.09367218555732337',
#       'bar': '00000000000000000000000000000004.0.7050544488448334',
#       'branch:to:node_b': '00000000000000000000000000000004.0.7050544488448334'
#   },
#   'versions_seen': {
#       '__input__': {},
#       '__start__': {'__start__': '00000000000000000000000000000001.0.1588133495552586'},
#       'node_a': {'branch:to:node_a': '00000000000000000000000000000002.0.10107370784247527'},
#       'node_b': {'branch:to:node_b': '00000000000000000000000000000003.0.09367218555732337'}
#   },
#   'updated_channels': ['bar', 'foo'],
#   'channel_values': {'foo': 'b', 'bar': ['a', 'b']}
# },
# metadata={
#   'source': 'loop',
#   'step': 2,
#   'parents': {}},
#   parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f0e918e-5cd5-600a-8001-de950544f69c'}
# },
# pending_writes=[])