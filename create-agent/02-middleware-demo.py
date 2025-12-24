from langchain.agents.middleware import AgentMiddleware
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import create_openapi_agent_toolkit
from langchain.agents.agent_toolkits import create_openapi_agent
from langchain.agents.agent_toolkits import create_sql_agent
from langchain.agents.agent_toolkits import create_sql_agent_toolkit
from langchain.agents.agent_toolkits import create_vectorstore_agent
from langchain.agents.agent_toolkits import create_vectorstore_agent_toolkit

class loggingMiddleware(AgentMiddleware):
    """记录 Agent 执行日志的中间件。"""
    def __init__(self):
        self.log = []

    def before_step(self, agent, intermediate_steps, **kwargs):
        self.log.append(intermediate_steps)

    def after_step(self, agent, intermediate_steps, output, **kwargs):
        self.log.append(output)