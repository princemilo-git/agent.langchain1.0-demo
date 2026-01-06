
## (langchain1.0-py311) D:\Work\Workspace\AIProjects\Agent\langchain1.0-demo>python .\tutorials\multi-agent\01-calendar-agent.py

# 1. Define tools
from langchain.tools import tool

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]


# 2. Create specialized sub-agents
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

## Create a calendar agent
from langchain.agents import create_agent
CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model,
    tools=[get_available_time_slots, create_calendar_event],
    system_prompt=CALENDAR_AGENT_PROMPT,
)

print("\ntest calendar agent")

# query = "Schedule a team meeting next Tuesday at 2pm for 1 hour"
query = "Schedule a team meeting [designteam@example.com] on 2026-1-8 at 2pm for 1 hour"
for step in calendar_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()

## Create an email agent
EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)

print("\ntest email agent")

# query = "Send the design team a reminder about reviewing the new mockups"
query = "Send an mail to  the design team ['designteam@example.com'] about reviewing the new mockups"
for step in email_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()

# 3. Wrap sub-agents as tools
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

# 4. Create the supervisor agent
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)

# 5. Use the supervisor
print("\ntest supervisor agent：Simple single-domain request")
## Example 1: Simple single-domain request
# query = "Schedule a team standup for tomorrow at 9am"
#
# for step in supervisor_agent.stream(
#     {"messages": [{"role": "user", "content": query}]}
# ):
#     for update in step.values():
#         for message in update.get("messages", []):
#             message.pretty_print()

print("\ntest supervisor agent：Complex multi-domain request")
## Example 2: Complex multi-domain request
# query = (
#     "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
#     "and send them an email reminder about reviewing the new mockups."
# )
#
# for step in supervisor_agent.stream(
#     {"messages": [{"role": "user", "content": query}]}
# ):
#     for update in step.values():
#         for message in update.get("messages", []):
#             message.pretty_print()

## Complete working example

## Understanding the architecture

# 6. Add human-in-the-loop review
# from langchain.agents import create_agent
# from langchain.agents.middleware import HumanInTheLoopMiddleware
# from langgraph.checkpoint.memory import InMemorySaver
#
# print("test agent middleware")
#
# calendar_agent = create_agent(
#     model,
#     tools=[create_calendar_event, get_available_time_slots],
#     system_prompt=CALENDAR_AGENT_PROMPT,
#     middleware=[
#         HumanInTheLoopMiddleware(
#             interrupt_on={"create_calendar_event": True},
#             description_prefix="Calendar event pending approval",
#         ),
#     ],
# )
#
# email_agent = create_agent(
#     model,
#     tools=[send_email],
#     system_prompt=EMAIL_AGENT_PROMPT,
#     middleware=[
#         HumanInTheLoopMiddleware(
#             interrupt_on={"send_email": True},
#             description_prefix="Outbound email pending approval",
#         ),
#     ],
# )
#
# supervisor_agent = create_agent(
#     model,
#     tools=[schedule_event, manage_email],
#     system_prompt=SUPERVISOR_PROMPT,
#     checkpointer=InMemorySaver(),
# )
#
# query = (
#     "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
#     "and send them an email reminder about reviewing the new mockups."
# )
#
# config = {"configurable": {"thread_id": "6"}}
#
# interrupts = []
# for step in supervisor_agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     config,
# ):
#     for update in step.values():
#         if isinstance(update, dict):
#             for message in update.get("messages", []):
#                 message.pretty_print()
#         else:
#             interrupt_ = update[0]
#             interrupts.append(interrupt_)
#             print(f"\nINTERRUPTED: {interrupt_.id}")
#
# for interrupt_ in interrupts:
#     for request in interrupt_.value["action_requests"]:
#         print(f"INTERRUPTED: {interrupt_.id}")
#         print(f"{request['description']}\n")

# INTERRUPTED: 4f994c9721682a292af303ec1a46abb7
# Calendar event pending approval
#
# Tool: create_calendar_event
# Args: {'title': 'Meeting with the Design Team', 'start_time': '2024-06-18T14:00:00', 'end_time': '2024-06-18T15:00:00', 'attendees': ['design team']}
#
# INTERRUPTED: 2b56f299be313ad8bc689eff02973f16
# Outbound email pending approval
#
# Tool: send_email
# Args: {'to': ['designteam@example.com'], 'subject': 'Reminder: Review New Mockups Before Meeting Next Tuesday at 2pm', 'body': "Hello Team,\n\nThis is a reminder to review the new mockups ahead of our meeting scheduled for next Tuesday at 2pm. Your feedback and insights will be valuable for our discussion and next steps.\n\nPlease ensure you've gone through the designs and are ready to share your thoughts during the meeting.\n\nThank you!\n\nBest regards,\n[Your Name]"}

# from langgraph.types import Command
#
# resume = {}
# for interrupt_ in interrupts:
#     if interrupt_.id == "2b56f299be313ad8bc689eff02973f16":
#         # Edit email
#         edited_action = interrupt_.value["action_requests"][0].copy()
#         edited_action["arguments"]["subject"] = "Mockups reminder"
#         resume[interrupt_.id] = {
#             "decisions": [{"type": "edit", "edited_action": edited_action}]
#         }
#     else:
#         resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}
#
# interrupts = []
# for step in supervisor_agent.stream(
#     Command(resume=resume),
#     config,
# ):
#     for update in step.values():
#         if isinstance(update, dict):
#             for message in update.get("messages", []):
#                 message.pretty_print()
#         else:
#             interrupt_ = update[0]
#             interrupts.append(interrupt_)
#             print(f"\nINTERRUPTED: {interrupt_.id}")


# # 7. Advanced: Control information flow
# ## Pass additional conversational context to sub-agents
# from langchain.tools import tool, ToolRuntime
#
# @tool
# def schedule_event(
#     request: str,
#     runtime: ToolRuntime
# ) -> str:
#     """Schedule calendar events using natural language."""
#     # Customize context received by sub-agent
#     original_user_message = next(
#         message for message in runtime.state["messages"]
#         if message.type == "human"
#     )
#     prompt = (
#         "You are assisting with the following user inquiry:\n\n"
#         f"{original_user_message.text}\n\n"
#         "You are tasked with the following sub-request:\n\n"
#         f"{request}"
#     )
#     result = calendar_agent.invoke({
#         "messages": [{"role": "user", "content": prompt}],
#     })
#     return result["messages"][-1].text
#
# ## Control what supervisor receives
# import json
#
# @tool
# def schedule_event(request: str) -> str:
#     """Schedule calendar events using natural language."""
#     result = calendar_agent.invoke({
#         "messages": [{"role": "user", "content": request}]
#     })
#
#     # Option 1: Return just the confirmation message
#     return result["messages"][-1].text
#
#     # Option 2: Return structured data
#     # return json.dumps({
#     #     "status": "success",
#     #     "event_id": "evt_123",
#     #     "summary": result["messages"][-1].text
#     # })
#
#
# # 8. Key takeaways