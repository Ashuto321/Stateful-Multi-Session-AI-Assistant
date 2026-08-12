from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import InMemorySaver # for RAM based memeory
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

load_dotenv()

LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# search tool for the chatbot
search_tool = DuckDuckGoSearchRun(
    name="internet_search",
    description=(
        "Search the internet for current or up-to-date information. "
        "Use this tool when the user asks about recent events, "
        "news, current information, or information that requires "
        "an internet search."
    )
)

@tool
def calculate_tool(first_num: float, second_num: float, operation: Literal['add','sub','multi','div']) -> dict:
    """
    Calculate two numbers.

    operation:
    - add: addition
    - sub: subtraction
    - multi: multiplication
    - div: division
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "multi":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": "invalid operation. please use 'add', 'sub', 'multi', or 'div'."}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

# combining the tools 
tools = [search_tool, calculate_tool]

# binding the tools with the llm
llm_with_tools = LLM.bind_tools(tools)

class chatbotstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: chatbotstate):
    """
    This node takes the query and either returns the response or redirects it to tools for the response
    """
    chatmessages = state["messages"]
    response = llm_with_tools.invoke(chatmessages)
    return {'messages': [response]}
    
tool_node = ToolNode(tools)
    
graph = StateGraph(chatbotstate)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
# for refined output we will redict the tools response back to llm
graph.add_edge("tools", "chat_node")


# checkpointer = InMemorySaver()
# creating the sqlite db
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=checkpointer)

# for retrieving the chat threads from the db
def retrieve_all_threads():
    all_thread = set() # for unique thread
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config["configurable"]["thread_id"])
        
    return list(all_thread)
 