from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import InMemorySaver # for RAM based memeory
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool

load_dotenv()

LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

class chatbotstate(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# tool
search_tool = DuckDuckGoSearchRun(region="us-en")


# @Tool
# def calculate_tool() 
def chat_node(state: chatbotstate):
    chatmessages = state["messages"]
    response = LLM.invoke(chatmessages)
    return {'messages': [response]}
    
graph = StateGraph(chatbotstate)
graph.add_node("chat_node", chat_node)
graph.add_tool_node("search_tool", search_tool, tools_condition)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

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
 