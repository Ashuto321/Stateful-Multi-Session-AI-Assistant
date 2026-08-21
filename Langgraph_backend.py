from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, Literal, Optional, Dict, Any
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import InMemorySaver # for RAM based memeory
import sqlite3
import tempfile
import os
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

LLM = ChatGroq(model="qwen/qwen3.6-27b", temperature=0)

# adding the rag
_thread_retriever: Dict[str, Any] = {}
_thread_metadata: Dict[str, dict] = {}


# fetching the retriever data
# reterining the data if the retriever has it for that particular thread.

def _get_retriever(thread_id: Optional[str]):
    "fetch the thread id for a retriever if exists"
    if thread_id and thread_id in _thread_retriever:
        return _thread_retriever(thread_id)
    return None


# function injest for rag pipeline like indexing and retrival

def ingest(file_bytes: bytes, thread_id: str, filename: Optional[str] = None)-> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    
    """
    
    #checking i
    if not file_bytes:
        return ValueError("no bites received during injestion")
    
    # else we will create a temp file
    #do import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file: 
        # puuting this file in temporary file
        temp_file.write(file_bytes)
        # saving the path of the temporary file
        temp_path = temp_file.name
    
    
    try:
       
       #loader
       loader = PyPDFLoader(temp_path)
       doc = loader.load()
       
       #splitter
       splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10, separators=["\n\n", "\n", " ", ""])
       chunk = splitter.split_documents(doc)
       
       # embedding generator(with model) and stroing them in vector store
       embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
       vector_store = FAISS.from_documents(chunk, embedding_model) 
       
       #retriever
       retriever = vector_store.as_retriever(search_type= 'similarity', search_kwags={'k':4})
       
       # now for corresponsid retriever we need to store thread id
       _thread_retriever[str(thread_id)] = retriever
       
       # now we need store thread_metadata
       _thread_metadata[str(thread_id)] = {
           "filename": filename or os.path.basename(temp_path),
           "documents": len(doc),
           "chunks": len(chunk),
       }
       
       # now we will be returing
       return {
           "filename": filename or os.path.basename(temp_path),
           "documents": len(doc),
           "chunks": len(chunk)
       }
       
    finally:
          # (finally)means wheather everything succeedes or crashed excute the cleanup code.
          try:
              os.remove(temp_path)    
          except OSError:
              pass
                
    

    
# builtin tool for chatbot
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

# custom tool for the chatbot
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
    
# custom tool for chatbot

@tool
def rag_tool(query):
    """ Retrieve relevant information from the pdf document
        use this tool when user ask for factual/conceptual questions
        that might be answered from the loaded document by the user
    """
    

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
 
#  helper function for cheking if thread 