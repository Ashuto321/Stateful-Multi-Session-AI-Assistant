import streamlit as st
# lets import the chatbot object form the langgraph here
from Langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# for automating the threadid
import uuid

# *************************************************utility_function***********************************
# thread_id_generating_function
def generate_thread_id():
    thread_id = str(uuid.uuid4())
    return thread_id

# we will create a new chat with new threadid
def reset_chat():
    thread_id = generate_thread_id() # generating new thread
    st.session_state['thread_id'] = thread_id   #stroing thread in session state
    add_thread(st.session_state['thread_id'])  # thread is also getting reset
    st.session_state['message_history'] = [] # initialing message history as new
    
# adding the chat threads innto the chat_thread_list
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)
        
# for getting the conversation 
def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    return state.values.get("messages", [])
# ***************************************************SESSION MEMORY***********************************
if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]  # session state a dictionary  

# adding thread_id in session
if 'thread_id' not in st.session_state: # curren threads in the chatbot
    st.session_state['thread_id'] = generate_thread_id()

# list to store the thread ids
if 'chat_thread' not in st.session_state: # all the threads in over time
    st.session_state['chat_thread'] = retrieve_all_threads()

# calling with thread id
add_thread(st.session_state['thread_id'])


# ***************************************************slidebar UI*************************************

st.sidebar.title("Langgraph-Chatbot")

if st.sidebar.button("New-chat"):
    reset_chat()

st.sidebar.header("My conversation")

for thread_id in st.session_state['chat_thread'][::-1]:
    if st.sidebar.button(str(thread_id)): # loading a thread inside a button
        st.session_state['thread_id'] = thread_id  #current thread id in the session
        messages = load_conversation(thread_id) #we will get list of messages
        
        # now the problem is that we extracted is bigger and not in the form that can be stored by message_history
        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            
            temp_messages.append({'role': role, 'content': msg.content}) 
            
        st.session_state["message_history"] = temp_messages # saving it into the session history

#***************************************************Main UI*******************************************
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("type here")

if user_input:
    # first append in history
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message("user"):
        st.text(user_input)
        
    # configure:
    # config1 = {"configurable": {"thread_id": st.session_state['thread_id']}}
    
    # for the langsmith we will add another type of config
    config1 = {"configurable": {"thread_id": st.session_state['thread_id']},
               "meta_data": {"thread_id": st.session_state['thread_id']},
               "run_name": "chatbot_run"} 
    
    # invkoing the chatbot object
    # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=config)
    # ai_message = response['messages'][-1].content
    
    # using the st.write_stream in streamlit
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        def ai_stream_only():

            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config1,
                stream_mode="messages"
            ):

                # Tool message → show tool status
                if isinstance(message_chunk, ToolMessage):

                    tool_name = getattr(message_chunk, "name", "tool")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"Using `{tool_name}` …",
                            expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"Using `{tool_name}` …",
                            state="running",
                            expanded=True
                        )

                # AI message → stream assistant tokens
                elif isinstance(message_chunk, AIMessage):

                    if message_chunk.content:
                        yield message_chunk.content

        ai_message = st.write_stream(ai_stream_only())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="Tool finished",
                state="complete",
                expanded=False
            )

        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": ai_message
            }
        )