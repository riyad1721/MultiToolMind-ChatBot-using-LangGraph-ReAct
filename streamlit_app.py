import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict
from typing import Annotated
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Tools Initialization
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="Query arxiv papers")

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

tavily = TavilySearchResults()

tools = [arxiv, wiki, tavily]

# LLM Initialization
llm = ChatGroq(model="qwen/qwen3-32b")
llm_with_tools = llm.bind_tools(tools=tools)

# LangGraph State Schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Node Definition
def tool_calling_llm(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# LangGraph Build
builder = StateGraph(State)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()

# Streamlit UI
st.set_page_config(page_title="MultiToolMind", layout="centered", page_icon="🧠")

st.title("🧠 MultiToolMind-ChatBot-using-LangGraph-ReAct")
st.markdown(
    """
    This intelligent chatbot integrates **LangGraph** + **ReAct** architecture with access to:
    - 🔍 Wikipedia
    - 📄 Arxiv (Research papers)
    - 📰 Tavily (Latest web search)
    
    **Ask any question combining facts, research, and recent trends!**
    """
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    output = graph.invoke({"messages": st.session_state.chat_history})
    response_message = output["messages"][-1]
    st.session_state.chat_history.append(response_message)

# Display chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
