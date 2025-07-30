# 🧠 MultiToolMind-ChatBot-using-LangGraph-ReAct

An intelligent, production-ready AI chatbot powered by **LangGraph** and **ReAct**, enhanced with real-time access to **Arxiv**, **Wikipedia**, and **Tavily** search tools.

## 🚀 Features

- 🔍 **Wikipedia Integration** – Get factual background knowledge on any topic.
- 📄 **Arxiv Integration** – Retrieve the latest research papers and summaries.
- 📰 **Tavily Search** – Access real-time web search results for current events and news.
- 🧠 **LangGraph + ReAct Architecture** – Enables dynamic tool calling and decision-making.
- 💬 **Multi-turn Conversational Memory** – Keeps track of your previous queries.
- 🎨 **Streamlit UI** – Clean, modern frontend for interactive chat experience.

## 🛠️ Technologies Used

- [LangGraph](https://github.com/langchain-ai/langgraph) – For graph-based conversational flow.
- [LangChain](https://github.com/langchain-ai/langchain) – Tool abstraction and integration.
- [LangChain Community Tools](https://github.com/langchain-ai/langchain) – Arxiv, Wikipedia, and Tavily tools.
- [ChatGroq + Qwen3-32B](https://groq.com/) – Fast LLM inference with Groq API.
- [Streamlit](https://streamlit.io) – Frontend UI.
- [Python Dotenv](https://pypi.org/project/python-dotenv/) – For environment variable management.

## ⚙️ Environment Variables Required

Create a `.env` file in your project root with the following keys:

```bash
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_API_KEY=your_groq_api_key_here
