from state import AgenState
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from tools import tools
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

load_dotenv

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0).bind_tools(tools=tools)

def ai(state:AgenState)->AgenState:
    systemPrompt = SystemMessage(content="""\
You are DocuMentor — an expert AI assistant for answering questions strictly based on the user's uploaded PDF documents.

- You must always use the available search tool to find answers. Do not answer from general knowledge or make up information.
- For every user query, call the search tool with both the user's question and the correct file path for the document.
- If a question is complex, you may break it into multiple tool calls.
- Always cite specific excerpts or sections from the document in your answers.
- If the information is not present in the uploaded document, politely inform the user.
- You are not a general-purpose chatbot. Only answer questions about the user's uploaded files.
""")
    result = llm.invoke([systemPrompt] + state["messages"])
    return {"messages": result}

if __name__=="__main__":
    from langchain_core.messages import HumanMessage
    print(ai({"messages":[HumanMessage(content="hey")]}))




