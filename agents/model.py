from state import AgenState
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from tools import tools
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0).bind_tools(tools=tools)

def ai(state: AgenState) -> AgenState:
    systemPrompt = SystemMessage(content="""\
You are DocuMentor — an expert AI assistant for answering questions strictly based on the user's uploaded PDF documents.

Instructions:
- For every user question, always use the `search1` tool to search the document.
- When calling the tool, always provide:
    - 'query': the user's question or keywords.
    - 'supabase_path': the folder path for the document's vector index files. This is always provided in your state as 'supabase_path'.
- Never answer from your own knowledge. Only use the tool's results to answer.
- If the tool returns no results, politely inform the user that no relevant information was found in the document.
- Example tool call:
    search1({
        "query": "What is absorption?",
        "supabase_path": "<the value from your state>"
    })
""")
    result = llm.invoke([systemPrompt] + state["messages"])
    return {"messages": state["messages"] + [result]}

if __name__=="__main__":
    from langchain_core.messages import HumanMessage
    print(ai({"messages":[HumanMessage(content="hey")]}))




