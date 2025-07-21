from typing import List,TypedDict,Sequence,Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgenState(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    supabase_path:str