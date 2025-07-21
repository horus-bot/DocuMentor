from state import AgenState
from tools import tools
from langchain_core.messages import ToolMessage

tools_dict = {our_tool.name: our_tool for our_tool in tools}

def take_action(state: AgenState) -> AgenState:
    """Execute tool calls from the LLM's response."""
    last_msg = state["messages"][-1]
    # If using OpenAI function calling, tool_calls is a list of tool call dicts
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        new_messages = state["messages"][:]
        for tool_call in last_msg.tool_calls:
            tool_name = tool_call["name"]
            params = tool_call["args"]
            # Ensure supabase_path is passed if needed
            if "supabase_path" not in params and "supabase_path" in state:
                params["supabase_path"] = state["supabase_path"]
            # Call the tool dynamically
            if tool_name in tools_dict:
                tool_func = tools_dict[tool_name]
                tool_result = tool_func(**params)
                tool_msg = ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                new_messages.append(tool_msg)
        return {"messages": new_messages, "supabase_path": state.get("supabase_path")}
    # If no tool call, just return state
    return state
