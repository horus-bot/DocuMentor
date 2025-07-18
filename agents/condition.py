from state import AgenState

def should_continue(state:AgenState):
    """ this is to check wether the tool will be used or not """
    result=state["messages"][-1]
    if  hasattr(result, 'tool_calls') and len(result.tool_calls) > 0:
        return True
    else :
        return False