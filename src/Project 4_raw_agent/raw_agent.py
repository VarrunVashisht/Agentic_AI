import os
from dotenv import load_dotenv
from rich import print
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "groq:qwen/qwen3-8b")


@tool
def search(query: str) -> str:
    """Look up a fact (mocked for the demo)."""
    facts = {
        "population of japan": "Japan's population is about 124 million.",
        "population of india": "India's population is about 1.43 billion.",
    }

    query_lower = query.lower()
    for key, value in facts.items():
        if key in query_lower:
            return value

    return "No result found (mocked search)."


TOOLS = {"search": search}


def main() -> None:
    """Run a minimal tool-using agent against a Groq model."""
    llm = init_chat_model(
        MODEL,
        model_provider="groq",
        temperature=0.0,
    )
    llm_with_tools = llm.bind_tools([search])

    prompt = input("What fact would you like to look up? ").strip()
    if not prompt:
        print("[bold yellow]No prompt provided.[/bold yellow]")
        return

    print(f"[bold cyan]User:[/bold cyan] {prompt}")
    message = llm_with_tools.invoke(
        [
            SystemMessage(content="Use the search tool when the user asks for factual lookup."),
            HumanMessage(content=prompt),
        ]
    )

    if not message.tool_calls:
        print(f"[bold cyan]Assistant:[/bold cyan] {message.content}")
        return

    tool_call = message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_arguments = tool_call["args"]
    tool_result = TOOLS[tool_name].invoke(tool_arguments)

    print(f"[bold green]{tool_name}:[/bold green] {tool_result}")
    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"],
    )

    final_response = llm_with_tools.invoke(
        [
            SystemMessage(content="Use the tool result to answer the user clearly."),
            HumanMessage(content=prompt),
            message,
            tool_message,
        ]
    )
    print(f"[bold cyan]Assistant:[/bold cyan] {final_response.content}")


if __name__ == "__main__":
    main()
