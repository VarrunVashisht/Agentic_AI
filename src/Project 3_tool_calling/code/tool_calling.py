# Converted from the companion Jupyter notebook.
# Run from this directory so relative paths resolve as in the notebook.

import os
from pathlib import Path
from dotenv import load_dotenv
from rich import print
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model

# ## Calling LLM

load_dotenv()
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MODEL_NAME

# # Making first tool

@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

add

# # Weather tool

@tool
def get_weather(location: str) -> str:
    """Get the weather for a given location."""
    weather = {"Melbourne" : "32°C, Sunny", "Sydney" : "28°C, Partly Cloudy", "Brisbane" : "30°C, Sunny"}
    
    return f"The weather in {location} is {weather.get(location, 'city unknown')}."

# We have used decorative @tool over the get_weather function so it is no more function but tool.To call @tool, we don't use tool_name()

get_weather

# Calling individual attribute of the tool

get_weather.description
get_weather.name

@tool
def lookup_user(user_id: int) -> str:
    """Lookup a user by ID."""
    users = {1: "Alice", 2: "Bob", 3: "Charlie"}
    
    return f"User ID {user_id} is {users.get(user_id, 'unknown')}."

lookup_user

lookup_user.name
lookup_user.description

RESUME_PATH = Path(__file__).resolve().parent.parent / "resumes"
RESUME_PATH

@tool
def get_resume(name: str) -> str:
    """Read and return the resume of a given person."""
    name = name.strip()
    if not name:
        raise ValueError("Name must not be empty.")
    file_path = RESUME_PATH / f"{name}.txt"

    return file_path.read_text(encoding="utf-8") if file_path.exists() else "resume not found"

get_resume

TOOLS = [add, get_weather, lookup_user, get_resume]

TOOLS
#type(TOOLS)

TOOLS_DATA = {
                "add": add, 
                "get_weather": get_weather, 
                "lookup_user": lookup_user, 
                "get_resume": get_resume
            }

TOOLS_DATA["add"]

def main() -> None:
    """Ask the model to select a tool, run it, and print the final response."""
    llm = init_chat_model(
        MODEL_NAME,
        model_provider="groq",
        temperature=0.0,
    )
    llm_with_tools = llm.bind_tools(list(TOOLS_DATA.values()))
    prompt = input("What would you like to know? ")

    print(f"[bold cyan] User: [/bold cyan] {prompt}")
    message = llm_with_tools.invoke([HumanMessage(content=prompt)])

    if not message.tool_calls:
        print(message.content)
        return

    tool_messages = []
    for tool_call in message.tool_calls:
        tool_name = tool_call["name"]
        tool_arguments = tool_call["args"]
        tool_result = TOOLS_DATA[tool_name].invoke(tool_arguments)
        print(f"[bold green] {tool_name} [/bold green]: {tool_result}")
        tool_messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
        )

    response = llm_with_tools.invoke(
        [
            HumanMessage(content=prompt),
            message,
            *tool_messages,
        ]
    )
    print(f"[bold cyan] Assistant: [/bold cyan] {response.content}")


if __name__ == "__main__":
    main()
