import os
from dotenv import load_dotenv
from rich import print
from langchain_groq import ChatGroq

load_dotenv()

# Connect to the Groq API (using the model we know works!)
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# Send a test message
print("[bold cyan]Sending message to Groq...[/bold cyan]")
response = llm.invoke("is astrology a barnhum effect only?")

# Print the result using rich
print(f"\n[bold green]AI Response:[/bold green] {response.content}")
