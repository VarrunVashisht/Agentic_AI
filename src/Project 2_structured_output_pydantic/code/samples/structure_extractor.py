from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


load_dotenv()

# Connect to the Groq API
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
)


class Resume(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str | None = Field(description="Get email if present, else none")
    years_exp: float = Field(description="Total years of professional experience")
    skills: list[str] = Field(description="List of technical skills mentioned")


def get_resume_path() -> Path:
    candidates = [
        Path.cwd() / "code" / "samples" / "resume_messy.txt",
        Path.cwd() / "samples" / "resume_messy.txt",
        Path(__file__).resolve().parent / "resume_messy.txt",
        Path(__file__).resolve().parent.parent / "code" / "samples" / "resume_messy.txt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("resume_messy.txt not found in the expected directories.")


def main() -> None:
    resume_path = get_resume_path()
    text = resume_path.read_text(encoding="utf-8")

    structured_llm = llm.with_structured_output(Resume)
    result = structured_llm.invoke(
        "Extract the candidate details from the resume below and return them in the required schema.\n\n"
        f"{text}"
    )

    print(result.model_dump())


if __name__ == "__main__":
    main()
