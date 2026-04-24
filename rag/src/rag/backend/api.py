from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables before any other imports
env_path = Path(__file__).parents[3] / ".env"
load_dotenv(env_path)

from rag.backend.agents import bot_answer  # noqa: E402
from rag.backend.data_models import Prompt, RagResponse  # noqa: E402

app = FastAPI()


@app.post("/rag/query")
async def query_documentation(query: Prompt) -> RagResponse:
    """Query the RAG system using Azure OpenAI."""
    return await bot_answer(query.prompt)
