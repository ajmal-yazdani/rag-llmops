from fastapi import FastAPI

from .agents import bot_answer
from .data_models import Prompt, RagResponse
from .middleware import logging_middleware

app = FastAPI()
logging_middleware(app=app)


@app.get("/")
async def status() -> dict[str, str]:
    return {"status": "it works"}


@app.post("/rag/query")
async def query_documentation(query: Prompt) -> RagResponse | str:
    result = await bot_answer(query.prompt)

    return result
