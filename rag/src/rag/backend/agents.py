import lancedb  # type: ignore[import-untyped]
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag.backend.constants import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_ENDPOINT,
    MODEL,
    VECTOR_DB_PATH,
)
from rag.backend.data_models import RagResponse

vector_db = lancedb.connect(uri=VECTOR_DB_PATH)

azure_client = AsyncAzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    api_key=AZURE_OPENAI_API_KEY,
)

rag_agent = Agent(
    model=OpenAIChatModel(
        MODEL,
        provider=OpenAIProvider(openai_client=azure_client),
    ),
    system_prompt="You are an animal expert who loves helping young pet owners (ages 10-15).",
    output_type=RagResponse,
)


@rag_agent.tool_plain
def retrieve_top_documents(query: str, k: int = 3) -> str:
    """Retrieves top k documents."""
    k = int(k)

    results = _retrieve(query, k)

    return "\n\n".join(
        f"""Filename: {result["metadata"]["source"]}\
        Filepath: {result["metadata"]["filepath"]}\
        Content: {result["page_content"]}"""
        for result in results
    )


def _retrieve(query: str, k: int) -> list[dict]:
    results = vector_db["articles"].search(query=query).limit(k).to_list()
    return [
        {
            "page_content": result["content"],
            "metadata": {
                "source": result.get("document_name", ""),
                "filepath": result.get("filepath", ""),
            },
        }
        for result in results
    ]


async def bot_answer(question: str) -> RagResponse:
    result = await rag_agent.run(question)
    return result.output
