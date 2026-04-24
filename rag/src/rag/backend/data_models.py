from dotenv import load_dotenv
from lancedb.embeddings import get_registry  # type: ignore[import-untyped]
from lancedb.pydantic import LanceModel, Vector  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from rag.backend.constants import EMBEDDING_MODEL

load_dotenv()

# AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION from env
embedding_model = (
    get_registry()
    .get("openai")
    .create(
        name=EMBEDDING_MODEL,
        use_azure=True,
        dim=1536,  # For text-embedding-3-small with Azure OpenAI
    )
)


class Article(LanceModel):
    document_name: str
    filepath: str
    content: str = embedding_model.SourceField()
    embedding: Vector(embedding_model.ndims()) = embedding_model.VectorField()  # type: ignore[valid-type]


class Prompt(BaseModel):
    prompt: str = Field(
        description="prompt from user, if empty consider prompt as missing",
    )


class RagResponse(BaseModel):
    filename: str | None = Field(
        default=None,
        description="filename of the retrieved file without suffix",
    )
    filepath: str | None = Field(
        default=None,
        description="absolute path to the retrieved file",
    )
    answer: str | None = Field(
        description="answer based on the retrieved file, concise but captures essential meaning",
    )
