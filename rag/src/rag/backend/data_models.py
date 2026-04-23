import os
from typing import Any

# Must configure SSL BEFORE any imports that might create SSL contexts
from dotenv import load_dotenv
from lancedb.embeddings import get_registry  # type: ignore[import-untyped]
from lancedb.pydantic import LanceModel, Vector  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from rag.backend.constants import EMBEDDING_MODEL

load_dotenv()

# Configure SSL verification for corporate networks BEFORE any other imports
verify_ssl = os.getenv("COHERE_VERIFY_SSL", "true").lower() != "false"
if not verify_ssl:
    import ssl

    import httpx

    # Disable SSL verification globally (for corporate networks with SSL inspection)
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]  # noqa: SLF001

    # Monkey-patch httpx.Client to disable SSL verification by default
    _original_httpx_client_init = httpx.Client.__init__

    def _patched_httpx_client_init(
        self: httpx.Client,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        kwargs.setdefault("verify", False)
        _original_httpx_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_httpx_client_init  # type: ignore[method-assign]
    print("⚠️  SSL verification disabled globally (corporate network mode)")


# COHERE_API_KEY will be read from environment by LanceDB
# HTTP_PROXY and HTTPS_PROXY environment variables will be respected automatically
embedding_model = get_registry().get("cohere").create(name=EMBEDDING_MODEL)

VECTOR_DIM = embedding_model.ndims()
VectorType = Vector(VECTOR_DIM)


class Article(LanceModel):
    document_name: str
    filepath: str
    content: str = embedding_model.SourceField()
    embedding: VectorType = embedding_model.VectorField()  # type: ignore[valid-type]


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
