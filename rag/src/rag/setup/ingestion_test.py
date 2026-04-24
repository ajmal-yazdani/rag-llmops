"""Simple test for Azure OpenAI embeddings via LanceDB registry."""

from dotenv import load_dotenv
from lancedb.embeddings import get_registry  # type: ignore[import-untyped]

from rag.backend.constants import EMBEDDING_MODEL


def test_azure_openai_embeddings() -> None:
    """Test Azure OpenAI embedding model with sample text."""
    load_dotenv()

    embedding_model = (
        get_registry()
        .get("openai")
        .create(
            name=EMBEDDING_MODEL,
            use_azure=True,
            dim=1536,  # For text-embedding-3-small with Azure OpenAI
        )
    )

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
    ]

    print("Testing Azure OpenAI embeddings...\n")
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"Dimensions: {embedding_model.ndims()}")
    print(f"Number of texts: {len(texts)}\n")

    embeddings = embedding_model.generate_embeddings(texts)  # type: ignore[attr-defined]

    print("Results:")
    for idx, (text, emb) in enumerate(zip(texts, embeddings, strict=True)):
        print(f"\nText {idx + 1}: {text}")
        print(f"Embedding dimension: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")

    print("\n✓ Azure OpenAI embeddings test completed successfully!")


if __name__ == "__main__":
    test_azure_openai_embeddings()
