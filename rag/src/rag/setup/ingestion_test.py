"""Simple test for Cohere embeddings."""

import os

import cohere
import httpx
from dotenv import load_dotenv


def test_cohere_embeddings() -> None:
    """Test Cohere embedding model with sample text."""
    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError(
            "COHERE_API_KEY not found in environment. "
            "Please add it to your .env file in the project root.",
        )

    # Handle SSL verification for corporate networks
    verify_ssl = os.getenv("COHERE_VERIFY_SSL", "true").lower() != "false"

    # Handle proxy settings
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    proxy_url = https_proxy or http_proxy

    # Build httpx client with appropriate settings
    if not verify_ssl:
        print("⚠️  SSL verification disabled (corporate network mode)\n")

    if proxy_url:
        print(f"ℹ️  Using proxy: {proxy_url}\n")

    # Initialize Cohere client with custom httpx client if needed
    if not verify_ssl or proxy_url:
        httpx_client = httpx.Client(
            verify=verify_ssl,
            proxy=proxy_url if proxy_url else None,
        )
        co = cohere.ClientV2(api_key=api_key, httpx_client=httpx_client)
    else:
        co = cohere.ClientV2(api_key=api_key)

    # Sample texts to embed
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
    ]

    print("Testing Cohere embeddings...\n")
    print("Model: embed-english-v3.0")
    print(f"Number of texts: {len(texts)}\n")

    # Generate embeddings
    response = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"],
    )

    # Display results
    embeddings = response.embeddings.float_
    if embeddings is None:
        raise ValueError("Failed to generate embeddings")

    print("Results:")
    for idx, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
        print(f"\nText {idx + 1}: {text}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

    print("\n✓ Cohere embeddings test completed successfully!")


if __name__ == "__main__":
    test_cohere_embeddings()
