from pathlib import Path

import lancedb  # type: ignore[import-untyped]
from lancedb.db import DBConnection  # type: ignore[import-untyped]
from lancedb.table import Table  # type: ignore[import-untyped]

from rag.backend.constants import DATA_PATH, VECTOR_DB_PATH
from rag.backend.data_models import Article


def setup_vector_db(path: str | Path) -> DBConnection:
    Path(path).mkdir(exist_ok=True)
    print(f"📂 Setting up vector database at: {path}")
    vector_db = lancedb.connect(uri=path)
    vector_db.create_table("articles", schema=Article, exist_ok=True)
    print("✓ Vector database initialized\n")
    return vector_db


def ingest_docs_to_vector_db(table: Table) -> None:
    txt_files = list(DATA_PATH.glob("*.txt"))
    print(f"📄 Found {len(txt_files)} text file(s) to ingest\n")

    for file in txt_files:
        print(f"Processing: {file.name}")
        with file.open(encoding="utf-8") as f:
            content = f.read()

        document_name = file.name
        # Delete existing entry if present
        table.delete(f"document_name = '{document_name}'")

        # Add new entry with embeddings
        table.add(
            [
                {
                    "document_name": file.name,
                    "filepath": str(file),
                    "content": content,
                },
            ],
        )
        print(f"  ✓ Ingested: {file.name} ({len(content)} chars)\n")

    # Show final table contents
    df = table.to_pandas()
    print(f"\n📊 Total documents in vector DB: {len(df)}")
    print(f"Documents: {list(df['document_name'])}")


if __name__ == "__main__":
    vector_db = setup_vector_db(VECTOR_DB_PATH)
    ingest_docs_to_vector_db(vector_db["articles"])
