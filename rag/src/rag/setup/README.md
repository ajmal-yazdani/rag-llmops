# Setup: PDF to Text Export

## What this script does

- Reads every `.pdf` file from `src/rag/data`.
- Extracts text page by page using `pypdf`.
- Writes one `.txt` file per PDF into the same data folder.
- Saves output as UTF-8 (important for special characters on Windows).

## Run

```powershell
# from repository root: C:\AI\rag-llmops\rag
cd .\src\rag\setup
C:\AI\rag-llmops\rag\src\rag\setup

# run with uv workspace environment
uv run .\pdfs_to_text.py
uv run .\ingestion.py
```

## Notes

- Input files: `src/rag/data/*.pdf`
- Output files: `src/rag/data/<pdf-name>.txt`
- If no PDFs exist in the data folder, the script will finish without output.
