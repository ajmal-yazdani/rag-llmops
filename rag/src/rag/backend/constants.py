import os
from pathlib import Path

import mlflow

ROOT_PATH = Path(__file__).parents[1]
DATA_PATH = ROOT_PATH / "data"
PROMPTS_PATH = ROOT_PATH / "prompt_engineering"
VECTOR_DB_PATH = ROOT_PATH / "knowledge_base"

EMBEDDING_MODEL = "text-embedding-3-small"

MODEL = os.getenv("MODEL", "gpt-4o-mini")
LLM_JUDGE = os.getenv("LLM_JUDGE", "gpt-4o-mini")

# Azure OpenAI settings (loaded from .env)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

EXPERIMENT_NAME = "animal-guider-bot"

MLFLOW_DB = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
print(MLFLOW_DB)

mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(EXPERIMENT_NAME)
