from pathlib import Path

ROOT_PATH = Path(__file__).parents[1]
DATA_PATH = ROOT_PATH / "data"
PROMPTS_PATH = ROOT_PATH / "prompt_engineering"
VECTOR_DB_PATH = ROOT_PATH / "knowledge_base"

# from cohere
EMBEDDING_MODEL = "embed-english-v3.0"

MODEL = "openrouter:openai/gpt-4.1-nano"
LLM_JUDGE = "openrouter:/openai/gpt-4.1-nano"

EXPERIMENT_NAME = "animal-guider-bot"
