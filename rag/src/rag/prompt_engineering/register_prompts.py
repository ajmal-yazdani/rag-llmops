from typing import Any

from mlflow.genai import register_prompt

from rag.backend.constants import PROMPTS_PATH


def register_prompts(**kwargs: Any) -> None:  # noqa: ANN401
    for filepath in PROMPTS_PATH.glob("*.md"):
        if filepath.name.casefold() == "readme.md":
            continue

        with filepath.open(encoding="utf-8") as file:
            filename = filepath.stem
            prompt = file.read()

        register_prompt(name=filename, template=prompt, **kwargs)


if __name__ == "__main__":
    register_prompts(tags={"author": "ajmaly", "stage": "dev"})
