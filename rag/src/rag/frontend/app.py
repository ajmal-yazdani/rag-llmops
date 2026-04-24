import os
import warnings
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parents[3] / ".env"
load_dotenv(dotenv_path=env_path)

# Suppress authlib deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="authlib")

API_URL = os.getenv("API_URL", "http://localhost:8000/rag/query")


def layout() -> None:
    st.markdown("# RAGnimals")
    st.markdown("Ask a question about different animals")

    text_input = st.text_input(label="ask a question")

    if st.button("send") and text_input.strip() != "":
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    API_URL,
                    json={"prompt": text_input},
                    timeout=120.0,  # 120 second timeout for AI model responses
                )
                response.raise_for_status()
                data = response.json()

                # Handle both error string and success dict responses
                if isinstance(data, str):
                    st.error(f"Agent error: {data}")
                else:
                    st.markdown("## Question:")
                    st.markdown(text_input)

                    st.markdown("## Answer:")
                    st.markdown(data["answer"])

                    st.markdown("## Source:")
                    st.markdown(data["filepath"])
            except httpx.TimeoutException:
                st.error("Request timed out. Please try again.")
            except httpx.HTTPError as e:
                st.error(f"HTTP error occurred: {e}")
            except (KeyError, ValueError) as e:
                st.error(f"Invalid response format: {e}")


if __name__ == "__main__":
    layout()
