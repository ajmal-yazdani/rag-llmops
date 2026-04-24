import os

import httpx
import streamlit as st

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
                    timeout=60.0,  # 60 second timeout for AI model responses
                )
                response.raise_for_status()
                data = response.json()

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
