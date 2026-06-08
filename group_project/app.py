"""Streamlit chatbot for the Day 08 group project."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task10_generation import answer_question


st.set_page_config(page_title="Drug Law RAG", layout="wide")
st.title("Drug Law RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Retrieval")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    st.caption("Hybrid TF-IDF + BM25 + RRF, Jina reranking when available.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    metadata = source.get("metadata", {})
                    st.markdown(
                        f"**{metadata.get('title') or metadata.get('source', 'Source')}**  \n"
                        f"`score={source.get('score', 0.0):.3f}` · "
                        f"`{metadata.get('path', '')}`"
                    )

prompt = st.chat_input("Ask about Vietnamese drug law or the indexed news...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence..."):
            result = answer_question(prompt, top_k=top_k)
        st.markdown(result["answer"])
        with st.expander("Sources", expanded=True):
            for source in result["sources"]:
                metadata = source.get("metadata", {})
                st.markdown(
                    f"**{metadata.get('title') or metadata.get('source', 'Source')}**  \n"
                    f"`score={source.get('score', 0.0):.3f}` · "
                    f"`{metadata.get('path', '')}`"
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
