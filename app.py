"""
Streamlit Chat UI for the RAG pipeline.

Usage:
    streamlit run app.py
"""

import asyncio

import streamlit as st

from src.agent.rag_agent import run_agent_query

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚")
st.title("📚 RAG Document Q&A")
st.caption("Ask questions about your documents — powered by Gemini + ChromaDB")

# Shared event loop so agent keeps session memory across questions
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = st.session_state.loop.run_until_complete(run_agent_query(question))
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
