import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Document Chat", page_icon="📄", layout="wide")

st.title("📄 RAG Document Chat")
st.write("Upload PDF documents and chat with them using retrieval-augmented generation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Upload and Index Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            files = []

            for uploaded_file in uploaded_files:
                files.append(
                    (
                        "files",
                        (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    )
                )

            with st.spinner("Uploading, extracting text/tables, embedding, and indexing documents..."):
                response = requests.post(f"{BACKEND_URL}/upload", files=files)

            if response.status_code == 200:
                result = response.json()

                st.success("Documents indexed successfully.")
                st.write(f"Documents: {result.get('num_documents')}")
                st.write(f"Chunks: {result.get('num_chunks')}")
                st.write(f"Text chunks: {result.get('text_chunks')}")
                st.write(f"Table chunks: {result.get('table_chunks')}")

                st.session_state.messages = []
            else:
                st.error(response.text)

    st.divider()

    st.header("Evaluation")

    if st.button("Run Evaluation"):
        with st.spinner("Running retrieval evaluation..."):
            response = requests.get(f"{BACKEND_URL}/evaluate")

        if response.status_code == 200:
            results = response.json().get("results", [])

            for i, item in enumerate(results, start=1):
                st.subheader(f"Test Question {i}")
                st.write("**Question:**", item["question"])
                st.write("**Expected Answer:**", item["expected_answer"])
                st.write("**Retrieval Score:**", round(item["retrieval_score"], 4))

                top_source = item.get("top_source")

                if top_source:
                    st.write("**Top Source:**")
                    st.write(
                        f"{top_source.get('document')} — Page {top_source.get('page')} — Type: {top_source.get('type')}"
                    )

                    with st.expander("Show source preview"):
                        st.markdown(top_source.get("preview") or "")
        else:
            st.error(response.text)

    st.divider()

    if st.button("Clear Conversation Memory"):
        st.session_state.messages = []
        st.success("Conversation memory cleared.")


st.header("Chat")


def render_sources(sources):
    if not sources:
        return

    st.subheader("Sources")

    for source in sources:
        source_title = (
            f"{source.get('document')} — Page {source.get('page')} "
            f"| Type: {source.get('type')} "
            f"| Score: {round(source.get('score', 0), 4)}"
        )

        with st.expander(source_title):
            st.markdown(
                source.get("full_text")
                or source.get("preview")
                or ""
            )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            retrieval_score = message.get("retrieval_score")
            sources = message.get("sources", [])

            if retrieval_score is not None:
                st.metric("Retrieval Score", round(retrieval_score, 4))

            render_sources(sources)


question = st.chat_input("Ask a question about the uploaded PDFs...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer..."):
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "question": question,
                    "chat_history": [
                        {
                            "role": message["role"],
                            "content": message["content"]
                        }
                        for message in st.session_state.messages[-6:]
                    ]
                }
            )

        if response.status_code == 200:
            result = response.json()

            answer = result.get("answer", "")
            retrieval_score = result.get("retrieval_score", 0)
            sources = result.get("sources", [])

            st.write(answer)

            if retrieval_score is not None:
                st.metric("Retrieval Score", round(retrieval_score, 4))

            render_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "retrieval_score": retrieval_score,
                "sources": sources
            })

        else:
            st.error(response.text)