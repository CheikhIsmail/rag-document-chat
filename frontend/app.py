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

            with st.spinner("Uploading, chunking, embedding, and indexing documents..."):
                response = requests.post(f"{BACKEND_URL}/upload", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success("Documents indexed successfully.")
                st.write(f"Documents: {result.get('num_documents')}")
                st.write(f"Chunks: {result.get('num_chunks')}")
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
                    st.write(f"{top_source.get('document')} — Page {top_source.get('page')}")
                    st.caption(top_source.get("preview"))
        else:
            st.error(response.text)


st.header("Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about the uploaded PDFs...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer..."):
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"question": question}
            )

        if response.status_code == 200:
            result = response.json()

            answer = result.get("answer", "")
            retrieval_score = result.get("retrieval_score", 0)
            sources = result.get("sources", [])

            st.write(answer)
            st.metric("Retrieval Score", round(retrieval_score, 4))

            if sources:
                st.subheader("Sources")
                for source in sources:
                    with st.expander(
                        f"{source['document']} — Page {source['page']} | Score: {round(source['score'], 4)}"
                    ):
                        st.write(source["preview"])

            st.session_state.messages.append({"role": "assistant", "content": answer})

        else:
            st.error(response.text)