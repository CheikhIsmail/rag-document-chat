import os
from typing import List, Dict, Any, Tuple

import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunks: List[Dict[str, Any]] = []
index = None

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def extract_pdf_text(file_path: str, filename: str) -> List[Dict[str, Any]]:
    pages = []
    doc = fitz.open(file_path)

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "document": filename,
                "page": page_number,
                "text": text
            })

    return pages


def chunk_text(text: str) -> List[str]:
    result = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            result.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return result


def build_index(pdf_files: List[Tuple[str, str]]) -> Dict[str, Any]:
    global chunks, index

    chunks = []

    for file_path, filename in pdf_files:
        pages = extract_pdf_text(file_path, filename)

        for page in pages:
            page_chunks = chunk_text(page["text"])

            for chunk_id, chunk in enumerate(page_chunks):
                chunks.append({
                    "document": page["document"],
                    "page": page["page"],
                    "chunk_id": chunk_id,
                    "text": chunk
                })

    if not chunks:
        raise ValueError("No readable text found in uploaded PDFs.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return {
        "message": "Index built successfully",
        "num_chunks": len(chunks),
        "num_documents": len(pdf_files)
    }


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    global index, chunks

    if index is None:
        raise ValueError("No index available. Please upload PDFs first.")

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]
        results.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results


def generate_answer(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context = "\n\n".join([
        f"[Source: {chunk['document']} | Page {chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    prompt = f"""
You are a helpful document assistant.

Answer the user's question only using the provided context.
If the answer is not in the context, say that the document does not contain enough information.

Question:
{question}

Context:
{context}
"""

    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not configured. Retrieved context is available, but no LLM answer was generated."

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You answer questions based only on retrieved PDF context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def ask_question(question: str) -> Dict[str, Any]:
    retrieved_chunks = retrieve(question)
    answer = generate_answer(question, retrieved_chunks)

    retrieval_score = 0.0
    if retrieved_chunks:
        retrieval_score = max(chunk["score"] for chunk in retrieved_chunks)

    sources = [
        {
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "preview": chunk["text"][:300]
        }
        for chunk in retrieved_chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "retrieval_score": retrieval_score,
        "sources": sources
    }


TEST_QUESTIONS = [
    {
        "question": "What is the main topic of the uploaded documents?",
        "expected_answer": "The answer should identify the main topic discussed in the uploaded PDFs."
    },
    {
        "question": "Which problem or challenge is described in the document?",
        "expected_answer": "The answer should summarize the main problem or challenge from the document."
    },
    {
        "question": "What method, system, or approach is explained?",
        "expected_answer": "The answer should describe the method, system, or approach mentioned in the document."
    },
    {
        "question": "What are the most important details or findings?",
        "expected_answer": "The answer should extract the key details or findings from the document."
    },
    {
        "question": "What conclusion can be drawn from the document?",
        "expected_answer": "The answer should summarize the conclusion or final message of the document."
    }
]


def run_evaluation() -> List[Dict[str, Any]]:
    results = []

    for item in TEST_QUESTIONS:
        question = item["question"]
        expected_answer = item["expected_answer"]

        retrieved_chunks = retrieve(question)
        retrieval_score = max([chunk["score"] for chunk in retrieved_chunks], default=0.0)

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "retrieval_score": retrieval_score,
            "top_source": {
                "document": retrieved_chunks[0]["document"] if retrieved_chunks else None,
                "page": retrieved_chunks[0]["page"] if retrieved_chunks else None,
                "preview": retrieved_chunks[0]["text"][:300] if retrieved_chunks else None
            }
        })

    return results