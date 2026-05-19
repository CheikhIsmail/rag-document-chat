import os
import re
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
import pdfplumber
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
CANDIDATE_K = 30

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunks: List[Dict[str, Any]] = []
index = None

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


def table_to_markdown(table: list) -> str:
    cleaned_rows = []

    for row in table:
        if not row:
            continue

        cleaned_row = [
            str(cell).replace("\n", " ").strip() if cell is not None else ""
            for cell in row
        ]

        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)

    if len(cleaned_rows) < 2:
        return ""

    max_cols = max(len(row) for row in cleaned_rows)

    normalized_rows = [
        row + [""] * (max_cols - len(row))
        for row in cleaned_rows
    ]

    headers = normalized_rows[0]
    data_rows = normalized_rows[1:]

    descriptive_keywords = [
        "bezeichnung",
        "positions",
        "produkt",
        "nummer",
        "besonderheiten",
        "hilfsmittel"
    ]

    last_values = [""] * max_cols
    filled_rows = []

    for row in data_rows:
        filled_row = []

        for col_idx, value in enumerate(row):
            header = headers[col_idx].lower() if col_idx < len(headers) else ""

            if value:
                last_values[col_idx] = value
                filled_row.append(value)
            else:
                if any(keyword in header for keyword in descriptive_keywords):
                    filled_row.append(last_values[col_idx])
                else:
                    filled_row.append("")

        filled_rows.append(filled_row)

    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * max_cols) + " |\n"

    for row in filled_rows:
        markdown += "| " + " | ".join(row) + " |\n"

    row_records = []

    for row in filled_rows:
        parts = []

        for header, value in zip(headers, row):
            header = header.strip()
            value = value.strip()

            if header and value:
                parts.append(f"{header}: {value}")

        if parts:
            row_records.append("- " + " | ".join(parts))

    records_text = "\n".join(row_records)

    return f"{markdown.strip()}\n\nRow records:\n{records_text}".strip()


def extract_pdf_content(file_path: str, filename: str) -> List[Dict[str, Any]]:
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "document": filename,
                    "page": page_number,
                    "type": "text",
                    "text": text.strip()
                })

            tables = page.extract_tables() or []

            if not tables:
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "intersection_tolerance": 5,
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                    }
                ) or []

            for table_index, table in enumerate(tables, start=1):
                table_markdown = table_to_markdown(table)

                if table_markdown.strip():
                    pages.append({
                        "document": filename,
                        "page": page_number,
                        "type": "table",
                        "table_index": table_index,
                        "text": f"Table {table_index} on page {page_number}:\n{table_markdown}"
                    })

    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    result = []
    start = 0
    step = chunk_size - CHUNK_OVERLAP

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            result.append(chunk)

        start += step

    return result


def build_index(pdf_files: List[Tuple[str, str]]) -> Dict[str, Any]:
    global chunks, index

    chunks = []

    for file_path, filename in pdf_files:
        extracted_items = extract_pdf_content(file_path, filename)

        for item in extracted_items:
            item_type = item.get("type", "text")

            if item_type == "table":
                item_chunks = [item["text"]]
            else:
                item_chunks = chunk_text(item["text"])

            for chunk_id, chunk in enumerate(item_chunks):
                chunks.append({
                    "document": item["document"],
                    "page": item["page"],
                    "chunk_id": chunk_id,
                    "type": item_type,
                    "text": chunk
                })

    if not chunks:
        raise ValueError("No readable text or tables found in uploaded PDFs.")

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
        "num_documents": len(pdf_files),
        "table_chunks": sum(1 for chunk in chunks if chunk["type"] == "table"),
        "text_chunks": sum(1 for chunk in chunks if chunk["type"] == "text")
    }


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    global index, chunks

    if index is None:
        raise ValueError("No index available. Please upload PDFs first.")

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)

    candidate_k = min(CANDIDATE_K, len(chunks))
    scores, indices = index.search(query_embedding, candidate_k)

    candidates = []

    for semantic_score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks[idx]

        candidates.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "type": chunk.get("type", "text"),
            "text": chunk["text"],
            "semantic_score": float(semantic_score),
            "score": float(semantic_score)
        })

    if not candidates:
        return []

    tokenized_corpus = [
        candidate["text"].lower().split()
        for candidate in candidates
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

    money_pattern = re.compile(
        r"(\d{1,3}(?:\.\d{3})*,\d{2}\s?(?:€|Euro)|\d+\s?(?:€|Euro)|fünf Euro|zehn Euro)",
        re.IGNORECASE
    )

    price_keywords = [
        "preis", "preise", "betrag", "beträge", "kosten",
        "vergütung", "vergütungen", "pauschale", "pauschalen",
        "versorgungspauschale", "fallpauschale",
        "euro", "eur", "€", "netto", "brutto",
        "vertragspreise", "festbeträge", "kostet", "kosten"
    ]

    query_lower = query.lower()
    is_money_query = any(keyword in query_lower for keyword in price_keywords)

    for candidate, bm25_score in zip(candidates, bm25_scores):
        normalized_bm25 = float(bm25_score / max_bm25)
        semantic_score = candidate["semantic_score"]
        candidate_text = candidate["text"].lower()

        candidate["bm25_score"] = normalized_bm25

        table_boost = 0.10 if candidate.get("type") == "table" else 0.0

        price_keyword_boost = 0.12 if any(
            keyword in candidate_text
            for keyword in price_keywords
        ) else 0.0

        exact_money_boost = 0.25 if money_pattern.search(candidate["text"]) else 0.0

        if is_money_query:
            candidate["rerank_score"] = (
                0.50 * semantic_score
                + 0.25 * normalized_bm25
                + table_boost
                + price_keyword_boost
                + exact_money_boost
            )
        else:
            candidate["rerank_score"] = (
                0.65 * semantic_score
                + 0.25 * normalized_bm25
                + table_boost
                + price_keyword_boost
            )

        candidate["score"] = candidate["rerank_score"]

    candidates = sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return candidates[:top_k]


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]] | None = None
) -> str:
    chat_history = chat_history or []

    context = "\n\n".join([
        f"[Source: {chunk['document']} | Page {chunk['page']} | Type: {chunk.get('type', 'text')}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])

    history_text = "\n".join(
        f"{message.get('role', '').capitalize()}: {message.get('content', '')}"
        for message in chat_history[-6:]
    )

    prompt = f"""
You are a helpful document assistant.

Answer the user's current question only using the provided document context.
Use the conversation history only to understand follow-up questions.
If the answer is not in the document context, say that the document does not contain enough information.

When tables are included, use both the Markdown table and the row records.
The row records are normalized table rows created from the PDF to make merged cells easier to understand.

Conversation history:
{history_text}

Current question:
{question}

Document context:
{context}
"""

    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not configured. Retrieved context is available, but no LLM answer was generated."

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "You answer questions based only on retrieved PDF context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def ask_question(
    question: str,
    chat_history: List[Dict[str, str]] | None = None
) -> Dict[str, Any]:
    retrieved_chunks = retrieve(question)
    answer = generate_answer(question, retrieved_chunks, chat_history)

    retrieval_score = 0.0

    if retrieved_chunks:
        retrieval_score = max(chunk["score"] for chunk in retrieved_chunks)

    sources = [
        {
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "type": chunk.get("type", "text"),
            "score": chunk["score"],
            "semantic_score": chunk.get("semantic_score"),
            "bm25_score": chunk.get("bm25_score"),
            "preview": chunk["text"][:1200],
            "full_text": chunk["text"]
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
        "question": "What prices, fees, reimbursements or Euro amounts are mentioned in the document?",
        "expected_answer": "The answer should extract prices, fees, reimbursements or Euro amounts from text or tables."
    }
]


def run_evaluation() -> List[Dict[str, Any]]:
    results = []

    for item in TEST_QUESTIONS:
        question = item["question"]
        expected_answer = item["expected_answer"]

        retrieved_chunks = retrieve(question)
        retrieval_score = max(
            [chunk["score"] for chunk in retrieved_chunks],
            default=0.0
        )

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "retrieval_score": retrieval_score,
            "top_source": {
                "document": retrieved_chunks[0]["document"] if retrieved_chunks else None,
                "page": retrieved_chunks[0]["page"] if retrieved_chunks else None,
                "type": retrieved_chunks[0].get("type", "text") if retrieved_chunks else None,
                "preview": retrieved_chunks[0]["text"][:1200] if retrieved_chunks else None
            }
        })

    return results