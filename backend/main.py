from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import build_index, ask_question, run_evaluation

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    chat_history: list[ChatMessage] = []


app = FastAPI(title="RAG Document Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "RAG Document Chat API is running"}


@app.post("/upload")
async def upload_pdfs(
    files: Annotated[list[UploadFile], File(description="Upload one or more PDF files")]
):
    saved_files = []

    try:
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                continue

            file_path = UPLOAD_DIR / file.filename

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            saved_files.append((str(file_path), file.filename))

        if not saved_files:
            raise HTTPException(status_code=400, detail="No valid PDF files uploaded.")

        return build_index(saved_files)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded PDF files: {str(e)}"
        )


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return ask_question(
            question=request.question,
            chat_history=[msg.model_dump() for msg in request.chat_history]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )


@app.get("/evaluate")
def evaluate():
    try:
        return {"results": run_evaluation()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run evaluation: {str(e)}"
        )