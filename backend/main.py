from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import build_index, ask_question, run_evaluation

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ChatRequest(BaseModel):
    question: str


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
    # Store uploaded PDF file paths and names
    saved_files = []

    try:
        # Process each uploaded file
        for file in files:
            # Skip files that are not PDFs
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                continue

            # Create destination path inside the uploads directory
            file_path = UPLOAD_DIR / file.filename

            # Read file contents and save to disk
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            # Store file path and original filename for indexing
            saved_files.append((str(file_path), file.filename))

        # Return an error if no valid PDFs were uploaded
        if not saved_files:
            raise HTTPException(
                status_code=400,
                detail="No valid PDF files uploaded."
            )

        # Extract text, generate embeddings, and build the FAISS index
        return build_index(saved_files)

    except ValueError as e:
        # Handle known validation errors (e.g., no readable text)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        # Handle unexpected processing errors
        raise HTTPException(
            status_code=500,
            detail="Failed to process uploaded PDF files."
        )


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        return ask_question(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate answer.")


@app.get("/evaluate")
def evaluate():
    try:
        return {"results": run_evaluation()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to run evaluation.")