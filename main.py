# main.py

from fastapi import FastAPI
from services.document_processor import DocumentProcessor
from services.retriever import Retriever
from services.generator import Generator
from contextlib import asynccontextmanager

# ------------------------------
# Lifespan: Initialize RAG system
# ------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("📄 Loading PDF and creating chunks (API mode)...")
    
    pdf_path = "data/Boeing_B737_Manual.pdf"
    
    dp = DocumentProcessor()
    pages = dp.extract_pages(pdf_path)
    chunks = dp.create_chunks(pages)
    
    app.state.retriever = Retriever(chunks)
    app.state.generator = Generator()
    
    print("✅ RAG system ready!")
    yield
    print("Shutting down RAG system...")

app = FastAPI(title="RAG service for Boeing_B737_Manual", lifespan=lifespan)

# ------------------------------
# API endpoint
# ------------------------------
@app.get("/query")
async def query_rag(query: str):
    retriever: Retriever = app.state.retriever
    generator: Generator = app.state.generator

    # Step 1: Retrieve top chunks
    retrieved_chunks = retriever.retrieve(query, top_k=5)
    if not retrieved_chunks:
        return {"answer": "Answer not found in the provided pages.", "pages": []}

    # Step 2: Generate LLM Answer
    try:
        answer, _ = generator.generate_answer(query, retrieved_chunks)
        answer = answer.strip() if answer else "(No answer returned)"
    except Exception as e:
        return {"answer": f"Error during LLM inference: {str(e)}", "pages": []}

    # Step 3: Collect referenced pages directly
    referenced_pages = sorted({p for c in retrieved_chunks for p in c.get("pages", [])})

    # Step 4: Return JSON response
    return {"answer": answer, "pages": referenced_pages}


# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
