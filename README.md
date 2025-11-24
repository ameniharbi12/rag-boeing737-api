# RAG-B737-API

## Overview
The **RAG-B737-API** is a Python-based service that answers questions using the **Boeing 737 technical manual** as the sole knowledge base. The API retrieves relevant information from the manual and generates grounded answers, citing specific pages.

Responses are returned in JSON format:

```json
{
  "answer": "string",
  "pages": [1, 5, 10]
}
```

---

## How the Solution Works
The project is structured into modular services:

### 1. Document Processor
- Parses the PDF manual (text, tables, diagrams).
- Splits content into smaller **chunks** (~200–400 words).
- Preserves **page numbers** for each chunk.
- Prepares chunks for embedding and retrieval.

### 2. Embedder
- Converts chunks into **vector embeddings** using a multimodal embedder (text, tables, images). For example:
  ```python
  from google.genai import Client
  client = Client(api_key="YOUR_GOOGLE_API_KEY")
  result = client.models.embed_content(model="models/embedding-001", contents=[{"text": "example text"}])
  embedding = result.embeddings[0].values
  ```
- Supports weighting by chunk type (table ×3.5, image ×2.0).
- Embeddings are stored in a **FAISS index** for fast retrieval.

### 3. Retriever
- Uses **FAISS** to retrieve the most relevant chunks:
  ```python
  import faiss
  index = faiss.IndexFlatIP(768)
  index.add(embeddings)
  D, I = index.search(query_embedding, k)
  ```
- Supports **keyword overlap boosting**.
- Returns top chunks with **1-based PDF page indices**.

### 4. Generator
- Receives the user query and retrieved chunks.
- Generates concise answers grounded in the retrieved content using **Google GenAI** and direct table extraction if applicable:
  ```python
  from services.generator import Generator

  generator = Generator()
  answer_text, pages = generator.generate_answer(query, retrieved_chunks, numeric_query)
  print(answer_text)
  print(pages)
  ```
- Features:
  - Direct numeric extraction from table chunks.
  - Uses OCR text from images if available.
  - Builds a context prompt with top relevant chunks.
  - LLM answer generation strictly based on the context.
  - Returns a list of page references corresponding to the content used.
- Ensures cited pages match the actual PDF indices, filtering for only valid pages present in retrieved chunks.

### 5. API Server
- Built with **FastAPI**.
- Endpoint:
  ```http
  GET /query?question=<your_question>
  ```
- Returns JSON with `"answer"` and `"pages"`.

---

## Challenges Faced and Solutions

| Challenge | Solution |
|-----------|---------|
| **PDF Parsing** | Used PyPDF2 and custom chunking to extract text, tables, and images while preserving page numbers. |
| **Retrieval Accuracy** | Combined vector similarity with keyword boosting to improve top-K chunk relevance. Tried different embedding models from Hugging Face and Google Gemini to optimize embeddings. |
| **Page Number Mapping** | Stored 1-based PDF indices for accurate page references. |
| **Answer Generation** | Provided only retrieved chunks as context and instructed the LLM to generate grounded answers. |
| **Performance** | Precomputed embeddings and used FAISS indexing for fast retrieval. |
---

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Add your API keys (GOOGLE_API_KEY, OPENAI_API_KEY, etc.)
```

3. Run the API server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. Query the API:
```http
GET /query?question=<your_question>
```

Example response:
```json
{
  "answer": "Answer extracted from the Boeing 737 manual.",
  "pages": [12, 15, 27]
}
```

---

## Repository Structure

```
RAG-B737-API/
│
├── data/                     # PDF file
│   └── Boeing_B737_Manual.pdf
├── services/                 # Core services
│   ├── document_processor.py
│   ├── embedder.py
│   ├── retriever.py
│   └── generator.py
├── main.py                    # API server
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── README.md                  # This file
└── tests/                     # Optional test scripts
```

---

This README is fully **GitHub-ready**, with clear headings, code blocks, tables, and proper formatting for easy readability.

