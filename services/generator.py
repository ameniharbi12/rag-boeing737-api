# services/generator.py
import os
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from google.genai import Client
import pandas as pd

load_dotenv()


def extract_table_value(chunk: Dict, query_params: dict):
    """
    Extract numeric value(s) from a table chunk based on query_params.
    Returns first matching value and its page.
    """
    df = chunk.get("df")
    if df is None or df.empty:
        return None

    # Normalize columns and string values
    df_norm = df.copy()
    df_norm.columns = [str(c).strip().lower() for c in df.columns]

    mask = pd.Series(True, index=df_norm.index)
    for col, val in query_params.items():
        col = str(col).strip().lower()
        if col not in df_norm.columns:
            return None
        mask &= df_norm[col].astype(str).str.strip() == str(val)

    if mask.any():
        row = df_norm.loc[mask].iloc[0]
        value_col = df_norm.columns[-1]  # assume last column contains the desired numeric
        page = chunk.get("pages", [None])[0]
        return row[value_col], page

    return None


class Generator:
    """
    Multimodal answer generator using Google GenAI.
    Supports:
      - Direct table numeric extraction
      - LLM grounded text answers
      - OCR text from images
      - Page references
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        self.client = Client(api_key=api_key)

        # Select first available preferred model
        available_models = [m.name for m in self.client.models.list()]
        preferred_models = [
            "models/gemini-2.5-flash",
            "models/gemini-flash-latest",
            "models/gemini-2.0-pro-exp-02-05",
        ]
        self.model_name = next((m for m in preferred_models if m in available_models), None)

        if not self.model_name:
            raise ValueError(f"No preferred models available. Found: {available_models}")

        print(f"✅ Using GenAI model: {self.model_name}")

    # -----------------------------------
    # Build prompt
    # -----------------------------------
    def _create_prompt(self, query: str, context: str) -> str:
        return f"""You are an expert aviation technical assistant.
Answer the question using ONLY the information in the context below.

QUESTION: {query}

CONTEXT:
{context}

GUIDELINES:
- Answer concisely in 1–2 sentences.
- Use only information explicitly in the context.
- Extract numeric values exactly; do NOT calculate.
- Include page numbers as (Page X) or (Pages X, Y) if multiple.
- Do not speculate or add external info.

ANSWER:
"""

    # -----------------------------------
    # Main generator
    # -----------------------------------
    def generate_answer(
        self, query: str, retrieved_chunks: List[Dict], numeric_query: dict = None
    ) -> Tuple[str, List[int]]:
        """
        Returns (answer_text, list_of_pages_used)
        """

        # 1️⃣ Direct numeric table extraction
        if numeric_query:
            for chunk in retrieved_chunks:
                if chunk.get("df") is not None:
                    result = extract_table_value(chunk, numeric_query)
                    if result:
                        value, page = result
                        return f"The value is {value} (Page {page})", [page]

        # 2️⃣ Build context for LLM
        top_chunks = sorted(retrieved_chunks, key=lambda c: c.get("score", 0), reverse=True)[:5]

        context_lines = []
        for c in top_chunks:
            page = c.get("pages", [None])[0]
            text = c.get("text", "").strip()
            # include OCR text if chunk is image
            if c.get("chunk_type") == "image" and not text:
                text = "<OCR text unavailable>"
            context_lines.append(f"[Source - Page {page}]\n{text}")

        context = "\n\n".join(context_lines)
        prompt = self._create_prompt(query, context)

        # 3️⃣ Call LLM
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        answer = response.text.strip()

        # 4️⃣ Extract page references from answer
        pages_raw = re.findall(r'\(?Pages?\s*[: ]?\s*([\d, ]+)\)?', answer, re.IGNORECASE)
        pages = []
        for p_str in pages_raw:
            for pp in p_str.split(","):
                pp = pp.strip()
                if pp.isdigit():
                    pages.append(int(pp))

        # Filter pages that exist in retrieved chunks
        valid_pages = set()
        for c in top_chunks:
            for p in c.get("pages", []):
                if p in pages:
                    valid_pages.add(p)

        return answer, sorted(valid_pages)
