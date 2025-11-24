# services/document_processor.py

import pdfplumber
import camelot
import pandas as pd
import pytesseract
from PIL import Image
import io
import re
from typing import List, Dict

class DocumentProcessor:
    """
    Robust PDF processor optimized for multimodal RAG:
    - Extracts tables (Camelot + pdfplumber fallback)
    - Extracts text and splits into overlapping chunks
    - Extracts images and OCR text
    - Preserves page numbers
    """

    UNIT_MAP = {
        r'\bfoot\b|\bfeet\b': 'ft',
        r'\bcelsius\b': 'degC',
        r'\bfahrenheit\b': 'degF',
        r'\bpercent\b': 'percent',
        r'°C': 'degC',
        r'°F': 'degF'
    }

    # -------------------------
    # PDF PAGE EXTRACTION
    # -------------------------
    def extract_pages(self, pdf_path: str) -> List[Dict]:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"📄 Processing {total_pages} pages...")

            for page_num, page in enumerate(pdf.pages, start=1):
                page_chunks = []

                # --- 1. Camelot tables (lattice + stream) ---
                tables_found = False
                for flavor in ['lattice', 'stream']:
                    try:
                        tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor=flavor)
                        for t in tables:
                            df = t.df
                            if df is not None and not df.empty:
                                table_text = self._table_to_text(df)
                                page_chunks.append({
                                    "chunk_type": "table",
                                    "text": f"[TABLE]\n{table_text}\n[/TABLE]",
                                    "df": df,
                                    "pages": [page_num]
                                })
                                tables_found = True
                    except Exception as e:
                        print(f"⚠️ Camelot ({flavor}) failed on page {page_num}: {e}")

                # --- 2. Fallback pdfplumber tables only if Camelot found nothing ---
                if not tables_found:
                    for table in page.extract_tables():
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            table_text = self._table_to_text(df)
                            page_chunks.append({
                                "chunk_type": "table",
                                "text": f"[TABLE]\n{table_text}\n[/TABLE]",
                                "df": df,
                                "pages": [page_num]
                            })

                # --- 3. Extract text ---
                text = page.extract_text() or ""
                text = self._clean_text(text)
                if text.strip():
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    for para in paragraphs:
                        page_chunks.append({
                            "chunk_type": "text",
                            "text": para,
                            "df": None,
                            "pages": [page_num]
                        })

                # --- 4. OCR images ---
                for img_info in page.images:
                    try:
                        im = page.to_image(resolution=400)
                        x0, top, x1, bottom = img_info['x0'], img_info['top'], img_info['x1'], img_info['bottom']
                        cropped = im.crop((x0, top, x1, bottom)).original

                        buf = io.BytesIO()
                        cropped.save(buf, format='PNG')
                        raw_bytes = buf.getvalue()
                        cropped.close()

                        ocr_text = pytesseract.image_to_string(Image.open(io.BytesIO(raw_bytes)))
                        if ocr_text.strip():
                            page_chunks.append({
                                "chunk_type": "image",
                                "text": self._clean_text(ocr_text),
                                "df": None,
                                "pages": [page_num],
                                "image_bytes": raw_bytes
                            })
                    except Exception as e:
                        print(f"⚠️ OCR failed on page {page_num}: {e}")

                if page_chunks:
                    pages.extend(page_chunks)

                if page_num % 10 == 0:
                    print(f"   Processed {page_num}/{total_pages} pages")

        print(f"✅ Extracted chunks from {total_pages} unique PDF pages.")
        return pages

    # -------------------------
    # CHUNK CREATION
    # -------------------------
    def create_chunks(self, pages: List[Dict], chunk_size=250, overlap=80) -> List[Dict]:
        """
        Create smaller, overlapping chunks for text, tables, and images.
        Each table is preserved as a separate chunk.
        """
        chunks = []
        chunk_id = 0
        unique_pages_set = set()

        for entry in pages:
            text = entry["text"]
            page_num = entry["pages"][0]
            unique_pages_set.add(page_num)
            chunk_type = entry["chunk_type"]

            # --- Table chunks ---
            if chunk_type == "table" and entry.get("df") is not None:
                df = entry["df"]
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,  # already has [TABLE] tags
                    "pages": [page_num],
                    "chunk_type": "table",
                    "df": df,
                    "image_bytes": entry.get("image_bytes")
                })
                chunk_id += 1
                continue

            # --- Image chunks ---
            if chunk_type == "image":
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "pages": [page_num],
                    "chunk_type": "image",
                    "df": None,
                    "image_bytes": entry.get("image_bytes")
                })
                chunk_id += 1
                continue

            # --- Text chunks ---
            words = text.split()
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_text = " ".join(words[start:end])
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "pages": [page_num],
                    "chunk_type": "text",
                    "df": None,
                    "image_bytes": None
                })
                chunk_id += 1
                start += chunk_size - overlap

        print(f"✂️ Created {len(chunks)} chunks from {len(unique_pages_set)} unique PDF pages.")
        return chunks

    # -------------------------
    # HELPERS
    # -------------------------
    def _table_to_text(self, df: pd.DataFrame) -> str:
        """Convert table to searchable text including numeric values."""
        df_copy = df.fillna("").astype(str)
        columns_str = [str(c) if c is not None else "" for c in df_copy.columns]
        text = " | ".join(columns_str) + "\n"
        for _, row in df_copy.iterrows():
            row_values = [str(v) if v is not None else "" for v in row.values]
            text += " | ".join(row_values) + "\n"
        nums = re.findall(r'\d+\.?\d*', text)
        if nums:
            text += "\nNUMS: " + " ".join(nums)
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Clean text and normalize units."""
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r'\s+', ' ', text)
        for pattern, replacement in self.UNIT_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text.strip()
