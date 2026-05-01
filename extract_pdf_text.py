from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader


PDF_DIR = Path("data_pdf")
TEXT_DIR = Path("data_text")
JSONL_PATH = Path("data_text_pages.jsonl")


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)


def main() -> None:
    TEXT_DIR.mkdir(exist_ok=True)
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {PDF_DIR}")

    total_pages = 0
    total_text_pages = 0

    with JSONL_PATH.open("w", encoding="utf-8") as jsonl_file:
        for pdf_path in pdf_paths:
            reader = PdfReader(str(pdf_path))
            page_texts: list[str] = []

            for page_index, page in enumerate(reader.pages, start=1):
                total_pages += 1
                text = normalize_text(page.extract_text() or "")

                if text:
                    total_text_pages += 1

                page_texts.append(f"===== Page {page_index} =====\n{text}")

                record = {
                    "source": pdf_path.name,
                    "page": page_index,
                    "text": text,
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            output_path = TEXT_DIR / f"{pdf_path.stem}.txt"
            output_path.write_text("\n\n".join(page_texts), encoding="utf-8")
            print(f"Extracted {pdf_path.name} -> {output_path}")

    print(f"Done. PDFs: {len(pdf_paths)}, pages: {total_pages}, pages with text: {total_text_pages}")
    print(f"JSONL dataset: {JSONL_PATH}")


if __name__ == "__main__":
    main()
