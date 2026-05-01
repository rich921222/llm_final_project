from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from pypdf import PdfReader


PDF_DIR = Path("data_pdf")
TEXT_DIR = Path("data_text")
JSONL_PATH = Path("data_text_pages.jsonl")
TRANSLATION_CACHE_PATH = Path("translation_cache.json")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)


def has_chinese(text: str) -> bool:
    return CHINESE_PATTERN.search(text) is not None


def load_translation_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_translation_cache(path: Path, cache: dict[str, str]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_to_english(text: str, cache: dict[str, str], sleep_seconds: float) -> str:
    if not text or not has_chinese(text):
        return ""

    if text in cache:
        return cache[text]

    try:
        from deep_translator import GoogleTranslator
    except ImportError as error:
        raise SystemExit(
            "deep-translator is required for --translate-chinese. "
            "Install it with: pip install -r requirements.txt"
        ) from error

    translated = GoogleTranslator(source="auto", target="en").translate(text) or ""
    translated = normalize_text(translated)
    cache[text] = translated

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return translated


def combine_original_and_translation(text: str, translated_text: str) -> str:
    if not translated_text:
        return text

    return f"{text}\n\n[English translation]\n{translated_text}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from lecture PDFs.")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR, help="directory containing PDF files")
    parser.add_argument("--text-dir", type=Path, default=TEXT_DIR, help="directory for per-PDF text outputs")
    parser.add_argument("--jsonl", type=Path, default=JSONL_PATH, help="output JSONL path")
    parser.add_argument(
        "--translate-chinese",
        action="store_true",
        help="translate pages containing Chinese to English and append the translation to the searchable text",
    )
    parser.add_argument(
        "--translation-cache",
        type=Path,
        default=TRANSLATION_CACHE_PATH,
        help="cache file for page translations",
    )
    parser.add_argument(
        "--translation-sleep",
        type=float,
        default=0.2,
        help="seconds to wait between Google Translate requests",
    )
    args = parser.parse_args()

    args.text_dir.mkdir(exist_ok=True)
    pdf_paths = sorted(args.pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        raise SystemExit(f"No PDF files found in {args.pdf_dir}")

    total_pages = 0
    total_text_pages = 0
    total_translated_pages = 0
    translation_cache = load_translation_cache(args.translation_cache) if args.translate_chinese else {}

    with args.jsonl.open("w", encoding="utf-8") as jsonl_file:
        for pdf_path in pdf_paths:
            reader = PdfReader(str(pdf_path))
            page_texts: list[str] = []

            for page_index, page in enumerate(reader.pages, start=1):
                total_pages += 1
                text = normalize_text(page.extract_text() or "")

                if text:
                    total_text_pages += 1

                translated_text = ""
                if args.translate_chinese:
                    translated_text = translate_to_english(text, translation_cache, args.translation_sleep)
                    if translated_text:
                        total_translated_pages += 1

                searchable_text = combine_original_and_translation(text, translated_text)
                page_texts.append(f"===== Page {page_index} =====\n{searchable_text}")

                record = {
                    "source": pdf_path.name,
                    "page": page_index,
                    "text": searchable_text,
                    "original_text": text,
                }
                if translated_text:
                    record["translated_text"] = translated_text
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            output_path = args.text_dir / f"{pdf_path.stem}.txt"
            output_path.write_text("\n\n".join(page_texts), encoding="utf-8")
            print(f"Extracted {pdf_path.name} -> {output_path}")

    if args.translate_chinese:
        save_translation_cache(args.translation_cache, translation_cache)

    print(f"Done. PDFs: {len(pdf_paths)}, pages: {total_pages}, pages with text: {total_text_pages}")
    if args.translate_chinese:
        print(f"Translated pages: {total_translated_pages}")
        print(f"Translation cache: {args.translation_cache}")
    print(f"JSONL dataset: {args.jsonl}")


if __name__ == "__main__":
    main()
