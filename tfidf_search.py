from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


DATA_PATH = Path("data_text_pages.jsonl")
TOP_K = 3
WINDOW_SIZE = 3


TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "了",
    "什",
    "麼",
    "是",
    "的",
    "大",
    "約",
    "多",
    "少",
    "和",
    "與",
    "在",
    "中",
    "有",
    "請",
    "問",
    "個",
    "一",
    "幾",
    "誰",
    "這",
    "門",
    "人",
    "課",
}
QUERY_TRANSLATIONS = {
    "自然語言處理": "natural language processing nlp",
    "文字": "text",
    "文本": "text",
    "單字": "word",
    "詞": "word token",
    "詞彙": "vocabulary",
    "詞彙表": "vocabulary",
    "斷詞": "tokenization tokenize tokens",
    "標記": "token tokens",
    "向量": "vector",
    "特徵": "feature",
    "tfidf": "tf idf tf-idf term frequency inverse document frequency formula calculate computation",
    "TFIDF": "tf idf tf-idf term frequency inverse document frequency formula calculate computation",
    "TF-IDF": "tf idf tf-idf term frequency inverse document frequency formula calculate computation",
    "計算方法": "formula calculate computation how works",
    "計算方式": "formula calculate computation how works",
    "公式": "formula equation",
    "機率": "probability probabilities",
    "概率": "probability probabilities",
    "總可能組合數": "number probabilities possible letters combinations factorial",
    "可能組合數": "number probabilities possible letters combinations factorial",
    "組合數": "number probabilities combinations factorial",
    "有幾種可能": "number probabilities possible letters combinations factorial",
    "幾種可能": "number probabilities possible letters combinations factorial",
    "可能有幾種": "number probabilities possible letters combinations factorial",
    "總可能": "possible number probabilities combinations",
    "大約多少": "number approximately",
    "多少": "number how many",
    "替換式密碼": "substitution cipher",
    "替換密碼": "substitution cipher",
    "密碼": "cipher",
    "解密": "decryption decode",
    "加密": "encryption encode",
    "馬可夫鏈": "markov chain markov property markov model state transition",
    "馬可夫": "markov",
    "狀態": "state states",
    "轉移": "transition transitions",
    "目前狀態": "current state",
    "下一個狀態": "next state",
    "期中考": "midterm exam",
    "考試": "exam",
    "日期": "date",
    "幾月幾號": "date",
    "神經網路": "neural network",
    "深度學習": "deep learning",
    "分類": "classification classifier",
    "貝氏": "bayes naive bayes",
    "貝葉斯": "bayes naive bayes",
    "平滑": "smoothing",
    "字典": "dictionary dictionaries",
    "老師": "teacher lecturer professor instructor albert yang",
    "教授": "teacher lecturer professor instructor albert yang",
    "教這門課的人": "teacher lecturer professor instructor course albert yang",
    "這門課的人": "teacher lecturer professor instructor course albert yang",
    "誰教": "teacher lecturer professor instructor course albert yang",
    "授課": "teacher lecturer professor instructor course albert yang",
    "授課老師": "teacher lecturer professor instructor albert yang",
    "授課教師": "teacher lecturer professor instructor albert yang",
    "授課教授": "teacher lecturer professor instructor albert yang",
    "課程老師": "teacher lecturer professor instructor albert yang",
    "課程教師": "teacher lecturer professor instructor albert yang",
    "講師": "teacher lecturer professor instructor albert yang",
    "助教": "ta teaching assistant roger wu",
    "辦公室": "office room",
    "信箱": "email",
    "電子郵件": "email",
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"(?i)\btf[-\s]?idf\b", "tf idf", text)
    return [
        token
        for token in (match.group(0).lower() for match in TOKEN_PATTERN.finditer(text))
        if token not in STOPWORDS
    ]


def has_cjk(text: str) -> bool:
    return CJK_PATTERN.search(text) is not None


def expand_query_with_dictionary(query: str) -> str:
    translations = [
        translation
        for chinese_phrase, translation in QUERY_TRANSLATIONS.items()
        if chinese_phrase in query
    ]

    if not translations:
        return query

    return f"{query} {' '.join(translations)}"


def translate_query_to_english(query: str, translator: str) -> str:
    if translator == "none" or not has_cjk(query):
        return ""

    if translator in {"auto", "google"}:
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            if translator == "google":
                raise SystemExit(
                    "Translator package not found. Install it with: pip install deep-translator"
                )
            return ""

        try:
            translated = GoogleTranslator(source="auto", target="en").translate(query)
        except Exception as error:
            if translator == "google":
                raise SystemExit(f"Google translation failed: {error}")
            return ""

        return translated or ""

    if translator == "dictionary":
        return ""

    raise SystemExit(f"Unknown translator: {translator}")


def prepare_query(query: str, translator: str) -> str:
    expanded_query = expand_query_with_dictionary(query)
    translated_query = translate_query_to_english(query, translator)
    query_parts = [expanded_query]

    if translated_query and translated_query.lower() not in expanded_query.lower():
        query_parts.append(translated_query)

    return " ".join(query_parts)


def load_pages(path: Path) -> list[dict[str, object]]:
    pages: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            pages.append(
                {
                    "source": record["source"],
                    "page": record["page"],
                    "text": text,
                }
            )

    return pages


def build_page_windows(pages: list[dict[str, object]], window_size: int) -> list[dict[str, object]]:
    pages_by_source: dict[str, list[dict[str, object]]] = {}

    for page in pages:
        pages_by_source.setdefault(str(page["source"]), []).append(page)

    windows: list[dict[str, object]] = []

    for source, source_pages in pages_by_source.items():
        source_pages.sort(key=lambda page: int(page["page"]))

        for start_index in range(0, len(source_pages)):
            window_pages = source_pages[start_index : start_index + window_size]
            if not window_pages:
                continue

            first_page = int(window_pages[0]["page"])
            last_page = int(window_pages[-1]["page"])

            windows.append(
                {
                    "source": source,
                    "start_page": first_page,
                    "end_page": last_page,
                    "pages": window_pages,
                    "text": "\n".join(str(page["text"]) for page in window_pages),
                }
            )

            if len(window_pages) < window_size:
                break

    return windows


def build_tfidf_vectors(documents: list[dict[str, object]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    doc_term_counts = [Counter(tokenize(str(document["text"]))) for document in documents]
    doc_freq: Counter[str] = Counter()

    for term_counts in doc_term_counts:
        doc_freq.update(term_counts.keys())

    doc_count = len(documents)
    idf = {
        term: math.log((doc_count + 1) / (freq + 1)) + 1
        for term, freq in doc_freq.items()
    }

    doc_vectors = [to_tfidf_vector(term_counts, idf) for term_counts in doc_term_counts]
    return doc_vectors, idf


def to_tfidf_vector(term_counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    vector: dict[str, float] = {}

    for term, count in term_counts.items():
        if term not in idf:
            continue
        tf = 1 + math.log(count)
        vector[term] = tf * idf[term]

    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if norm == 0:
        return vector

    return {term: weight / norm for term, weight in vector.items()}


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


def make_snippet(text: str, query_terms: set[str], max_length: int = 220) -> str:
    compact_text = " ".join(text.split())
    lower_text = compact_text.lower()

    first_match = min(
        (lower_text.find(term) for term in query_terms if term and lower_text.find(term) != -1),
        default=0,
    )
    start = max(first_match - 70, 0)
    end = min(start + max_length, len(compact_text))
    snippet = compact_text[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(compact_text):
        snippet += "..."

    return snippet


def overlaps_selected(
    document: dict[str, object],
    selected_documents: list[dict[str, object]],
) -> bool:
    source = str(document["source"])
    start_page = int(document["start_page"])
    end_page = int(document["end_page"])

    for selected_document in selected_documents:
        if source != str(selected_document["source"]):
            continue

        selected_start = int(selected_document["start_page"])
        selected_end = int(selected_document["end_page"])
        if start_page <= selected_end and end_page >= selected_start:
            return True

    return False


def search(
    expanded_query: str,
    documents: list[dict[str, object]],
    doc_vectors: list[dict[str, float]],
    idf: dict[str, float],
    top_k: int,
    allow_overlap: bool,
) -> list[tuple[float, dict[str, object]]]:
    query_terms = Counter(tokenize(expanded_query))
    query_vector = to_tfidf_vector(query_terms, idf)

    if not query_vector:
        return []

    scored_documents = [
        (cosine_similarity(query_vector, doc_vector), document)
        for document, doc_vector in zip(documents, doc_vectors)
    ]
    scored_documents = [(score, document) for score, document in scored_documents if score > 0]
    scored_documents.sort(key=lambda item: item[0], reverse=True)

    if allow_overlap:
        return scored_documents[:top_k]

    selected: list[tuple[float, dict[str, object]]] = []
    selected_documents: list[dict[str, object]] = []

    for score, document in scored_documents:
        if overlaps_selected(document, selected_documents):
            continue
        selected.append((score, document))
        selected_documents.append(document)
        if len(selected) == top_k:
            break

    return selected


def print_results(
    expanded_query: str,
    results: list[tuple[float, dict[str, object]]],
    show_query: bool,
) -> None:
    query_terms = set(tokenize(expanded_query))

    if show_query:
        print(f"Expanded query: {expanded_query}")
        print()

    if not results:
        print("No related pages found.")
        return

    for rank, (score, document) in enumerate(results, start=1):
        print(
            f"{rank}. score={score:.4f} | {document['source']} | "
            f"pages {document['start_page']}-{document['end_page']}"
        )
        for page in document["pages"]:
            print(f"   [page {page['page']}]")
            print(f"   {make_snippet(str(page['text']), query_terms)}")
        print()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Rank adjacent lecture PDF page windows with TF-IDF.")
    parser.add_argument("query", nargs="*", help="query words")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="number of ranked page windows to return")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="number of adjacent pages per TF-IDF document")
    parser.add_argument("--allow-overlap", action="store_true", help="allow returned page windows to overlap")
    parser.add_argument(
        "--translator",
        choices=["auto", "google", "dictionary", "none"],
        default="auto",
        help="query translation mode for Chinese input",
    )
    parser.add_argument("--show-query", action="store_true", help="show the translated/expanded query before results")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="path to JSONL page data")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}. Run extract_pdf_text.py first.")

    if args.window_size < 1:
        raise SystemExit("--window-size must be at least 1")

    pages = load_pages(args.data)
    documents = build_page_windows(pages, args.window_size)
    doc_vectors, idf = build_tfidf_vectors(documents)

    query = " ".join(args.query).strip()
    if not query:
        query = input("Query: ").strip()

    expanded_query = prepare_query(query, args.translator)
    results = search(
        expanded_query,
        documents,
        doc_vectors,
        idf,
        args.top_k,
        args.allow_overlap,
    )
    print_results(expanded_query, results, args.show_query)


if __name__ == "__main__":
    main()
