from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

from tfidf_search import (
    DATA_PATH,
    TOP_K,
    WINDOW_SIZE,
    build_page_windows,
    build_tfidf_vectors,
    load_pages,
    prepare_query,
    search,
    tokenize,
)


DEFAULT_MODEL = "gpt-4.1-mini"
SENTENCE_PATTERN = re.compile(r"(?<=[.!?。！？])\s+|\n+|●|○|- ")
COURSE_INFO_SOURCE = "c0_course_introduction.pdf"
COURSE_INFO_PAGES = 6
COURSE_INFO_KEYWORDS = {
    "老師",
    "教授",
    "授課",
    "講師",
    "教師",
    "助教",
    "誰教",
    "教這門課",
    "教課",
    "課程資訊",
    "課程",
    "office",
    "office hour",
    "辦公室",
    "信箱",
    "email",
    "期中考",
    "期末考",
    "考試",
    "評分",
    "成績",
    "加選",
    "上課",
    "出席",
}


def load_dotenv_if_available() -> None:
    def load_dotenv_manually() -> None:
        env_path = Path(".env")
        if not env_path.exists():
            return

        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not os.environ.get(key):
                os.environ[key] = value

    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv_manually()
        return

    load_dotenv()
    load_dotenv_manually()


def build_context(results: list[tuple[float, dict[str, object]]], max_chars: int) -> str:
    chunks: list[str] = []
    current_length = 0

    for score, document in results:
        source = str(document["source"])
        for page in document["pages"]:
            page_number = int(page["page"])
            text = " ".join(str(page["text"]).split())
            block = f"[{source} page {page_number}, score={score:.4f}]\n{text}\n"

            if current_length + len(block) > max_chars:
                remaining = max_chars - current_length
                if remaining > 200:
                    chunks.append(block[:remaining])
                return "\n".join(chunks)

            chunks.append(block)
            current_length += len(block)

    return "\n".join(chunks)


def is_course_info_question(question: str, expanded_query: str) -> bool:
    combined = f"{question} {expanded_query}".lower()
    return any(keyword.lower() in combined for keyword in COURSE_INFO_KEYWORDS)


def build_course_info_document(pages: list[dict[str, object]], page_count: int) -> dict[str, object] | None:
    course_pages = [
        page
        for page in pages
        if str(page["source"]) == COURSE_INFO_SOURCE and int(page["page"]) <= page_count
    ]
    if not course_pages:
        return None

    course_pages.sort(key=lambda page: int(page["page"]))
    return {
        "source": COURSE_INFO_SOURCE,
        "start_page": int(course_pages[0]["page"]),
        "end_page": int(course_pages[-1]["page"]),
        "pages": course_pages,
        "text": "\n".join(str(page["text"]) for page in course_pages),
    }


def prepend_result(
    results: list[tuple[float, dict[str, object]]],
    score: float,
    document: dict[str, object],
) -> list[tuple[float, dict[str, object]]]:
    source = str(document["source"])
    start_page = int(document["start_page"])
    end_page = int(document["end_page"])
    filtered_results = [
        (result_score, result_document)
        for result_score, result_document in results
        if not (
            str(result_document["source"]) == source
            and int(result_document["start_page"]) <= end_page
            and int(result_document["end_page"]) >= start_page
        )
    ]
    return [(score, document), *filtered_results]


def collect_sources(results: list[tuple[float, dict[str, object]]]) -> list[str]:
    seen: set[tuple[str, int]] = set()
    sources: list[str] = []

    for _score, document in results:
        source = str(document["source"])
        for page in document["pages"]:
            key = (source, int(page["page"]))
            if key in seen:
                continue
            seen.add(key)
            sources.append(f"{source} page {page['page']}")

    return sources


def split_candidate_sentences(context: str) -> list[str]:
    sentences: list[str] = []

    for sentence in SENTENCE_PATTERN.split(context):
        sentence = " ".join(sentence.split())
        if len(sentence) < 12:
            continue
        if sentence.startswith("[") and sentence.endswith("]"):
            continue
        sentences.append(sentence)

    return sentences


def iter_evidence_sentences(
    results: list[tuple[float, dict[str, object]]],
) -> list[tuple[float, str, int, str]]:
    evidence: list[tuple[float, str, int, str]] = []

    for retrieval_score, document in results:
        source = str(document["source"])
        for page in document["pages"]:
            page_number = int(page["page"])
            for sentence in split_candidate_sentences(str(page["text"])):
                evidence.append((retrieval_score, source, page_number, sentence))

    return evidence


def answer_extractive(
    question: str,
    expanded_query: str,
    results: list[tuple[float, dict[str, object]]],
    max_sentences: int,
) -> str:
    query_terms = Counter(tokenize(expanded_query))
    if not query_terms:
        return "我找不到足夠的關鍵詞來回答這個問題。"

    scored_sentences: list[tuple[float, int, str, int, str]] = []
    for index, (retrieval_score, source, page_number, sentence) in enumerate(iter_evidence_sentences(results)):
        sentence_terms = Counter(tokenize(sentence))
        if not sentence_terms:
            continue
        overlap_score = sum(query_terms[term] * sentence_terms.get(term, 0) for term in query_terms)
        score = overlap_score * (1.0 + retrieval_score)
        if score > 0:
            scored_sentences.append((score, index, source, page_number, sentence))

    if not scored_sentences:
        return "我在檢索到的講義內容中找不到明確答案。"

    scored_sentences.sort(key=lambda item: (-item[0], item[1]))
    selected = sorted(scored_sentences[:max_sentences], key=lambda item: item[1])
    evidence_lines = [
        f"- {sentence}（{source} page {page_number}）"
        for _score, _index, source, page_number, sentence in selected
    ]

    return (
        "根據檢索到的講義內容，最相關的答案依據是：\n"
        + "\n".join(evidence_lines)
        + "\n\n"
        "這是抽取式回答，也就是從講義片段中挑出最相關的句子；如果要更像人類整理後的回答，"
        "可以使用 --llm openai。"
    )


def answer_with_openai(question: str, context: str, model: str, allow_general_answer: bool) -> str:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("OpenAI package not found. Install it with: pip install openai") from error

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    client = OpenAI()
    if allow_general_answer:
        system_prompt = (
            "You answer questions in Traditional Chinese. First use the provided lecture context. "
            "If the lecture context contains enough information, answer from it and cite source pages. "
            "If the lecture context does not contain enough information, clearly say that the lecture "
            "does not provide the answer, then provide a general-knowledge answer under a separate "
            "label: '一般知識補充（非講義來源）'. Do not pretend that general knowledge came from the lecture."
        )
    else:
        system_prompt = (
            "You answer questions using only the provided lecture context. "
            "If the context does not contain the answer, say that the lecture context "
            "does not provide enough information. Answer in Traditional Chinese. "
            "Cite the source pages you used."
        )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Lecture context:\n{context}\n\n"
                    "Please give a concise answer and cite source pages."
                ),
            },
        ],
    )
    return response.output_text


def rewrite_query_with_openai(question: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("OpenAI package not found. Install it with: pip install openai") from error

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user's question into a concise retrieval query for searching "
                    "English NLP lecture slides. Preserve important names, dates, formulas, "
                    "acronyms, and technical terms. Add likely English synonyms when useful. "
                    "Return only the rewritten query, with no explanation."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        max_output_tokens=120,
    )
    rewritten_query = " ".join(response.output_text.split())
    return rewritten_query or question


def retrieve(
    question: str,
    retrieval_query: str,
    data_path: Path,
    top_k: int,
    window_size: int,
    translator: str,
    allow_overlap: bool,
) -> tuple[str, list[tuple[float, dict[str, object]]]]:
    pages = load_pages(data_path)
    documents = build_page_windows(pages, window_size)
    doc_vectors, idf = build_tfidf_vectors(documents)
    expanded_query = prepare_query(retrieval_query, translator)
    results = search(expanded_query, documents, doc_vectors, idf, top_k, allow_overlap)

    if is_course_info_question(question, expanded_query):
        course_info_document = build_course_info_document(pages, COURSE_INFO_PAGES)
        if course_info_document is not None:
            results = prepend_result(results, 1.0, course_info_document)

    return expanded_query, results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv_if_available()

    parser = argparse.ArgumentParser(description="Retrieve lecture pages and answer with a simple RAG pipeline.")
    parser.add_argument("question", nargs="*", help="question to answer")
    parser.add_argument("-k", "--top-k", type=int, default=TOP_K, help="number of retrieved page windows")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="number of adjacent pages per document")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="path to JSONL page data")
    parser.add_argument(
        "--translator",
        choices=["auto", "google", "dictionary", "none"],
        default="auto",
        help="query translation mode for Chinese input",
    )
    parser.add_argument("--allow-overlap", action="store_true", help="allow retrieved windows to overlap")
    parser.add_argument(
        "--llm",
        choices=["auto", "extractive", "openai"],
        default="extractive",
        help="answer generation mode",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name when --llm openai is used")
    parser.add_argument(
        "--rewrite-query",
        choices=["auto", "openai", "none"],
        default="auto",
        help="rewrite the user question into a retrieval query before TF-IDF",
    )
    parser.add_argument("--max-context-chars", type=int, default=6000, help="maximum context length")
    parser.add_argument("--max-sentences", type=int, default=3, help="sentences used by extractive answer")
    parser.add_argument(
        "--allow-general-answer",
        action="store_true",
        help="allow OpenAI mode to answer with general knowledge when lecture context is insufficient",
    )
    parser.add_argument("--show-context", action="store_true", help="print retrieved context before the answer")
    parser.add_argument("--show-query", action="store_true", help="print translated/expanded query")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}. Run extract_pdf_text.py first.")

    if args.window_size < 1:
        raise SystemExit("--window-size must be at least 1")

    question = " ".join(args.question).strip()
    if not question:
        question = input("Question: ").strip()

    llm_mode = args.llm
    if llm_mode == "auto":
        llm_mode = "openai" if os.environ.get("OPENAI_API_KEY") else "extractive"

    should_rewrite_query = args.rewrite_query == "openai" or (
        args.rewrite_query == "auto" and llm_mode == "openai"
    )
    retrieval_query = rewrite_query_with_openai(question, args.model) if should_rewrite_query else question

    expanded_query, results = retrieve(
        question,
        retrieval_query,
        args.data,
        args.top_k,
        args.window_size,
        args.translator,
        args.allow_overlap,
    )

    if args.show_query:
        if retrieval_query != question:
            print(f"Rewritten query: {retrieval_query}")
        print(f"Expanded query: {expanded_query}\n")

    if not results:
        print("我找不到相關講義頁面，因此無法回答。")
        return

    context = build_context(results, args.max_context_chars)

    if args.show_context:
        print("Retrieved context:")
        print(context)
        print()

    if llm_mode == "openai":
        answer = answer_with_openai(question, context, args.model, args.allow_general_answer)
    else:
        answer = answer_extractive(question, expanded_query, results, args.max_sentences)

    print("Answer:")
    print(answer)
    print()
    print("Sources:")
    for source in collect_sources(results):
        print(f"- {source}")


if __name__ == "__main__":
    main()
