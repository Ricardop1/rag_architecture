from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from answer import answer_with_context, retrieve

EVAL_FILE = ROOT_DIR / "eval.jsonl"
RESULTS_FILE = ROOT_DIR / "eval_results.jsonl"


def load_eval_set() -> list[dict]:
    rows = []
    with EVAL_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float | None:
    if not reference.strip():
        return None
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float | None:
    if not reference.strip():
        return None

    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = 0
    remaining = ref_tokens.copy()
    for token in pred_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def source_hit(retrieved_chunks: list[dict], expected_sources: list[dict]) -> float | None:
    valid_sources = [
        source for source in expected_sources if source.get("pdf_name", "").strip() or str(source.get("page", "")).strip()
    ]
    if not valid_sources:
        return None

    retrieved_pairs = {
        (chunk["metadata"].get("source", ""), str(chunk["metadata"].get("page", "")))
        for chunk in retrieved_chunks
    }
    expected_pairs = {
        (source.get("pdf_name", ""), str(source.get("page", "")))
        for source in valid_sources
    }
    return float(bool(retrieved_pairs & expected_pairs))


def evaluate_case(row: dict) -> dict:
    question = row.get("question", "").strip()
    if not question:
        return {
            "id": row.get("id"),
            "question": "",
            "skipped": True,
            "reason": "empty question",
        }

    chunks = retrieve(question)
    if not chunks:
        return {
            "id": row.get("id"),
            "question": question,
            "skipped": False,
            "answer": "",
            "retrieved_sources": [],
            "exact_match": exact_match("", row.get("expected_answer", "")),
            "token_f1": token_f1("", row.get("expected_answer", "")),
            "source_hit": source_hit(chunks, row.get("source_passages", [])),
        }

    answer = answer_with_context(question, chunks)
    retrieved_sources = [
        {
            "pdf_name": chunk["metadata"].get("source", ""),
            "page": chunk["metadata"].get("page", ""),
        }
        for chunk in chunks
    ]
    return {
        "id": row.get("id"),
        "question": question,
        "skipped": False,
        "answer": answer,
        "expected_answer": row.get("expected_answer", ""),
        "retrieved_sources": retrieved_sources,
        "expected_sources": row.get("source_passages", []),
        "exact_match": exact_match(answer, row.get("expected_answer", "")),
        "token_f1": token_f1(answer, row.get("expected_answer", "")),
        "source_hit": source_hit(chunks, row.get("source_passages", [])),
    }


def summarize(results: list[dict]) -> None:
    scored_exact = [result["exact_match"] for result in results if result.get("exact_match") is not None]
    scored_f1 = [result["token_f1"] for result in results if result.get("token_f1") is not None]
    scored_source = [result["source_hit"] for result in results if result.get("source_hit") is not None]

    print(f"Cases: {len(results)}")
    print(f"Scored exact match: {len(scored_exact)}")
    if scored_exact:
        print(f"Average exact match: {sum(scored_exact) / len(scored_exact):.3f}")
    print(f"Scored token F1: {len(scored_f1)}")
    if scored_f1:
        print(f"Average token F1: {sum(scored_f1) / len(scored_f1):.3f}")
    print(f"Scored source hit: {len(scored_source)}")
    if scored_source:
        print(f"Average source hit: {sum(scored_source) / len(scored_source):.3f}")


def main() -> int:
    eval_rows = load_eval_set()
    results = [evaluate_case(row) for row in eval_rows]

    with RESULTS_FILE.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summarize(results)
    print(f"Saved results to {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
