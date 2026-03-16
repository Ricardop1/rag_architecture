from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from sentence_transformers import CrossEncoder

from ingest import CHROMA_DIR, COLLECTION_NAME, build_embeddings

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RETRIEVAL_TOP_K = 6
RERANK_TOP_K = 3
RERANKER_MAX_LENGTH = 512
LLM_MODEL = "Qwen/Qwen3.5-9B:together"
LLM_BASE_URL = "https://router.huggingface.co/v1"


@lru_cache(maxsize=1)
def get_reranker():

    return CrossEncoder(
        RERANKER_MODEL,
        max_length=RERANKER_MAX_LENGTH,
        trust_remote_code=True,
    )


def rerank(query: str, documents: Sequence[Document]) -> list[tuple[int, float]]:
    if not documents:
        return []

    scores = get_reranker().predict([(query, document.page_content) for document in documents])
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
    return [(index, float(score)) for index, score in ranked[:RERANK_TOP_K]]


def retrieve(query: str) -> list[dict]:
    if not query.strip():
        raise ValueError("Query must not be empty.")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=build_embeddings(),
    )
    vector_results = vector_store.similarity_search_with_score(query=query, k=RETRIEVAL_TOP_K)
    if not vector_results:
        return []

    documents = [document for document, _score in vector_results]
    vector_scores = [float(score) for _document, score in vector_results]
    reranked_documents = rerank(query, documents)

    return [
        {
            "content": documents[index].page_content,
            "metadata": dict(documents[index].metadata),
            "vector_score": vector_scores[index],
            "rerank_score": score,
        }
        for index, score in reranked_documents
    ]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        parts.append(
            "\n".join(
                [
                    f"[Chunk {index}]",
                    f"PDF: {metadata.get('source', 'unknown')}",
                    f"Page: {metadata.get('page', 'unknown')}",
                    f"Text: {chunk['content'].strip()}",
                ]
            )
        )
    return "\n\n".join(parts)


def answer_with_context(query: str, chunks: list[dict]) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set.")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=token,
    )
    message = HumanMessage(
        content=(
            "Answer the user question using only with the information in the context below. "
            "If the answer is not in the context, say that the provided document do not contain the answer.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{build_context(chunks)}"
        )
    )
    return llm.invoke([message]).content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Search query.")
    args = parser.parse_args(argv)
    results = retrieve(args.query)

    if not results:
        print("No results found.")
        return 0

    print("Answer:")
    print(answer_with_context(args.query, results))
    print()
    print("Sources used as context:")
    for result in results:
        metadata = result["metadata"]
        print(f"- {metadata.get('source', 'unknown')} page {metadata.get('page', 'unknown')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
