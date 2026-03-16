import chromadb
import logging
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / ".chroma"
COLLECTION_NAME = "pdf_rag"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 80


def discover_pdf_paths() -> list[Path]:
    return sorted(path for path in DATA_DIR.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")


def build_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"trust_remote_code": True},
    )
    embeddings._client.max_seq_length = CHUNK_SIZE
    return embeddings


def normalize_documents(documents: Iterable[Document], pdf_path: Path) -> list[Document]:
    relative_source = str(pdf_path.relative_to(DATA_DIR))
    normalized: list[Document] = []
    for page_number, document in enumerate(documents):
        metadata = dict(document.metadata)
        metadata["source"] = relative_source
        metadata["page"] = metadata.get("page", page_number)
        normalized.append(Document(page_content=document.page_content, metadata=metadata))
    return normalized


def reset_collection() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def ingest_pdfs() -> tuple[int, int, int]:

    pdf_paths = discover_pdf_paths()
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {DATA_DIR}")

    page_documents: list[Document] = []
    for pdf_path in pdf_paths:
        list_docs = PyPDFLoader(str(pdf_path)).load()
        page_documents.extend(normalize_documents(list_docs, pdf_path))

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, trust_remote_code=True)
    tokenizer.model_max_length = 10**9
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    chunks = text_splitter.split_documents(page_documents)
    if not chunks:
        raise ValueError(f"Loaded PDFs from {DATA_DIR}, but no chunks were produced.")

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index

    reset_collection()
    
    embedding_model = build_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    return len(pdf_paths), len(page_documents), len(chunks)


def main() -> int:
    pdf_count, page_count, chunk_count = ingest_pdfs()
    print(
        f"Ingested {pdf_count} PDF(s), {page_count} page(s), "
        f"{chunk_count} chunk(s) into {CHROMA_DIR}/{COLLECTION_NAME}."
    )



if __name__ == "__main__":
    main()
