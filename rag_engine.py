import json
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any

import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI


# Project paths.
# Put your .txt files in: data/txt/
BASE_PATH = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_PATH / "data" / "txt"))
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", BASE_PATH / "chromadb_genai_midterm"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "genai_midterm_chunks")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_txt_files() -> Dict[str, str]:
    txt_files = sorted(DATA_DIR.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {DATA_DIR}. "
            "Copy your MS ADS .txt files from Google Drive into data/txt/ before running."
        )

    all_texts: Dict[str, str] = {}
    for file_path in txt_files:
        try:
            all_texts[str(file_path)] = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            all_texts[str(file_path)] = file_path.read_text(encoding="latin-1")
    return all_texts


def build_chunks() -> List[Dict[str, Any]]:
    all_texts = load_txt_files()

    cleaned_docs = []
    for idx, (source, text) in enumerate(all_texts.items()):
        cleaned_docs.append(
            {
                "doc_id": idx,
                "source": source,
                "text": clean_text(text),
            }
        )

    all_chunks = []
    for row in cleaned_docs:
        pieces = chunk_text(row["text"])
        for chunk_id, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "doc_id": int(row["doc_id"]),
                    "source": row["source"],
                    "chunk_id": int(chunk_id),
                    "char_count": int(len(piece)),
                    "text": piece,
                }
            )
    return all_chunks


def get_chroma_collection():
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    existing_collections = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing_collections:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    else:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    if collection.count() == 0:
        chunks = build_chunks()

        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "doc_id": int(chunk["doc_id"]),
                "chunk_id": int(chunk["chunk_id"]),
                "char_count": int(chunk["char_count"]),
            }
            for chunk in chunks
        ]
        ids = [str(uuid.uuid4()) for _ in chunks]

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            collection.add(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        # Optional index for debugging.
        index_path = PERSIST_DIR / "chunk_index.jsonl"
        with index_path.open("w", encoding="utf-8") as f:
            for item_id, metadata in zip(ids, metadatas):
                f.write(json.dumps({"id": item_id, **metadata}, ensure_ascii=False) + "\n")

    return collection


def get_vectorstore() -> Chroma:
    # This creates/reuses the Chroma DB before LangChain connects to it.
    get_chroma_collection()

    embedding_model = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{EMBEDDING_MODEL_NAME}"
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embedding_model,
    )


_vectorstore = None
_openai_client = None


def vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    return _vectorstore


def openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable is missing.")
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_collection_count() -> int:
    try:
        return get_chroma_collection().count()
    except Exception:
        return 0


def ask_rag(
    question: str,
    show_sources: bool = False,
    k: int = 3,
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    active_vector_store = vectorstore()

    # Same logic as your notebook: retrieve more first, then filter.
    docs = active_vector_store.similarity_search(question, k=6)

    filtered_docs = []
    for d in docs:
        source = d.metadata.get("source", "").lower()
        text = d.page_content.lower()

        if "ms in applied data science" in text:
            filtered_docs.append(d)
        elif "booth" in source and "core courses" in text:
            filtered_docs.append(d)

    # Fallback: if the strict filter removes everything, keep top retrieved docs.
    # This makes the API more robust for questions where the phrase is not repeated in every chunk.
    docs = (filtered_docs or docs)[:k]
    context = format_docs(docs)

    instructions = (
        "You are a helpful assistant answering questions about the University of Chicago "
        "MS in Applied Data Science program. "
        "Use ONLY the provided context. If multiple programs are mentioned, prioritize answers "
        "specifically about the MS in Applied Data Science program, not joint MBA programs. "
        "If unclear, say you don't know. "
        "If the answer is not clearly in the context, say: "
        "\"I don't know based on the provided context.\" "
        "Keep the answer clear, concise, and factual. Do not make up details."
    )

    user_input = f"""Context:
{context}

Question:
{question}

Answer:"""

    response = openai_client().responses.create(
        model=model,
        instructions=instructions,
        input=user_input,
    )

    sources = []
    for doc in docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown source"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "preview": doc.page_content[:300].replace("\n", " "),
            }
        )

    if show_sources:
        print("QUESTION:", question)
        print("ANSWER:", response.output_text)
        print("SOURCES:", sources)

    return {
        "question": question,
        "answer": response.output_text,
        "sources": sources,
    }
