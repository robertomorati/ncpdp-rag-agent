import chromadb
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_text
from app.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    PROCESSED_TEXT_PATH,
    PROCESSED_AUDIO_TEXT_PATH,
)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vector_store() -> None:
    pdf_text = load_text(PROCESSED_TEXT_PATH)
    audio_text = load_text(PROCESSED_AUDIO_TEXT_PATH)

    pdf_chunks = chunk_text(pdf_text)
    audio_chunks = chunk_text(audio_text)

    print(f"PDF chunks: {len(pdf_chunks)}")
    print(f"Audio chunks: {len(audio_chunks)}")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # cleanup the collection to avoid duplicated chunks
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)

    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = pdf_chunks + audio_chunks

    metadatas = (
        [{"source": "pdf", "chunk_index": i} for i in range(len(pdf_chunks))]
        + [{"source": "audio", "chunk_index": i} for i in range(len(audio_chunks))]
    )

    ids = [f"doc-{i}" for i in range(len(documents))]
    embeddings = model.encode(documents).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    chunks = pdf_chunks + audio_chunks
    print(f"Stored {len(chunks)} chunks in ChromaDB collection '{CHROMA_COLLECTION}'")


if __name__ == "__main__":
    build_vector_store()