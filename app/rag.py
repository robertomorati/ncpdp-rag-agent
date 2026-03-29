from typing import Any, Dict, List, Tuple

import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    TOP_K,
    LLM_MODEL,
)
from app.prompts import GENERATE_ANSWER


class RAGAssistant:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

        genai.configure(api_key=GEMINI_API_KEY)
        self.llm = genai.GenerativeModel(LLM_MODEL)

    def retrieve(self, question: str, top_k: int = TOP_K):
        question_embedding = self.embedding_model.encode([question]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return list(zip(documents, metadatas))

    def format_contexts(self, contexts, max_chars: int = 900) -> str:
        formatted = []
        for i, (ctx, meta) in enumerate(contexts, start=1):
            source = meta.get("source", "unknown")
            chunk_index = meta.get("chunk_index", "n/a")
            shortened = ctx[:max_chars].strip()
            formatted.append(f"Context {i} | source={source} | chunk={chunk_index}\n{shortened}")
        return "\n\n".join(formatted)

    def generate_answer(self, question: str, contexts) -> str:
        if not contexts:
            return "Not found in provided database."

        joined_context = self.format_contexts(contexts, max_chars=900)
        prompt = GENERATE_ANSWER.format(
            question=question,
            joined_context=joined_context,
        )
        response = self.llm.generate_content(prompt)
        return getattr(response, "text", "No response generated.").strip()