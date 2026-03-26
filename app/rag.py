from typing import List

import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    TOP_K,
)


class RAGAssistant:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file.")

        genai.configure(api_key=GEMINI_API_KEY)
        #print("Gemini key loaded:", bool(GEMINI_API_KEY))
        #for model in genai.list_models():
        #    if "generateContent" in model.supported_generation_methods:
        #        print(model.name)
        self.llm = genai.GenerativeModel("models/gemini-2.5-flash")

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[str]:
        question_embedding = self.embedding_model.encode([question]).tolist()[0]

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        return documents

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        joined_context = "\n\n".join(
            [f"Context {i + 1}:\n{ctx}" for i, ctx in enumerate(contexts)]
        )

        return f"""
You are an assistant specialized in the NCPDP guide, named NCPDP Assistant.

Answer the user's question only using the retrieved context below.
If the answer is not clearly supported by the context, say that the context is insufficient.

Your job is to:
- Answer ONLY based on the provided knowledge base
- Be concise, clear, and structured
- Use bullet points when possible
-  If the answer is not in the knowledge, say "Not found in provided database"
- When possible, reference concepts exactly as described in the material

Do NOT invent information. Use ONLY the DATABASE provided. Do not search outside of Knowledge database provideded.

User question:
{question}

Retrieved context:
{joined_context}

Give a clear and direct answer.
"""

    def ask(self, question: str) -> str:
        contexts = self.retrieve(question)
        prompt = self.build_prompt(question, contexts)
        response = self.llm.generate_content(prompt)
        return response.text