import json
from typing import Any, Dict, List, Tuple

from app.rag import RAGAssistant


class NCPDPAgent:
    def __init__(self) -> None:
        self.rag = RAGAssistant()

    def search_kb(self, query: str, top_k: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
        return self.rag.retrieve(query, top_k=top_k)

    def judge_context(
        self,
        question: str,
        contexts: List[Tuple[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if not contexts:
            return {
                "sufficient": False,
                "reason": "No context retrieved.",
                "rewritten_query": question,
            }

        context_text = self.rag.format_contexts(contexts, max_chars=700)

        prompt = f"""
You are an NCPDP Retrieval Judge.

Return ONLY valid JSON in this format:
{{
  "sufficient": true,
  "reason": "short explanation",
  "rewritten_query": ""
}}

Rules:
- If the context is enough to answer the question, set sufficient=true
- If the context is weak, incomplete, or from the wrong section (PDF), set sufficient=false
- If insufficient, provide a better rewritten_query (question)

Question:
{question}

Retrieved context:
{context_text}
"""
        raw = self.rag.llm.generate_content(prompt).text
        return self._safe_json(
            raw,
            fallback={
                "sufficient": True,
                "reason": "Could not parse judge output; assuming sufficient.",
                "rewritten_query": "",
            },
        )

    def generate_answer(
        self,
        question: str,
        contexts: List[Tuple[str, Dict[str, Any]]],
    ) -> str:
        return self.rag.generate_answer(question, contexts)

    def reflect_answer(
        self,
        question: str,
        contexts: List[Tuple[str, Dict[str, Any]]],
        answer: str,
    ) -> Dict[str, Any]:
        context_text = self.rag.format_contexts(contexts, max_chars=700)

        prompt = f"""
You are an NCPDP Answer Reviewer.

Return ONLY valid JSON in this format:
{{
  "grounded": true,
  "clarity_score": 5,
  "relevance_score": 5,
  "revision_needed": false,
  "feedback": "short explanation",
  "improved_answer": "final answer text"
}}

Rules:
- grounded=false if the answer contains unsupported claims
- clarity_score and relevance_score must be from 1 to 5
- revision_needed=true if the answer should be improved
- improved_answer must contain the revised answer
- Do not use markdown

Question:
{question}

Retrieved context:
{context_text}

Draft answer:
{answer}
"""
        raw = self.rag.llm.generate_content(prompt).text
        return self._safe_json(
            raw,
            fallback={
                "grounded": True,
                "clarity_score": 4,
                "relevance_score": 4,
                "revision_needed": False,
                "feedback": "Could not parse reflection output.",
                "improved_answer": answer,
            },
        )

    def run(self, question: str) -> Dict[str, Any]:
        # search with rag agent
        contexts = self.search_kb(question)

        # always run the judge
        judgment = self.judge_context(question, contexts)

        # if judment returned sufficienf false try with rewritten query
        rewritten_query = ""
        if not judgment.get("sufficient", False):
            rewritten_query = judgment.get("rewritten_query", "").strip()
            if rewritten_query:
                contexts = self.search_kb(rewritten_query)

        # get the final response
        draft_answer = self.generate_answer(question, contexts)
        reflection = self.reflect_answer(question, contexts, draft_answer)

        # improved answer
        final_answer = (
            reflection.get("improved_answer", draft_answer)
            if reflection.get("revision_needed", False)
            else draft_answer
        )

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "judgment": judgment,
            "draft_answer": draft_answer,
            "reflection": reflection,
            "final_answer": final_answer,
            "contexts_used": len(contexts),
        }

    def _safe_json(self, raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception:
            return fallback