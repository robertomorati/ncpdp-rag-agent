import json
from typing import Any, Dict, List, Tuple

from app.prompts import JUDGE_CONTEXT, REFLECT_ANSWER
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
        prompt = JUDGE_CONTEXT.format(question=question, context_text=context_text)
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
        prompt = REFLECT_ANSWER.format(
            question=question,
            context_text=context_text,
            answer=answer,
        )
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
        tools_used: List[str] = ["search_kb"]

        contexts = self.search_kb(question)
        tools_used.append("judge_context")
        judgment = self.judge_context(question, contexts)

        rewritten_query = ""
        if not judgment.get("sufficient", False):
            candidate = (judgment.get("rewritten_query") or "").strip()
            if candidate:
                rewritten_query = candidate
                tools_used.append("search_kb")
                contexts = self.search_kb(rewritten_query)

        tools_used.extend(["generate_answer", "reflect_answer"])

        draft_answer = self.generate_answer(question, contexts)
        reflection = self.reflect_answer(question, contexts, draft_answer)

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
            "tools_used": tools_used,
        }

    def _safe_json(self, raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception:
            return fallback