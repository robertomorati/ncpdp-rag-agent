import json
from typing import Any, Dict, List, Tuple

from app.config import MAX_ELABORATION_PASSES, REFLECTION_SCORE_THRESHOLD
from app.prompts import ELABORATE_QUERY, JUDGE_CONTEXT, REFLECT_ANSWER
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
        # remove fallback
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

    def elaborate_query(
        self,
        question: str,
        feedback: str,
        draft_answer: str,
    ) -> str:
        prompt = ELABORATE_QUERY.format(
            question=question,
            feedback=feedback,
            draft_answer=draft_answer,
        )
        raw = self.rag.llm.generate_content(prompt).text
        parsed = self._safe_json(
            raw,
            fallback={"elaborated_query": question},
        )
        return (parsed.get("elaborated_query") or "").strip() or question

    def _reflection_scores_low(self, reflection: Dict[str, Any]) -> bool:
        cs = reflection.get("clarity_score")
        rs = reflection.get("relevance_score")
        if cs is None or rs is None:
            return False
        try:
            clarity = int(cs)
            relevance = int(rs)
        except (TypeError, ValueError):
            # fallback
            return False
        
        return min(clarity, relevance) <= REFLECTION_SCORE_THRESHOLD

    def run(self, question: str) -> Dict[str, Any]:
        tools_used: List[str] = ["search_kb"]

        # search in knowledge base 
        contexts = self.search_kb(question)
        tools_used.append("judge_context")

        # checker if answer sufficient or not
        judgment = self.judge_context(question, contexts)

        rewritten_query = ""
        # if not sufficient, use the rewritten query to make a new search
        if not judgment.get("sufficient", False):
            rq = (judgment.get("rewritten_query") or "").strip()
            if rq:
                rewritten_query = rq
                tools_used.append("search_kb")
                contexts = self.search_kb(rewritten_query)

        elaborated_queries: List[str] = []
        elaboration_pass = 0

        # calls retrieve from rag
        tools_used.append("generate_answer")
        draft_answer = self.generate_answer(question, contexts)

        tools_used.append("reflect_answer")
        reflection = self.reflect_answer(question, contexts, draft_answer)

        while (
            elaboration_pass < MAX_ELABORATION_PASSES
            and self._reflection_scores_low(reflection)
        ):
            feedback = (reflection.get("feedback") or "").strip()
            tools_used.append("elaborate_query")
            elaborated = self.elaborate_query(question, feedback, draft_answer)
            if not elaborated or elaborated == question:
                break
            elaborated_queries.append(elaborated)
            tools_used.append("search_kb")
            contexts = self.search_kb(elaborated)
            elaboration_pass += 1

            tools_used.append("generate_answer")
            draft_answer = self.generate_answer(question, contexts)
            tools_used.append("reflect_answer")

            # do the reflection to update clarity score and relevance score
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
            "elaborated_queries": elaborated_queries,
            "elaboration_passes": elaboration_pass,
        }

    def _safe_json(self, raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception:
            return fallback