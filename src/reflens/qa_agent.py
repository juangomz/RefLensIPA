# src/lab4_agents/qa_agent.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Callable, Union

from src.reflens.qa_prompts import QA_SYSTEM, QA_SCHEMA_HINT

Verdict = Literal["pass", "revise", "reject"]
Chunk = Dict[str, Any]  # {"id": "...", "text": "...", "score": 0.8, ...}


@dataclass
class QAResult:
    verdict: Verdict
    answer: str
    qa_json: Dict[str, Any]
    raw_model_output: str


def _format_context(retrieved_context: Union[List[str], List[Chunk]]) -> str:
    if not retrieved_context:
        return ""

    if isinstance(retrieved_context[0], str):
        return "\n\n---\n\n".join([c.strip() for c in retrieved_context if c.strip()])

    blocks: List[str] = []
    for c in retrieved_context:  # type: ignore[assignment]
        cid = c.get("id") or c.get("chunk_id") or f'{c.get("doc_id","doc")}-{c.get("chunk","chunk")}'
        score = c.get("score", "")
        text = (c.get("text") or "").strip()
        title = c.get("title") or c.get("source") or ""
        header = f"[{cid}] score={score} {title}".strip()
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)


def _safe_json_load(raw: str) -> Dict[str, Any]:
    txt = raw.strip()
    try:
        return json.loads(txt)
    except Exception:
        i, j = txt.find("{"), txt.rfind("}")
        if i != -1 and j != -1 and j > i:
            return json.loads(txt[i : j + 1])
        raise


class QAAgent:
    def __init__(self, llm_call: Callable[[str, str], str]):
        self.llm_call = llm_call

    def review(
        self,
        user_query: str,
        draft_answer: str,
        retrieved_context: Union[List[str], List[Chunk]],
        tools_trace: Optional[Dict[str, Any]] = None,
    ) -> QAResult:
        context_txt = _format_context(retrieved_context)

        payload = {
            "user_query": user_query,
            "draft_answer": draft_answer,
            "retrieved_context": context_txt,
            "tools_trace": tools_trace or {},
            "instructions": QA_SCHEMA_HINT,
        }

        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        raw = self.llm_call(QA_SYSTEM, user_prompt)

        try:
            qa = _safe_json_load(raw)
        except Exception:
            # fallback: no rompemos el pipeline
            qa = {
                "verdict": "revise",
                "scores": {"faithfulness": 0.0, "coverage": 0.0, "clarity": 0.0},
                "issues": [
                    {
                        "type": "unclear",
                        "severity": "high",
                        "span": "",
                        "explanation": "QAAgent no pudo parsear JSON del modelo.",
                        "evidence": [],
                    }
                ],
                "citations": [],
                "fixed_answer": "",
            }

        verdict: Verdict = qa.get("verdict", "revise")
        fixed = (qa.get("fixed_answer") or "").strip()

        if verdict == "pass":
            final = draft_answer
        elif verdict == "revise" and fixed:
            final = fixed
        else:
            final = (
                "No tengo suficiente evidencia en las fuentes recuperadas para responder con seguridad. "
                "¿Quieres que recupere más contexto o puedes especificar mejor la pregunta?"
            )

        return QAResult(verdict=verdict, answer=final, qa_json=qa, raw_model_output=raw)