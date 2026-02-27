from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from src.reflens.answer.answer_propmts import ANSWER_SYSTEM, ANSWER_STYLE

Chunk = Dict[str, Any]  # {"id":"c_0","text":"...","score":..., "title":...}


@dataclass
class AnswerResult:
    draft_answer: str
    raw_model_output: str


def _format_context(retrieved_context: Union[List[str], List[Chunk]]) -> str:
    if not retrieved_context:
        return ""

    if isinstance(retrieved_context[0], str):
        # si te pasan strings, los numeramos igual
        blocks = []
        for i, txt in enumerate(retrieved_context):
            blocks.append(f"[c_{i}]\n{txt.strip()}")
        return "\n\n---\n\n".join(blocks)

    blocks: List[str] = []
    for c in retrieved_context:
        cid = c.get("chunk_id") or c.get("id") or c.get("meta", {}).get("chunk_id") or "c_?"
        score = c.get("score", "")
        title = c.get("title") or c.get("source") or ""
        text = (c.get("text") or "").strip()
        header = f"[{cid}] score={score} {title}".strip()
        blocks.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(blocks)


class AnswerAgent:
    def __init__(self, llm_call: Callable[[str, str], str]):
        """
        llm_call(system_prompt, user_prompt) -> str
        """
        self.llm_call = llm_call

    def run(
        self,
        user_query: str,
        retrieved_context: Union[List[str], List[Chunk]],
        tools_trace: Optional[Dict[str, Any]] = None,
    ) -> AnswerResult:
        context_txt = _format_context(retrieved_context)

        payload = {
            "user_query": user_query,
            "retrieved_context": context_txt,
            "tools_trace": tools_trace or {},
            "style": ANSWER_STYLE,
            "instruction": "Responde solo con texto (no JSON). Añade citas usando el ID exacto que aparece entre corchetes en el contexto (por ejemplo: (c_241), (c_0)). No inventes identificadores.",
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

        raw = self.llm_call(ANSWER_SYSTEM, user_prompt)
        draft = raw.strip()
        return AnswerResult(draft_answer=draft, raw_model_output=raw)