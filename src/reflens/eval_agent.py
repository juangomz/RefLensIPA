# src/lab4_agents/eval_agent.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Callable, Union

from reflens.eval_prompts import EVAL_SYSTEM, EVAL_SCHEMA_HINT

Verdict = Literal["pass", "revise", "reject"]
Chunk = Dict[str, Any]  # {"id": "...", "text": "...", "score": 0.8, ...}
ALLOWED_SCORE_VALUES = (0.0, 0.5, 1.0)
MAX_EVAL_CHUNKS = 10
MAX_EVAL_NEO4J_CHUNKS = 2
MAX_EVAL_CHARS_PER_CHUNK = 350
FIXED_POLISH_SYSTEM = """\
Eres un editor de estilo para respuestas RAG.
Tu tarea: mejorar la redacción para que suene natural, clara y útil para el usuario.

Reglas estrictas:
- No añadas hechos nuevos.
- No elimines hechos presentes en el texto original.
- No inventes citas ni IDs.
- Conserva los IDs de citas exactamente como están (por ejemplo: (c_115), (abc123)).
- Evita frases mecánicas tipo "el fragmento X indica...".
- Evita listados largos de IDs sin explicación.
- Devuelve solo el texto final, sin JSON ni encabezados técnicos.
"""


def _is_neo4j_chunk(chunk: Chunk) -> bool:
    source = str(chunk.get("source", "")).lower()
    cid = str(chunk.get("id") or chunk.get("chunk_id") or "").lower()
    return "neo4j" in source or cid.startswith("neo4j:")


def _select_eval_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Keep token usage low while preserving source diversity for judging."""
    if not chunks:
        return []

    selected: List[Chunk] = []
    selected_idxs: set[int] = set()
    neo4j_count = 0
    non_neo4j_count = 0
    non_neo4j_limit = max(MAX_EVAL_CHUNKS - MAX_EVAL_NEO4J_CHUNKS, 0)

    for idx, c in enumerate(chunks):
        is_neo4j = _is_neo4j_chunk(c)
        if is_neo4j:
            if neo4j_count >= MAX_EVAL_NEO4J_CHUNKS:
                continue
            neo4j_count += 1
        else:
            if non_neo4j_count >= non_neo4j_limit:
                continue
            non_neo4j_count += 1

        selected.append(c)
        selected_idxs.add(idx)
        if len(selected) >= MAX_EVAL_CHUNKS:
            return selected

    # Fill remaining slots from the original order if one source had fewer items.
    for idx, c in enumerate(chunks):
        if idx in selected_idxs:
            continue
        selected.append(c)
        if len(selected) >= MAX_EVAL_CHUNKS:
            break

    return selected


def select_chunks_for_eval(retrieved_context: Union[List[str], List[Chunk]]) -> Union[List[str], List[Chunk]]:
    """
    Public selector so caller can feed AnswerAgent and EvalAgent with the same references.
    Keeps eval token budget and source balance.
    """
    if not retrieved_context:
        return []
    if isinstance(retrieved_context[0], str):
        return [c for c in retrieved_context[:MAX_EVAL_CHUNKS] if isinstance(c, str)]
    return _select_eval_chunks(retrieved_context)  # type: ignore[arg-type]


@dataclass
class EvalResult:
    verdict: Verdict
    answer: str
    eval_json: Dict[str, Any]
    raw_model_output: str


def _format_context(retrieved_context: Union[List[str], List[Chunk]]) -> str:
    if not retrieved_context:
        return ""

    selected_context = select_chunks_for_eval(retrieved_context)

    if isinstance(selected_context[0], str):
        blocks = []
        for c in selected_context:  # type: ignore[assignment]
            txt = c.strip()
            if not txt:
                continue
            blocks.append(txt[:MAX_EVAL_CHARS_PER_CHUNK])
        return "\n\n---\n\n".join(blocks)

    blocks: List[str] = []
    for c in selected_context:  # type: ignore[assignment]
        cid = c.get("id") or c.get("chunk_id") or f'{c.get("doc_id","doc")}-{c.get("chunk","chunk")}'
        score = c.get("score", "")
        text = (c.get("text") or "").strip()[:MAX_EVAL_CHARS_PER_CHUNK]
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


class EvalAgent:
    def __init__(self, llm_call: Callable[[str, str], str]):
        self.llm_call = llm_call

    def _polish_fixed_answer(self, user_query: str, fixed_answer: str) -> str:
        payload = {
            "user_query": user_query,
            "text_to_polish": fixed_answer,
            "target_style": (
                "Respuesta final natural y trabajada, con 1-2 párrafos (bullets solo si aportan mucho), "
                "tono claro, sin formato telegráfico."
            ),
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        polished = self.llm_call(FIXED_POLISH_SYSTEM, user_prompt)
        return (polished or "").strip()

    def review(
        self,
        user_query: str,
        draft_answer: str,
        retrieved_context: Union[List[str], List[Chunk]],
        tools_trace: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        context_txt = _format_context(retrieved_context)

        payload = {
            "user_query": user_query,
            "draft_answer": draft_answer,
            "retrieved_context": context_txt,
            "tools_trace": tools_trace or {},
            "instructions": EVAL_SCHEMA_HINT,
        }

        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        raw = self.llm_call(EVAL_SYSTEM, user_prompt)

        try:
            eval = _safe_json_load(raw)
        except Exception:
            # fallback: no rompemos el pipeline
            eval = {
                "verdict": "revise",
                "scores": {"faithfulness": 0.0, "coverage": 0.0, "clarity": 0.0},
                "issues": [
                    {
                        "type": "unclear",
                        "severity": "high",
                        "span": "",
                        "explanation": "EvalAgent no pudo parsear JSON del modelo.",
                        "evidence": [],
                    }
                ],
                "citations": [],
                "fixed_answer": "",
            }

        eval_scores = eval.get("scores")
        if isinstance(eval_scores, dict):
            for k in ("faithfulness", "coverage", "clarity"):
                v = eval_scores.get(k, 0.0)
                try:
                    fv = float(v)
                except Exception:
                    fv = 0.0
                eval_scores[k] = min(ALLOWED_SCORE_VALUES, key=lambda x: abs(x - fv))

        verdict: Verdict = eval.get("verdict", "revise")
        fixed = (eval.get("fixed_answer") or "").strip()

        if fixed:
            try:
                polished = self._polish_fixed_answer(user_query=user_query, fixed_answer=fixed)
                if polished:
                    fixed = polished
                    eval["fixed_answer"] = polished
            except Exception:
                pass

        if verdict == "pass":
            final = fixed or draft_answer
        elif verdict == "revise" and fixed:
            final = fixed
        else:
            fallback = (
                "No tengo suficiente evidencia en las fuentes recuperadas para responder con seguridad. "
                "¿Quieres que recupere más contexto o puedes especificar mejor la pregunta?"
            )
            if not fixed:
                try:
                    polished_fallback = self._polish_fixed_answer(user_query=user_query, fixed_answer=fallback)
                    fixed = polished_fallback or fallback
                except Exception:
                    fixed = fallback
                eval["fixed_answer"] = fixed
            final = fixed

        return EvalResult(verdict=verdict, answer=final, eval_json=eval, raw_model_output=raw)
