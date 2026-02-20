from __future__ import annotations

import json
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import ValidationError

from dotenv import load_dotenv
from reflens.graph.schemas import Chunk, ExtractedGraph, EntityType, RelationType
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Load GROQ configuration from environment variables
GROQ_API_ENDPOINT = os.getenv("GROQ_API_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_API_ENDPOINT,
)

# ---- Prompt helpers ----

def _allowed_types_str() -> str:
    # Extract Literal options from typing.Literal
    # Pydantic doesn't expose them directly here, so keep as static docs in prompt.
    entity_types = [
        "Person", "Org", "Project", "Tool", "Dataset", "Deliverable", "Metric", "Concept", "Other"
    ]
    rel_types = ["USES", "REQUIRES", "PART_OF", "EVALUATED_BY", "RELATED_TO"]
    return (
        f"Allowed entity types: {entity_types}\n"
        f"Allowed relation types: {rel_types}\n"
    )


_SYSTEM_PROMPT = (
    "You are a precise information extraction system that builds a knowledge graph.\n"
    "Return ONLY valid JSON. No markdown, no explanations.\n"
    "Do not invent entities or relations not supported by the text.\n"
    "If nothing relevant is present, return empty lists.\n"
)

def _build_user_prompt(chunk_text: str) -> str:
    return f"""
Extract a small knowledge graph from the CHUNK below.

{_allowed_types_str()}

Rules:
- Output must be a single JSON object with keys: "entities" and "relations".
- "entities" is a list of objects: {{"name","type","aliases","confidence"}}.
- "relations" is a list of objects: {{"source","target","type","confidence","evidence"}}.
- source/target must match an entity name you produced in "entities".
- confidence is a float in [0,1]. Use lower confidence if unsure.
- evidence is an optional short quote from the chunk that supports the relation (max 20 words).
- Keep it minimal: prefer fewer, higher-quality items.

CHUNK:
{chunk_text}
""".strip()


def _safe_json_loads(text: str) -> Any:
    """
    Best-effort JSON parsing. The model should return pure JSON, but this
    guards against occasional leading/trailing text.
    """
    text = text.strip()
    # Fast path: direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract the first JSON object substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found", text, 0)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((json.JSONDecodeError, ValidationError, RuntimeError)),
)
def extract_graph(chunk: Chunk, *, model: str | None = None) -> ExtractedGraph:
    """
    Calls the LLM to extract entities and relations from a single chunk.
    Returns validated ExtractedGraph. Retries on malformed JSON / schema issues.
    """
    prompt = _build_user_prompt(chunk.text)

    resp = client.chat.completions.create(
        model=model or GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        top_p=0.9,
    )

    content = resp.choices[0].message.content or ""
    data = _safe_json_loads(content)

    # Validate against schema (raises ValidationError if wrong)
    extracted = ExtractedGraph.model_validate(data)

    # Extra guard: ensure relations reference declared entities
    entity_names = {e.name for e in extracted.entities}
    for r in extracted.relations:
        if r.source not in entity_names or r.target not in entity_names:
            raise RuntimeError(
                f"Relation references unknown entity: {r.source} -> {r.target}"
            )

    return extracted