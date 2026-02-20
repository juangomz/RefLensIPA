from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from reflens.graph.schemas import Chunk


def load_chunks_json(path: str | Path) -> list[Chunk]:
    """
    Load chunks from a JSON file containing either:
      - a list of chunk dicts
      - or a dict with key like {"chunks": [...]}

    Returns validated Chunk objects.
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        # Common patterns: {"chunks": [...]}, {"data": [...]}
        if "chunks" in raw and isinstance(raw["chunks"], list):
            items = raw["chunks"]
        elif "data" in raw and isinstance(raw["data"], list):
            items = raw["data"]
        else:
            raise ValueError(f"Unsupported JSON dict format keys={list(raw.keys())}")
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"Unsupported JSON format: {type(raw)}")

    chunks: list[Chunk] = []
    errors: list[str] = []

    for i, item in enumerate(items):
        try:
            chunks.append(Chunk.model_validate(item))
        except ValidationError as e:
            errors.append(f"Item {i} invalid: {e}")

    if errors:
        # Falla pronto para que lo veáis (podéis cambiar a "warn + skip" si preferís)
        raise ValueError("Invalid chunks:\n" + "\n".join(errors[:5]))

    return chunks