from __future__ import annotations

import hashlib
import unicodedata
from typing import Dict, Tuple

from src.reflens.graph.schemas import (
    Chunk,
    ExtractedGraph,
    NormalizedEntity,
    NormalizedGraph,
    NormalizedRelation,
)


def _normalize_text(s: str) -> str:
    """
    Normalize for stable IDs:
    - strip
    - lowercase
    - collapse whitespace
    - remove diacritics (á -> a)
    """
    s = s.strip().lower()
    s = " ".join(s.split())
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_entity_id(entity_type: str, canonical_name: str) -> str:
    return _sha1(f"{entity_type.lower()}::{canonical_name}")


def make_relation_id(source_id: str, rel_type: str, target_id: str) -> str:
    return _sha1(f"{source_id}::{rel_type}::{target_id}")


def normalize_graph(extracted: ExtractedGraph, chunk: Chunk) -> NormalizedGraph:
    """
    Turn ExtractedGraph -> NormalizedGraph:
    - canonicalize names
    - generate stable IDs
    - dedupe entities by (type, canonical_name)
    - resolve relations to source_id/target_id
    - dedupe relations by (source_id, rel_type, target_id)
    """
    # 1) Entities: dedupe + id
    entity_key_to_entity: Dict[Tuple[str, str], NormalizedEntity] = {}
    name_to_best_key: Dict[str, Tuple[str, str]] = {}

    for e in extracted.entities:
        canonical = _normalize_text(e.name)
        key = (e.type, canonical)

        if key not in entity_key_to_entity:
            ent_id = make_entity_id(e.type, canonical)
            entity_key_to_entity[key] = NormalizedEntity(
                id=ent_id,
                name=e.name,
                canonical_name=canonical,
                type=e.type,
                aliases=list(dict.fromkeys(e.aliases)),  # stable unique
                confidence=e.confidence,
            )
        else:
            # Merge aliases + keep the best (highest confidence) surface name
            existing = entity_key_to_entity[key]
            merged_aliases = list(dict.fromkeys(existing.aliases + e.aliases))
            best_name = existing.name
            best_conf = existing.confidence
            if e.confidence > best_conf:
                best_name = e.name
                best_conf = e.confidence

            entity_key_to_entity[key] = existing.model_copy(
                update={
                    "name": best_name,
                    "confidence": max(existing.confidence, e.confidence),
                    "aliases": merged_aliases,
                }
            )

        # Map raw name to this key (prefer highest confidence mapping)
        raw_name = e.name.strip()
        prev_key = name_to_best_key.get(raw_name)
        if prev_key is None:
            name_to_best_key[raw_name] = key
        else:
            # If same raw_name appears with multiple types, keep the higher confidence one
            prev_ent = entity_key_to_entity.get(prev_key)
            cur_ent = entity_key_to_entity.get(key)
            if prev_ent and cur_ent and cur_ent.confidence > prev_ent.confidence:
                name_to_best_key[raw_name] = key

    # Helper to resolve a relation endpoint name -> NormalizedEntity
    def resolve(name: str) -> NormalizedEntity | None:
        raw = name.strip()
        key = name_to_best_key.get(raw)
        if key and key in entity_key_to_entity:
            return entity_key_to_entity[key]

        # Fallback: try canonical match across all entities (handles tiny name variations)
        canon = _normalize_text(raw)
        for (t, c), ent in entity_key_to_entity.items():
            if c == canon:
                return ent
        return None

    # 2) Relations: resolve IDs + dedupe
    rel_key_to_rel: Dict[Tuple[str, str, str], NormalizedRelation] = {}

    for r in extracted.relations:
        src = resolve(r.source)
        tgt = resolve(r.target)
        if src is None or tgt is None:
            # Skip unresolved relation safely
            continue

        rel_key = (src.id, r.type, tgt.id)
        if rel_key not in rel_key_to_rel:
            rel_id = make_relation_id(src.id, r.type, tgt.id)
            rel_key_to_rel[rel_key] = NormalizedRelation(
                id=rel_id,
                source=r.source,
                target=r.target,
                type=r.type,
                confidence=r.confidence,
                evidence=r.evidence,
                source_id=src.id,
                target_id=tgt.id,
            )
        else:
            # Keep max confidence + keep any evidence
            existing = rel_key_to_rel[rel_key]
            rel_key_to_rel[rel_key] = existing.model_copy(
                update={
                    "confidence": max(existing.confidence, r.confidence),
                    "evidence": existing.evidence or r.evidence,
                }
            )

    return NormalizedGraph(
        entities=list(entity_key_to_entity.values()),
        relations=list(rel_key_to_rel.values()),
        chunk=chunk,
    )