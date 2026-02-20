from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


# ---------- Input (from ingestion pipeline) ----------

class Chunk(BaseModel):
    """Single text chunk to be converted into a knowledge graph update."""
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="Document identifier (stable across runs)")
    chunk_id: str = Field(..., description="Chunk identifier unique within doc_id")
    text: str = Field(..., description="Chunk text content")
    source: str = Field(..., description="Source filename or URL")
    timestamp: str = Field(..., description="ISO date or datetime string when chunk was produced")
    embedding: Optional[List[float]] = None


# ---------- Graph extraction output (LLM) ----------

# Keep this list short for MVP; you can extend later.
EntityType = Literal[
    "Person",
    "Org",
    "Project",
    "Tool",
    "Dataset",
    "Deliverable",
    "Metric",
    "Concept",
    "Other",
]

RelationType = Literal[
    "USES",
    "REQUIRES",
    "PART_OF",
    "EVALUATED_BY",
    "RELATED_TO",
]


class Entity(BaseModel):
    """An entity node extracted from text (pre-normalization)."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Surface form or canonical name")
    type: EntityType = Field(..., description="Entity type (controlled vocabulary)")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    confidence: float = Field(ge=0.0, le=1.0, description="Extractor confidence [0,1]")


class Relation(BaseModel):
    """A directed edge between two entities extracted from text (pre-normalization)."""
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="Source entity name (must match an Entity.name)")
    target: str = Field(..., description="Target entity name (must match an Entity.name)")
    type: RelationType = Field(..., description="Relation type (controlled vocabulary)")
    confidence: float = Field(ge=0.0, le=1.0, description="Extractor confidence [0,1]")
    evidence: Optional[str] = Field(
        default=None,
        description="Short quote/span from the chunk supporting this relation (optional)",
    )


class ExtractedGraph(BaseModel):
    """Structured extraction result for a single chunk."""
    model_config = ConfigDict(extra="forbid")

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# ---------- Optional: normalized forms (post-processing) ----------

class NormalizedEntity(Entity):
    """Entity with a stable ID after normalization."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable hash ID (e.g., sha1(type::normalized_name))")
    canonical_name: str = Field(..., description="Normalized/canonical name used for IDs")


class NormalizedRelation(Relation):
    """Relation with a stable ID and resolved entity IDs after normalization."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable hash ID (e.g., sha1(src_id::type::tgt_id))")
    source_id: str = Field(..., description="Resolved source entity ID")
    target_id: str = Field(..., description="Resolved target entity ID")


class NormalizedGraph(BaseModel):
    """Normalized graph update ready to write to Neo4j."""
    model_config = ConfigDict(extra="forbid")

    entities: list[NormalizedEntity] = Field(default_factory=list)
    relations: list[NormalizedRelation] = Field(default_factory=list)
    chunk: Chunk