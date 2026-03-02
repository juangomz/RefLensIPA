import os
import time
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI
from openai import BadRequestError
from langfuse import get_client, observe
from src.rag.config import settings
from src.rag.observability.langfuse_client import configure_langfuse

# Importa tus agentes (ajusta el path según dónde los tengas)
from src.reflens.answer.answer_agent import AnswerAgent
from src.reflens.query.query_agent import QueryAgent, QUERY_AGENT_SYSTEM
from src.reflens.eval_agent import EvalAgent, select_chunks_for_eval

import json

load_dotenv()

if (
    settings.langfuse_public_key
    and settings.langfuse_secret_key
    and settings.langfuse_base_url
):
    configure_langfuse()

GROQ_API_ENDPOINT = os.getenv("GROQ_API_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", GROQ_MODEL)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GROQ_MODEL)
QUERY_MODEL = os.getenv("QUERY_MODEL", ANSWER_MODEL)

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_ENDPOINT)

def _llm_call_with_model(model_name: str, system_prompt: str, user_prompt: str) -> str:
    r = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return r.choices[0].message.content


def llm_call_answer(system_prompt: str, user_prompt: str) -> str:
    return _llm_call_with_model(ANSWER_MODEL, system_prompt, user_prompt)


def llm_call_judge(system_prompt: str, user_prompt: str) -> str:
    return _llm_call_with_model(JUDGE_MODEL, system_prompt, user_prompt)

def tool_to_openai_schema(t) -> dict:
    params = getattr(t, "parameters", None)

    # Caso 1: ya es dict
    if isinstance(params, dict):
        schema = params

    # Caso 2: es un Pydantic BaseModel *CLASE* (metaclass)
    elif hasattr(params, "model_json_schema"):  # pydantic v2
        schema = params.model_json_schema()

    # Caso 3: es un Pydantic BaseModel *INSTANCIA*
    elif hasattr(params, "model_dump"):
        schema = params.model_dump()

    # Caso 4: no hay schema -> objeto vacío válido
    else:
        schema = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": getattr(t, "description", "") or "",
            "parameters": schema,
        },
    }


@observe(as_type="span")
def _execute_http_tool(tool_name: str, args: dict[str, Any], agent_tools: dict):
    source_db = "unknown"
    if tool_name == "vector_search":
        source_db = "chroma"
    elif tool_name == "graph_query":
        source_db = "neo4j"

    get_client().update_current_span(
        name=f"http-tool:{tool_name}",
        input=args,
        metadata={"tool_name": tool_name, "source_db": source_db},
    )

    tool = agent_tools.get(tool_name)
    if tool is None:
        result = {"error": f"Unknown tool: {tool_name}"}
        get_client().update_current_span(output=result, level="ERROR")
        return result

    try:
        result = tool.execute(args)
        get_client().update_current_span(output=result)
        return result
    except Exception as e:
        result = {"error": f"Tool failed: {tool_name}", "detail": str(e)}
        get_client().update_current_span(output=result, level="ERROR")
        return result

@observe(name="http-query-with-tools")
def llm_call_with_tools(system_prompt: str, user_prompt: str, agent_tools: dict, max_tool_rounds: int = 4) -> str:
    # agent_tools: dict[str, Tool]
    tool_schemas = [tool_to_openai_schema(t) for t in agent_tools.values()]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _extract_retrieved_from_tool_result(result: Any) -> list[dict[str, Any]]:
        extracted: list[dict[str, Any]] = []

        if isinstance(result, dict) and "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                if not isinstance(item, dict):
                    continue
                extracted.append(
                    {
                        "id": item.get("chunk_id") or item.get("id") or "c_?",
                        "text": item.get("text") or "",
                        "score": item.get("score", item.get("distance")),
                        "title": item.get("title") or item.get("source") or "",
                        "source": item.get("source") or "chroma",
                        "meta": item.get("meta") if isinstance(item.get("meta"), dict) else {},
                    }
                )
        elif isinstance(result, dict) and "rows" in result and isinstance(result["rows"], list):
            for idx, row in enumerate(result["rows"]):
                if not isinstance(row, dict):
                    continue
                row_text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                extracted.append(
                    {
                        "id": row.get("chunk_id") or row.get("id") or f"neo4j_{idx}",
                        "text": row_text,
                        "score": None,
                        "title": "Neo4j Graph",
                        "source": "neo4j",
                        "meta": row,
                    }
                )
        elif isinstance(result, list):
            for idx, item in enumerate(result):
                if isinstance(item, dict):
                    extracted.append(
                        {
                            "id": item.get("id") or f"item_{idx}",
                            "text": item.get("text") or " | ".join([f"{k}: {v}" for k, v in item.items()]),
                            "score": item.get("score"),
                            "title": item.get("title") or item.get("source") or "",
                            "source": item.get("source") or "unknown",
                            "meta": item,
                        }
                    )

        return extracted

    tool_results: list[Any] = []
    retrieved_context_accum: list[dict[str, Any]] = []
    called_tools: list[str] = []
    for _ in range(max_tool_rounds):
        try:
            resp = client.chat.completions.create(
                model=QUERY_MODEL,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
            )
        except BadRequestError as e:
            err_txt = str(e)
            # Algunos modelos pueden devolver nombres de tools corruptos/no existentes
            # o producir salida no parseable durante tool calling.
            # Degradamos de forma segura para que el endpoint siga y active fallbacks.
            if (
                "not in request.tools" in err_txt
                or "tool call validation failed" in err_txt
                or "output_parse_failed" in err_txt
                or "generated output that could not be parsed" in err_txt
            ):
                try:
                    print(f"[LLM_TOOLS] tool-calling parse/validation failure; fallback path. error={err_txt}", flush=True)
                except Exception:
                    pass
                return {
                    "assistant_text": "",
                    "tool_results": tool_results,
                    "retrieved_context": retrieved_context_accum,
                    "called_tools": called_tools,
                    "error": "tool_calling_failed_from_model",
                }
            raise

        msg = resp.choices[0].message

        # Si ya respondió normal, terminamos con lo acumulado
        if not getattr(msg, "tool_calls", None):
            # Debug: mostrar resumen de lo recuperado
            try:
                print(
                    f"[LLM_TOOLS] assistant returned; retrieved_context={len(retrieved_context_accum)} items; "
                    f"tool_results={len(tool_results)}",
                    flush=True,
                )
            except Exception:
                pass

            return {
                "assistant_text": msg.content or "",
                "tool_results": tool_results,
                "retrieved_context": retrieved_context_accum,
                "called_tools": called_tools,
            }

        # Guardamos el mensaje del asistente con tool_calls
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        # Ejecutamos cada tool y devolvemos "tool" messages
        for tc in msg.tool_calls:
            name = tc.function.name
            args_raw = tc.function.arguments or {}

            try:
                if isinstance(args_raw, str):
                    args = json.loads(args_raw) if args_raw.strip() else {}
                elif isinstance(args_raw, dict):
                    args = args_raw
                else:
                    args = {"_raw": args_raw}
            except Exception:
                args = {"_raw": args_raw}

            tool = agent_tools.get(name)
            called_tools.append(name)
            if tool is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.execute(args)
                except Exception as e:
                    result = {"error": f"Tool failed: {name}", "detail": str(e)}

            try:
                n_rows = len(result.get("rows")) if isinstance(result, dict) and isinstance(result.get("rows"), list) else None
                n_results = len(result.get("results")) if isinstance(result, dict) and isinstance(result.get("results"), list) else None
                print(
                    f"[LLM_TOOL_RESULT] tool={name} rows={n_rows} results={n_results}",
                    flush=True,
                )
            except Exception:
                pass

            tool_results.append(result)
            extracted = _extract_retrieved_from_tool_result(result)
            if extracted:
                retrieved_context_accum.extend(extracted)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
            
            name = tc.function.name
            print(f"DEBUG: El Agente Query está ejecutando la herramienta: {name}") # <--- AÑADE ESTO
            args_raw = tc.function.arguments or "{}"

    return {
        "assistant_text": "",
        "tool_results": tool_results,
        "retrieved_context": retrieved_context_accum,
        "called_tools": called_tools,
        "error": "Stopped after max_tool_rounds without a final answer.",
    }

def load_some_chunks(path: str, k: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ["chunks", "data", "items"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        return []

    chunks = data[:k]
    norm = []
    for i, c in enumerate(chunks):
        if isinstance(c, str):
            norm.append({"id": f"c_{i}", "text": c, "score": None, "title": ""})
        else:
            norm.append(
                {
                    "id": f"c_{i}",
                    "text": c.get("text") or c.get("content") or "",
                    "score": c.get("score", None),
                    "title": c.get("title") or c.get("source") or "",
                }
            )
    return norm


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=30)  # límite aplicado solo a chunks de Chroma
    show_debug: bool = True


class AskResponse(BaseModel):
    query: str
    verdict: str
    latency_ms: int
    draft_answer: str
    final_answer: str
    eval_json: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    source_distribution: Dict[str, Any]
    called_tools: List[str]
    retrieved_total: int


app = FastAPI(title="RefLens HTTP Demo")

# Static hosting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create agents once
answer_agent = AnswerAgent(llm_call_answer)
eval_agent = EvalAgent(llm_call_judge)
query_agent = QueryAgent(
    chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db"),
    chroma_collection=os.getenv("CHROMA_COLLECTION", "kb_chunks"),
    model=QUERY_MODEL,
)

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/ask")
@observe(name="http-ask")
def ask(req: AskRequest):
    import os, hashlib

    def fp(key: str | None) -> str:
        if not key:
            return "MISSING"
        return hashlib.sha256(key.encode()).hexdigest()[:10]
    
    t0 = time.time()
    get_client().update_current_trace(
        input=req.query,
        metadata={"top_k": req.top_k, "show_debug": req.show_debug},
    )
    out = llm_call_with_tools(
        system_prompt=QUERY_AGENT_SYSTEM,
        user_prompt=req.query,
        agent_tools=query_agent.tools,
        max_tool_rounds=4,  # evita loops infinitos
    )
    called_tools = out.get("called_tools") if isinstance(out, dict) else []

    # Aseguramos la forma esperada por la UI
    retrieved = out.get("retrieved_context") if isinstance(out, dict) else []

    # Si no hubo contexto, fallback a vector_search directo
    if not retrieved:
        try:
            vector_tool = query_agent.tools.get("vector_search")
            if vector_tool:
                vec_out = vector_tool.execute({"query": req.query, "k": 5, "where": None})
                retrieved = vec_out.get("results") if isinstance(vec_out, dict) else []
        except Exception:
            retrieved = []

    # Fallback pragmático: si el LLM no llamó Neo4j y la pregunta parece de entidad, consultamos grafo.
    def _entity_focus_query(q: str) -> str:
        ql = q.strip().lower()
        patterns = [
            r"^\s*who is\s+",
            r"^\s*what is\s+",
            r"^\s*tell me about\s+",
            r"^\s*quien es\s+",
            r"^\s*quién es\s+",
            r"^\s*que es\s+",
            r"^\s*qué es\s+",
            r"^\s*hablame de\s+",
            r"^\s*háblame de\s+",
        ]
        for p in patterns:
            if re.match(p, ql):
                cleaned = re.sub(p, "", ql).strip(" ?!.,;:")
                return cleaned or q.strip()
        return ""

    entity_q = _entity_focus_query(req.query)
    has_neo4j_context = any(
        isinstance(c, dict)
        and isinstance(c.get("source"), str)
        and "neo4j" in c.get("source", "").lower()
        for c in (retrieved or [])
    )
    if entity_q and not has_neo4j_context:
        try:
            graph_tool = query_agent.tools.get("graph_query")
            if graph_tool:
                graph_out = graph_tool.execute({
                    "cypher": """
                        MATCH (e:Entity)
                        WHERE (e.name IS NOT NULL AND toLower(e.name) CONTAINS toLower($q))
                           OR (e.canonical_name IS NOT NULL AND toLower(e.canonical_name) CONTAINS toLower($q))
                           OR (e.aliases IS NOT NULL AND any(a IN e.aliases WHERE toLower(a) CONTAINS toLower($q)))
                        RETURN e.name AS name, e.canonical_name AS canonical_name, e.type AS type, e.aliases AS aliases, e.id AS id
                        LIMIT 20
                    """,
                    "params": {"q": entity_q},
                })
                rows = graph_out.get("rows") if isinstance(graph_out, dict) else []
                try:
                    print(
                        f"[API_ASK_GRAPH_FALLBACK] entity_q={entity_q} rows={len(rows) if isinstance(rows, list) else 0}",
                        flush=True,
                    )
                except Exception:
                    pass
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        name = row.get("name") or row.get("canonical_name") or "Entity"
                        retrieved.append({
                            "id": f"neo4j:{row.get('id')}",
                            "source": "neo4j",
                            "title": name,
                            "score": None,
                            "text": f"Entity({row.get('type')}): {name} | canonical={row.get('canonical_name')} | aliases={row.get('aliases', [])}",
                            "meta": row,
                        })
        except Exception:
            pass

    # Debug: registrar antes de llamar a AnswerAgent
    try:
        print(
            f"[API_ASK] calling AnswerAgent with retrieved={len(retrieved)} chunks; "
            f"tool_results={len(out.get('tool_results') or []) if isinstance(out, dict) else 0}; "
            f"called_tools={called_tools}",
            flush=True,
        )
    except Exception:
        pass
    
    def compact_retrieved(retrieved: list, top_k: int, max_chars: int = 600) -> list:
        out = []
        for c in (retrieved or [])[:top_k]:
            if not isinstance(c, dict):
                continue
            # normaliza nombres de clave (tu chroma usa chunk_id)
            cid = c.get("chunk_id") or c.get("id")
            out.append({
                "chunk_id": cid,
                "source": c.get("source"),
                "score": c.get("score") or c.get("distance"),
                "text": (c.get("text") or "")[:max_chars],
                "meta": c.get("meta") or {"doc_id": c.get("doc_id")},
            })
        return out
    
    def canonical_source_label(source: Any, chunk_id: str, meta: dict[str, Any]) -> str:
        s = str(source or "").strip().lower()
        if "neo4j" in s or str(chunk_id).startswith("neo4j:"):
            return "neo4j"
        if "chroma" in s:
            return "chroma"
        # En vector_search de Chroma, 'source' suele venir como ruta de archivo (epub/pdf/txt/md/json...)
        if s and ("/" in s or "\\" in s or "." in s):
            return "chroma"
        if meta.get("doc_id") or meta.get("collection"):
            return "chroma"
        return "unknown"

    def normalize_chunks(retrieved: list, max_chars: int = 600):
        out = []
        for i, c in enumerate(retrieved or []):
            if not isinstance(c, dict):
                continue
            meta = c.get("meta") or {}
            cid = c.get("chunk_id") or c.get("id") or meta.get("chunk_id") or f"c_{i}"
            raw_source = c.get("source")
            source = canonical_source_label(raw_source, str(cid), meta)
            out.append({
                "id": cid,
                "chunk_id": cid,
                "text": (c.get("text") or "")[:max_chars],
                "source": source,
                "score": c.get("score") if c.get("score") is not None else c.get("distance"),
                "meta": {
                    **meta,
                    "doc_id": c.get("doc_id") or meta.get("doc_id"),
                    "chunk_id": cid,
                    "raw_source": raw_source,
                },
            })
        return out

    retrieved_all = normalize_chunks(retrieved, max_chars=600)

    def limit_only_chroma(chunks: list[dict[str, Any]], chroma_limit: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        chroma_count = 0
        for c in chunks:
            src = c.get("source")
            if src == "chroma":
                if chroma_count >= chroma_limit:
                    continue
                chroma_count += 1
            out.append(c)
        return out

    # Limita solo Chroma; Neo4j (y otros) pasan sin tope.
    retrieved = limit_only_chroma(retrieved_all, req.top_k)

    neo4j_count = sum(1 for c in retrieved_all if c.get("source") == "neo4j")
    chroma_count = sum(1 for c in retrieved_all if c.get("source") == "chroma")
    total_sources = len(retrieved_all)
    other_count = max(total_sources - neo4j_count - chroma_count, 0)
    neo4j_pct = round((neo4j_count / total_sources) * 100) if total_sources else 0
    chroma_pct = round((chroma_count / total_sources) * 100) if total_sources else 0
    other_pct = max(0, 100 - neo4j_pct - chroma_pct) if total_sources else 0

    source_distribution = {
        "total": total_sources,
        "neo4j": {"count": neo4j_count, "pct": neo4j_pct},
        "chroma": {"count": chroma_count, "pct": chroma_pct},
        "other": {"count": other_count, "pct": other_pct},
    }

    try:
        print(
            f"[API_ASK_DEBUG] called_tools={called_tools} total_retrieved={total_sources} "
            f"dist={source_distribution} topk_sources={[c.get('source') for c in retrieved]}",
            flush=True,
        )
    except Exception:
        pass

    # Usar exactamente el mismo subconjunto para answer y eval
    retrieved = select_chunks_for_eval(retrieved)  # type: ignore[assignment]

    # Generar borrador con AnswerAgent
    draft = answer_agent.run(
    req.query,
    retrieved,
    tools_trace={"tool_results": []},  # no duplicar
).draft_answer

    # Pasar por eval
    eval_out = eval_agent.review(
    user_query=req.query,
    draft_answer=draft,
    retrieved_context=retrieved,
    tools_trace={"tool_results": []},
)

    t1 = time.time()

    response = {
        "query": req.query,
        "verdict": eval_out.verdict,
        "latency_ms": int((t1 - t0) * 1000),
        "draft_answer": draft,
        "final_answer": eval_out.answer,
        "eval_json": eval_out.eval_json if req.show_debug else {},
        "chunks": retrieved if req.show_debug else [],
        "source_distribution": source_distribution,
        "called_tools": called_tools if isinstance(called_tools, list) else [],
        "retrieved_total": total_sources,
    }
    get_client().update_current_trace(output=response)
    return response
    
@app.get("/api/debug/chroma")
def debug_chroma(q: str = "Napoleón", k: int = 3):
    tool = query_agent.tools["vector_search"]
    return tool.execute({"query": q, "k": k, "where": None})

@app.get("/api/test-tools")
def test_tools(q: str = "Napoleón"):
    vector_tool = query_agent.tools["vector_search"]
    graph_tool = query_agent.tools["graph_query"]

    v = vector_tool.execute({"query": q, "k": 3, "where": None})
    g = graph_tool.execute({
        "cypher": """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($q)
            RETURN e.name AS name
            LIMIT 5
        """,
        "params": {"q": q},
    })

    return {"vector": v, "graph": g}
