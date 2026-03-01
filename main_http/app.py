import os
import time
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI, BadRequestError
from src.rag.config import settings

# Importa tus agentes (ajusta el path según dónde los tengas)
from src.reflens.answer.answer_agent import AnswerAgent
from src.reflens.query.query_agent import QueryAgent, QUERY_AGENT_SYSTEM
from src.reflens.eval_agent import EvalAgent

import json

load_dotenv()

GROQ_API_ENDPOINT = os.getenv("GROQ_API_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_API_ENDPOINT)

def llm_call(system_prompt: str, user_prompt: str) -> str:
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return r.choices[0].message.content

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

def llm_call_with_tools(system_prompt: str, user_prompt: str, agent_tools: dict, max_tool_rounds: int = 4) -> str:
    # agent_tools: dict[str, Tool]
    tool_schemas = [tool_to_openai_schema(t) for t in agent_tools.values()]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tool_results = []
    called_tools: list[str] = []
    for _ in range(max_tool_rounds):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
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
                    "retrieved_context": [],
                    "called_tools": called_tools,
                    "error": "tool_calling_failed_from_model",
                }
            raise

        msg = resp.choices[0].message

        # Si ya respondió normal, terminamos: recolectamos tool outputs
        if not getattr(msg, "tool_calls", None):
            for m in messages:
                if m.get("role") == "tool":
                    try:
                        parsed = json.loads(m.get("content") or "null")
                    except Exception:
                        parsed = m.get("content")
                    tool_results.append(parsed)

            # Extraemos contextos tipo chunks desde las respuestas de tools
            retrieved_context: list = []

            for r in tool_results:
                # Chroma: ya viene bien
                if isinstance(r, dict) and "results" in r and isinstance(r["results"], list):
                    retrieved_context.extend(r["results"])

                # Neo4j: rows -> convertir a pseudo-chunks para UI
                elif isinstance(r, dict) and "rows" in r and isinstance(r["rows"], list):
                    for row in r["rows"]:
                        if not isinstance(row, dict):
                            continue

                        # Entidad (tiene 'id' y 'type')
                        if "id" in row and "type" in row:
                            name = row.get("name") or row.get("canonical_name") or "Entity"
                            retrieved_context.append({
                                "id": f"neo4j:{row.get('id')}",
                                "source": "neo4j",
                                "title": name,
                                "name": name,
                                "score": None,
                                "text": f"Entity({row.get('type')}): {name} | canonical={row.get('canonical_name')} | aliases={row.get('aliases', [])}",
                                "meta": row,
                            })

                        # Relación (tiene 'rel' y 'target')
                        elif "rel" in row and "target" in row:
                            retrieved_context.append({
                                "id": f"neo4j:rel:{row.get('target')}",
                                "source": "neo4j",
                                "title": "REL",
                                "score": None,
                                "text": f"REL -> {row.get('target')}",
                                "meta": row,
                            })

                elif isinstance(r, list):
                    retrieved_context.extend(r)

            # Debug: mostrar resumen de lo recuperado
            try:
                print(f"[LLM_TOOLS] assistant returned; retrieved_context={len(retrieved_context)} items; tool_results={len(tool_results)}", flush=True)
            except Exception:
                pass

            return {
                "assistant_text": msg.content or "",
                "tool_results": tool_results,
                "retrieved_context": retrieved_context,
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
            args_raw = tc.function.arguments or "{}"

            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
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
        "retrieved_context": [],
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
    top_k: int = Field(5, ge=1, le=30)
    show_debug: bool = True


class AskResponse(BaseModel):
    query: str
    verdict: str
    latency_ms: int
    draft_answer: str
    final_answer: str
    eval_json: Dict[str, Any]
    chunks: List[Dict[str, Any]]


app = FastAPI(title="RefLens HTTP Demo")

# Static hosting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create agents once
answer_agent = AnswerAgent(llm_call)
eval_agent = EvalAgent(llm_call)
query_agent = QueryAgent(
    chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db"),
    chroma_collection=os.getenv("CHROMA_COLLECTION", "kb_chunks"),
    model=GROQ_MODEL,
)

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/ask")
def ask(req: AskRequest):
    import os, hashlib

    def fp(key: str | None) -> str:
        if not key:
            return "MISSING"
        return hashlib.sha256(key.encode()).hexdigest()[:10]
    
    t0 = time.time()
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
    used_graph = isinstance(called_tools, list) and ("graph_query" in called_tools)
    if entity_q and not used_graph:
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
    
    def normalize_chunks(retrieved: list, top_k: int, max_chars: int = 600):
        out = []
        for i, c in enumerate((retrieved or [])[:top_k]):
            if not isinstance(c, dict):
                continue
            meta = c.get("meta") or {}
            cid = c.get("chunk_id") or c.get("id") or meta.get("chunk_id") or f"c_{i}"
            out.append({
                "id": cid,
                "chunk_id": cid,
                "text": (c.get("text") or "")[:max_chars],
                "source": c.get("source"),
                "score": c.get("score") if c.get("score") is not None else c.get("distance"),
                "meta": {
                    **meta,
                    "doc_id": c.get("doc_id") or meta.get("doc_id"),
                    "chunk_id": cid,
                },
            })
        return out
    
    retrieved = normalize_chunks(retrieved, req.top_k, max_chars=600)

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

    return {
        "query": req.query,
        "verdict": eval_out.verdict,
        "latency_ms": int((t1 - t0) * 1000),
        "draft_answer": draft,
        "final_answer": eval_out.answer,
        "eval_json": eval_out.eval_json if req.show_debug else {},
        "chunks": retrieved if req.show_debug else [],
    }
    
@app.get("/api/debug/chroma")
def debug_chroma(q: str = "Napoleón", k: int = 3):
    tool = next(t for t in query_agent.tools if t.name == "vector_search")
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
