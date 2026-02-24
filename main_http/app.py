import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI
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

    tool_results = []
    retrieved_context: list[dict[str, Any]] = []
    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",   # ✅ CLAVE
        )

        msg = resp.choices[0].message

        # Si ya respondió normal, terminamos con lo acumulado
        if not getattr(msg, "tool_calls", None):
            # Debug: mostrar resumen de lo recuperado
            try:
                print(f"[LLM_TOOLS] assistant returned; retrieved_context={len(retrieved_context)} items; tool_results={len(tool_results)}", flush=True)
            except Exception:
                pass

            return {
                "assistant_text": msg.content or "",
                "tool_results": tool_results,
                "retrieved_context": retrieved_context,
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
            if tool is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.execute(args)
                except Exception as e:
                    result = {"error": f"Tool failed: {name}", "detail": str(e)}

            tool_results.append(result)
            retrieved_context.extend(_extract_retrieved_from_tool_result(result))

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
        "retrieved_context": retrieved_context,
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
    t0 = time.time()
    out = llm_call_with_tools(
        system_prompt=QUERY_AGENT_SYSTEM,
        user_prompt=req.query,
        agent_tools=query_agent.tools,
        max_tool_rounds=4,  # evita loops infinitos
    )

    # Aseguramos la forma esperada por la UI
    retrieved = out.get("retrieved_context") if isinstance(out, dict) else []

    # Si no hubo contexto, fallback a vector_search directo
    if not retrieved:
        try:
            vector_tool = query_agent.tools.get("vector_search")
            if vector_tool:
                vec_out = vector_tool.execute({"query": req.query, "k": req.top_k, "where": None})
                retrieved = vec_out.get("results") if isinstance(vec_out, dict) else []
        except Exception:
            retrieved = []

    # Debug: registrar antes de llamar a AnswerAgent
    try:
        print(f"[API_ASK] calling AnswerAgent with retrieved={len(retrieved)} chunks; tool_results={len(out.get('tool_results') or []) if isinstance(out, dict) else 0}", flush=True)
    except Exception:
        pass

    # Generar borrador con AnswerAgent
    draft = answer_agent.run(req.query, retrieved, tools_trace={"tool_results": out.get("tool_results") if isinstance(out, dict) else []}).draft_answer

    # Pasar por eval
    eval_out = eval_agent.review(
        user_query=req.query,
        draft_answer=draft,
        retrieved_context=retrieved,
        tools_trace={"tool_results": out.get("tool_results") if isinstance(out, dict) else [], "pipeline": "query->answer->eval"},
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