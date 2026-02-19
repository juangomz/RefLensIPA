import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

# Importa tus agentes (ajusta el path según dónde los tengas)
from src.reflens.answer_agent import AnswerAgent
from src.reflens.qa_agent import QAAgent

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
    qa_json: Dict[str, Any]
    chunks: List[Dict[str, Any]]


app = FastAPI(title="RefLens HTTP Demo")

# Static hosting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create agents once
answer_agent = AnswerAgent(llm_call)
qa_agent = QAAgent(llm_call)

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()

    # Por ahora: retrieval desde chunks.json (luego lo cambias por vuestro retriever real)
    repo_root = os.path.abspath(os.path.join(BASE_DIR, ".."))
    chunks_path = os.path.join(repo_root, "chunks.json")
    retrieved = load_some_chunks(chunks_path, req.top_k)

    # Answer -> QA
    draft = answer_agent.run(req.query, retrieved, tools_trace={"k": req.top_k}).draft_answer
    qa_out = qa_agent.review(
        user_query=req.query,
        draft_answer=draft,
        retrieved_context=retrieved,
        tools_trace={"k": req.top_k, "pipeline": "answer->qa"},
    )

    t1 = time.time()

    return AskResponse(
        query=req.query,
        verdict=qa_out.verdict,
        latency_ms=int((t1 - t0) * 1000),
        draft_answer=draft,
        final_answer=qa_out.answer,
        qa_json=qa_out.qa_json if req.show_debug else {},
        chunks=retrieved if req.show_debug else [],
    )