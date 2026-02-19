# main_qa.py
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from src.reflens.answer_agent import AnswerAgent
from src.reflens.qa_agent import QAAgent

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


def load_some_chunks(path="chunks.json", k=5):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        chunks = data[:k]
    elif isinstance(data, dict):
        for key in ["chunks", "data", "items"]:
            if key in data and isinstance(data[key], list):
                chunks = data[key][:k]
                break
        else:
            chunks = []
    else:
        chunks = []

    norm = []
    for i, c in enumerate(chunks):
        if isinstance(c, str):
            norm.append({"id": f"c_{i}", "text": c, "score": "", "title": ""})
        else:
            norm.append(
                {
                    "id": f"c_{i}",
                    "text": c.get("text") or c.get("content") or "",
                    "score": c.get("score", ""),
                    "title": c.get("title") or c.get("source") or "",
                }
            )
    return norm


if __name__ == "__main__":
    retrieved = load_some_chunks("chunks.json", k=5)
    user_query = "¿Qué se sabe de Napoleón?"

    answer_agent = AnswerAgent(llm_call)
    qa_agent = QAAgent(llm_call)

    draft = answer_agent.run(user_query, retrieved, tools_trace={"k": 5}).draft_answer

    out = qa_agent.review(
        user_query=user_query,
        draft_answer=draft,
        retrieved_context=retrieved,
        tools_trace={"k": 5, "pipeline": "answer->qa"},
    )

    print("\n=== DRAFT (AnswerAgent) ===\n")
    print(draft)

    print("\n=== VERDICT (QA) ===")
    print(out.verdict)

    print("\n=== FINAL ANSWER (post-QA) ===\n")
    print(out.answer)

    print("\n=== QA JSON ===")
    print(json.dumps(out.qa_json, ensure_ascii=False, indent=2))