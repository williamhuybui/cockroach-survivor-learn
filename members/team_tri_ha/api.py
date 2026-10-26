
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.agent import Agent


app = FastAPI(title="Team Tri-Ha Research RAG Agent")

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


class RAGRequest(BaseModel):
    query: str = Field(min_length=1)


class RAGResponse(BaseModel):
    answer: str
    context: str


@lru_cache(maxsize=1)
def load_rag():
    return Agent(multi_mode="per-paper")


@app.post("/team-tri-ha/rag-agent", response_model=RAGResponse)
@app.post("/team1/rag-agent", response_model=RAGResponse)
def rag_agent(request: RAGRequest):
    try:
        agent = load_rag()

        result = agent.ask(request.query)

        answer = result["answer"]
        context = "\n\n".join(result["context"])

        return RAGResponse(
            answer=answer,
            context=context
        )

    except Exception as exc:
        print(f"RAG error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="RAG processing failed. Check the server logs."
        ) from exc
