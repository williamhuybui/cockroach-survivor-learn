from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser

from rag import config
from rag.models import get_embeddings, get_llm, get_vectorstore
from phase2_ask import PROMPT, format_context


app = FastAPI(title="Team Tri-Ha Research RAG Agent")


class RAGRequest(BaseModel):
    query: str = Field(min_length=1)


class RAGResponse(BaseModel):
    answer: str
    context: str


@lru_cache(maxsize=1)
def load_rag():
    embeddings = get_embeddings(fake=False)
    store = get_vectorstore(
        embeddings,
        config.chroma_dir(fake=False)
    )
    llm = get_llm(fake=False)
    return store, llm


@app.post("/team-tri-ha/rag-agent", response_model=RAGResponse)
@app.post("/team1/rag-agent", response_model=RAGResponse)
def rag_agent(request: RAGRequest):
    try:
        store, llm = load_rag()

        results = store.similarity_search_with_score(
            request.query,
            k=config.TOP_K
        )

        docs = [doc for doc, score in results]
        context = format_context(docs)

        if not docs:
            return RAGResponse(
                answer="I could not find this in the documents.",
                context=""
            )

        chain = PROMPT | llm | StrOutputParser()

        answer = chain.invoke({
            "context": context,
            "question": request.query
        })

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
