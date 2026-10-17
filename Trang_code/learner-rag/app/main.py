
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import ask_question

# Create the FastAPI application
app = FastAPI(
    title="Research Paper RAG API",
    description="Ask questions about research papers using RAG.",
    version="1.0.0",
)


# Define the format of an incoming question
class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Your question about the research papers",
    )


# Define the format of the API response
class AnswerResponse(BaseModel):
    answer: str


# Create the question endpoint
@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        answer = ask_question(question)
        return AnswerResponse(answer=answer)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer question: {str(error)}",
        ) from error


# Simple endpoint to check whether the API is running
@app.get("/health")
def health():
    return {"status": "ok"}