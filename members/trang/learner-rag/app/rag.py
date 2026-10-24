
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END

from app.drive import BASE_DIR


# 1. Load configuration
load_dotenv(BASE_DIR / ".env")

EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma3")

CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "research_papers"


# 2. Connect to ChromaDB
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    persist_directory=str(CHROMA_DIR),
)


# 3. Initialize Gemma 3
llm = ChatOllama(
    model=CHAT_MODEL,
    temperature=0,
)


# 4. Define the shared workflow state
class RAGState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state: RAGState):
    results = vector_store.similarity_search_with_score(
        state["question"],
        k=10,
    )

    context_parts = []

    for document, score in results:
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        context_parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return {"context": context}

# def retrieve(state: RAGState):
#     results = vector_store.similarity_search_with_score(
#         state["question"],
#         k=10,
#     )

#     print("\n===== RETRIEVAL DEBUG =====")
#     print("Question:", state["question"])

#     for i, (document, score) in enumerate(results, start=1):
#         print(f"\n--- Result {i} | Distance: {score:.4f} ---")
#         print("Source:", document.metadata.get("source"))
#         print("Page:", document.metadata.get("page"))
#         print("Text:", document.page_content)

#     context = "\n\n".join(
#         document.page_content for document, score in results
#     )

#     return {"context": context}


# 6. Conditional routing
def check_context(state: RAGState):
    if state["context"].strip():
        return "generate"

    return "no_context"


# 7. Generate an answer with Gemma 3
def generate(state: RAGState):
    prompt = f"""
You are answering questions about research papers.

Answer the question using the provided context.
Prioritize passages that directly answer the question.
Do not substitute details about training data, model size,
or evaluation metrics for a question about loss functions.

If the context contains the answer, explain it clearly.
If the answer is not in the context, say you don't know.

Context:
{state["context"]}

Question:
{state["question"]}

Answer:
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


# 8. Fallback if no context is found
def no_context(state: RAGState):
    return {
        "answer": "I couldn't find relevant information in the documents."
    }


# 9. Build the LangGraph workflow
builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("no_context", no_context)

builder.add_edge(START, "retrieve")

builder.add_conditional_edges(
    "retrieve",
    check_context,
    {
        "generate": "generate",
        "no_context": "no_context",
    },
)

builder.add_edge("generate", END)
builder.add_edge("no_context", END)

rag_graph = builder.compile()


# 10. Function for FastAPI to call
def ask_question(question: str) -> str:
    result = rag_graph.invoke({
        "question": question,
        "context": "",
        "answer": "",
    })

    return result["answer"]


# 11. Test the RAG workflow directly
if __name__ == "__main__":
    question = "Why does the Transformer divide by √d_k in scaled dot-product attention?"
    answer = ask_question(question)

    print("Question:", question)
    print("\nAnswer:", answer)

    question =  "What training losses does vec2vec use to learn its translation? "
    answer = ask_question(question)

    print("Question:", question)
    print("\nAnswer:", answer)

    question =  "Compare the training data each method needs. Does each one require paired examples?"
    answer = ask_question(question)

    print("Question:", question)
    print("\nAnswer:", answer)


    # print("Matching chunks:", len(results["documents"]))

    # for document, metadata in zip(
    #     results["documents"],
    #     results["metadatas"],
    # ):
    #     print("\nSource:", metadata.get("source"))
    #     print("Page:", metadata.get("page"))
    #     print("Text:", document)

