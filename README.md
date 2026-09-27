# cockroach-survivor-learn

Paired learning plan. Two groups of two, one course each, weekly pace balanced by difficulty (not just runtime).

| Group | Members | Course |
|---|---|---|
| Group 1 | Hieu, Ha | [LangChain](https://www.udemy.com/course/langchain/) |
| Group 2 | Huy, Tri | [FastAPI - The Complete Course 2026](https://www.udemy.com/course/fastapi-the-complete-course/) |

---

## Group 1: Hieu & Ha

- **Core:** through "Let's Talk About LLM Applications In Production" — 10h08m
- **Extended:** LangGraph (weeks 5–6) and MCP Server / Deep Agent tooling (week 7) — +6h45m
- **Covers:** LangChain, agents & function calling, RAG/vector databases, basic deployment, LangGraph, MCP servers

| Week | Sections | Time |
|---|---|---|
| 1 — Foundations | Introduction · GIST of LangChain · GIST of AI Agents · Agents Under The Hood (1/4) · [Layer 1] ReAct Loop | 2h54m |
| 2 — Function Calling & RAG | [Layer 2] Raw Function Calling · [Layer 3] ReAct Prompt · Function Calling · GIST of RAG | 3h00m |
| 3 — Applied Project | Building a documentation assistant | 2h16m |
| 4 — Prompt Eng & Production | Prompt Engineering Theory · LLM Applications In Production | 2h00m |
| 5 — LangGraph (Part 1) | Introduction to LangGraph | 2h15m |
| 6 — LangGraph (Part 2) | Introduction to LangGraph | 2h15m |
| 7 — MCP Server | Useful tools when developing LLM applications · Deep Agent | 2h15m |

---

## Group 2: Huy & Tri

- **Core:** whole course minus skips — 13h09m
- **Skip:** Python Installation & Refresher, Git - Version Control, (Deprecated FastAPI) section
- **Covers:** FastAPI routing, auth, database, testing, full-stack project, deployment

| Week | Sections | Time |
|---|---|---|
| 1 — Basics & Core Projects | Introduction · Overview · Setup & Installation · Project 1 (Request Method Logic) · Project 2 (Move Fast) · Project 3 (RESTful APIs) · Setup Database | 4h16m |
| 2 — API Methods | API Request Methods | 2h54m |
| 3 — Auth | Authentication & Authorization · Authenticate Requests | 2h00m |
| 4 — Production DB & Migrations | Large Production Database Setup · Project 3.5 (Alembic Migration) | 2h10m |
| 5 — Testing | Project 4 (Unit & Integration Testing) | 2h21m |
| 6 — Full Stack Capstone & Deployment | Project 5 (Full Stack Application) · Deploying FastAPI Applications | 2h03m |

---

## Team Agent Project

### Purpose

1. Combine what we learn from the LangChain and FastAPI courses
2. Build an agent that is useful for the group

### Deadline: 4 weeks

**Flow 1 — Data ingestion:**
Upload data to shared folder → Ingestion Pipeline (manually/automatically) → vector database.

**Flow 2 — Usage:**
Ask question → Knowledge Retrieval from Vector Database → Answer Question

**Use cases:**

1. What did our team cover last week?
2. What is the Cockroach project about?
3. What is LangChain, what is Alembic, and what are their references?
4. Add a customized tool (quiz maker)
5. How do you downgrade a database using Alembic? Provide a code example — Tri's PPT and Tri's code as an example.

**Architecture:**

1. **Shared folder** (managed by everyone)
   - Google Drive: blob data (PPT, docs, PDF)
   - GitHub (code) — next phase
2. **Vector database:** FAISS, Pinecone, SQL (each person can use their own method)
3. **Retrieval:** [Hugging Face](https://huggingface.co/), Ollama — research more about models

**Next steps:**

1. Everyone uploads something to the shared database
2. Start building and push your code to the [GitHub repo](https://github.com/williamhuybui/cockroach-survivor-learn)
3. 30 minutes to show what you've built/plan at the next meeting

**Rules:**

- Use only resources from our two courses
- Limit AI usage
