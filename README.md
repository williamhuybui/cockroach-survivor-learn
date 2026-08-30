# cockroach-survivor-learn

A 4-week paired learning plan for the team to level up on LLM application development (LangChain/LangGraph) and backend/API skills (FastAPI). The team is split into two groups of two so each pair can learn together, unblock each other, and check in on progress.

## Groups

| Group | Members | Focus |
|---|---|---|
| Group 1 | Hieu, Ha | LangChain, vector databases, deployment basics |
| Group 2 | Huy, Tri | LangChain/LangGraph for agentic AI, FastAPI, UI/DB basics, deployment |

**Timeline:** 4 weeks for both groups.

**Suggested cadence:** pair up for study sessions (live or async), and do a short weekly sync (15–30 min) to share progress, blockers, and key takeaways.

---

## Group 1: Hieu & Ha

**Course:** [LangChain](https://www.udemy.com/course/langchain/)

- **Core content:** Watch through **"Let's Talk About LLM Applications In Production"** — 10h 8min
- **Stretch goal (if time allows):** Introduction to LangGraph — 1h 23min
- **Further stretch (optional, beyond the 4-week goal):** Reflection Agent, Reflexion Agent, Agentic RAG — ~3h more
- **Total time budget:** 4 weeks (~10h core, ~11.5h with the LangGraph stretch)
- **What this covers:** using LangChain, agents & function calling, working with vector databases (RAG), and the basics of deploying an LLM application

### Suggested weekly pace

Split by **effort** (runtime × difficulty), not just runtime — dense/hard sections get comparatively more calendar time even when their video length is short. See [methodology](#pacing-methodology) below.

| Week | Sections | Difficulty | Time |
|---|---|---|---|
| 1 — Foundations | Introduction · The GIST of LangChain ("Hello World" chain) · THE GIST of AI Agents · Agents Under The Hood (1/4) · [Layer 1] The ReAct Loop | Easy → Medium ramp | 2h54m |
| 2 — Function Calling & RAG Theory | [Layer 2] Raw Function Calling · [Layer 3] The ReAct Prompt · Function Calling · The GIST of RAG (Embeddings, Vector DBs, & Retrieval) | Medium–Hard (dense theory, the [Layer 3] ReAct Prompt section especially) | 3h00m |
| 3 — Applied Project | Building a documentation assistant (Embeddings, VectorDBs, Retrieval, Memory) | Hard (biggest hands-on build — everything from weeks 1–2 comes together) | 2h16m |
| 4 — Prompt Eng, Production & Stretch | Prompt Engineering Theory · Let's Talk About LLM Applications In Production · **Stretch (if ahead of schedule):** Introduction to LangGraph | Medium (lighter core load — deliberately leaves room for the stretch goal) | 1h58m (+1h23m stretch) |

**Total:** 10h08m core (+1h23m stretch) — same total as before, just re-cut so no single week stacks too much hard content at once.

---

## Group 2: Huy & Tri

**Courses:**
- LangChain — Agentic AI Engineering with LangChain & LangGraph *(curriculum breakdown pending — pace below is an estimate)*
- [FastAPI - The Complete Course 2026 (Beginner + Advanced)](https://www.udemy.com/course/fastapi-the-complete-course/)

- **FastAPI course — real breakdown (confirmed from curriculum):**
  - Skip: Python Installation & Refresher (3h58min), Git - Version Control (1h05min), (Deprecated FastAPI) section
  - *The "(Deprecated FastAPI)" section wasn't visible in the curriculum captured so far (likely further down the syllabus) — skip it regardless once you reach it; its runtime isn't in the totals below since we haven't confirmed its length yet.*
  - Effective watch time (visible sections, skips removed): **13h09min** (does not yet subtract the deprecated section's runtime)
- **LangChain/LangGraph course:** ~3.5h (estimate, unchanged until we have its real curriculum)
- **Total time budget:** 4 weeks — combined effective time (~16h39min) is more than originally estimated, so pace is faster than before (~4h10min/week vs. the original ~3.3h/week)
- **What this covers:** building APIs with FastAPI (routing, auth, database, testing), a full-stack project, and deployment basics

### Suggested weekly pace

Split by **effort** (runtime × difficulty), not just runtime — see [methodology](#pacing-methodology) below. This moved "API Request Methods" earlier (into week 2) and pushed the Auth/DB/Migration cluster into its own week, since that cluster is disproportionately hard for its runtime.

| Week | Sections | Difficulty | Time |
|---|---|---|---|
| 1 — Agentic AI Foundations | LangChain/LangGraph agentic content *(estimate — real breakdown TBD)* | Unknown — revisit once curriculum is confirmed | ~3.5h |
| 2 — FastAPI Basics & Core Projects | Introduction · FastAPI Overview · FastAPI Setup & Installation · Project 1 (Request Method Logic) · Project 2 (Move Fast with FastAPI) · Project 3 (Complete RESTful APIs) · Setup Database · API Request Methods | Easy → Medium ramp | 4h50m |
| 3 — Auth, Production DB & Migrations | Authentication & Authorization · Authenticate Requests · Large Production Database Setup · Project 3.5 (Alembic Data Migration) | Hard — this is the densest cluster in the course relative to its runtime; budget extra debugging time | 3h55m |
| 4 — Testing, Full Stack & Deployment | Project 4 (Unit & Integration Testing) · Project 5 (Full Stack Application) · Deploying FastAPI Applications | Hard — heaviest week overall; it's the capstone, so it can't realistically move earlier. Start it early if week 3 finishes ahead of schedule. | 4h24m |

**Total:** ~16h39m (~3.5h agentic + 13h09m FastAPI) — same total as before, re-cut so the hardest FastAPI content (Auth) isn't crammed alongside Testing/Full-Stack in the same week.

---

## Pacing methodology

Weekly cuts aren't based on raw video runtime alone — a short section can still be a slog (dense theory, fiddly setup, first exposure to a new concept), while a long section can be easy if it's mostly walkthrough/repetition. To account for this:

1. Each course section got a rough difficulty tag (easy / medium / hard) based on the type of content — conceptual density, whether it's brand-new vs. building on prior sections, and how much debugging/practice it typically needs beyond just watching.
2. "Effort" = runtime × difficulty weight, and weeks were cut to balance **effort**, not minutes — while keeping the original course order intact, so nothing gets watched out of sequence.
3. In practice this means: hard/dense clusters (e.g. Group 1's ReAct/function-calling theory, Group 2's Auth & production DB cluster) get their own lighter-runtime week instead of being padded with easier filler, and easier stretches get bundled together even if that week's total runtime looks longer on paper.

This is a judgment call, not a precise formula — if a pair finds a "light" week is actually harder than expected (or vice versa), flag it at the weekly sync and we can re-cut the following weeks.

## Notes

- Pace tables are guidelines — adjust based on each pair's schedule, but aim to stay on track to finish within 4 weeks.
- Stretch content (LangGraph for Group 1) is a bonus, not a requirement, for finishing on time.
- Group 2's LangChain/LangGraph agentic course still needs its real curriculum confirmed — revisit Group 2's week 1 once it's available. The "(Deprecated FastAPI)" section is confirmed skip regardless of where it falls in the syllabus; once located, subtract its runtime from the totals above.
- Consider a short end-of-course show-and-tell where each group shares what they built/learned.
