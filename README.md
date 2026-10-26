# RAG Project

## Goals

- Review what we've learned, and fill the gaps, in **FastAPI, LangChain, LangGraph, and RAG**.
- Build a complete RAG system end to end, understand how a RAG application is put together, and customize it for our own database.
- Build technical depth and critical thinking:
  - Understand what every function and line does. For example:
    - What does `sum([int(i) for i in ["5", "6", "seven"] if len(i) <= 2])` return, and why doesn't it crash?
    - What does `yield` do?
  - Using AI to generate code is fine, but expect to explain every detail of it.

## Teams

- **Judge:** Huy
- **Team 1:** Hieu (Captain), Trang. Code in `members/team_hieu_trang/`. Plugs into the evaluation as `/team1/rag-agent` by implementing `team1_agent` in `api_view.py`.
- **Team 2:** Ha (Captain), Tri. Code in `members/team_tri_ha/`. Plugs into the evaluation as `/team2/rag-agent` by implementing `team2_agent` in `api_view.py`.
- **Team 3:** Huy. Code in `members/team_huy/`. Runs as `/team3/rag-agent` (`team3_agent` in `api_view.py`). Huy's own entry, not on the leaderboard, currently a simple keyword-retrieval placeholder on Gemma 3.

Each team has its own channel for discussion, by text or meeting.

## Rules

- Both teams use the **same document set**, provided by Huy. For papers, everyone uses the same arXiv version.
- **LLM:** Gemma 3 4B, run offline through Ollama, for both teams.
- **LangGraph is required** for orchestrating the pipeline. FastAPI serves the system. Any embedding model or retrieval technique is allowed.
- Both teams use the **same GitHub repo**, and each team works in its own folder (`team1/`, `team2/`).
- There are no required milestones. Teams work at their own pace.

## Weekly Meeting (Sundays: Oct 18, Oct 25, Nov 1, Nov 8)

For each team:

1. **Presentation (20 min):** Present what you've built so far, without interruption.
2. **Q&A (20 min):** The other team asks questions, in as much detail as they want, down to individual lines of code. Examples:
   - High level: How is the LangGraph workflow structured? Why this chunking strategy or retriever and not another?
   - Low level: What does `yield` do? What does `uvicorn huy:app --reload` do?

"I don't know, I'll find out by next week" is an acceptable answer, as long as you follow up the next week.

## Evaluation

### Question Set (created by Huy)

| Type | Count |
|---|---|
| Answer comes from 1 document | 20 |
| Answer requires 2+ documents | 20 |
| Unanswerable (not in the documents) | 5 |

- **Dev set:** 10 sample questions with answers, shared with both teams for development.
- **Test set:** the rest, kept hidden.

### Weekly Practice Evaluation (Optional)

- Each team may run the test set **once per week** to check its progress.
- Practice scores don't count toward the final result.

### Final Evaluation (Nov 8)

Only this evaluation counts. The score rewards correct answers and penalizes hallucination. **Saying "I don't know" is always better than making something up.**

| Model's response | Points |
|---|---|
| Correct answer | +1 |
| Partially correct answer | +0.5 |
| "I don't know" / "Not in the documents" | 0 |
| Hallucinated (wrong or made-up) answer | −0.5 |

**For unanswerable questions:**

| Model's response | Points |
|---|---|
| Correctly says it's not in the documents | +1 |
| Gives any answer | −0.5 (hallucination) |

Retrieval quality, meaning whether the correct source document(s) were retrieved, is reported to each team as feedback but doesn't count toward the score.

### Scoring Examples

**Documents used in these examples:**
- **Doc A:** *Attention Is All You Need* (Vaswani et al., 2017)
- **Doc B:** *Harnessing the Universal Geometry of Embeddings* (vec2vec; Jha, Zhang, Shmatikov, Morris)

**Single-document question (Doc A)**

**Q:** Why does the Transformer divide by √d_k in scaled dot-product attention?
**Ground truth:** For large values of d_k, the dot products grow large in magnitude and push the softmax into regions with extremely small gradients. Scaling by 1/√d_k counteracts this.

| Model's answer | Result | Points |
|---|---|---|
| "Large d_k makes dot products large, which pushes softmax into regions with tiny gradients. Dividing by √d_k fixes that." | Correct | +1 |
| "To normalize the values so training is more stable." | Partially correct (vague, misses the softmax/gradient reason) | +0.5 |
| "I don't know." | Abstained | 0 |
| "To reduce the memory cost of storing the attention matrix." | Hallucinated | −0.5 |

**Single-document question (Doc B)**

**Q:** What training losses does vec2vec use to learn its translation?
**Ground truth:** Adversarial losses and cycle consistency, used to encode embeddings into a shared latent space and decode them with minimal loss. The paper also uses a vector space preservation loss to keep pairwise relationships between embeddings consistent under translation.

| Model's answer | Result | Points |
|---|---|---|
| "Adversarial loss and cycle-consistency loss, plus a loss that preserves pairwise relationships between embeddings." | Correct | +1 |
| "It uses adversarial training." | Partially correct (misses cycle consistency) | +0.5 |
| "It's trained with contrastive loss on paired sentences." | Hallucinated (vec2vec uses no paired data) | −0.5 |

**Multi-document question (Doc A + Doc B)**

**Q:** Compare the training data each method needs. Does each one require paired examples?
**Ground truth:** The Transformer is trained on parallel sentence pairs (WMT 2014 English–German and English–French), so it requires paired data (Doc A). vec2vec is the first method to translate embeddings from one model's space to another without paired data (Doc B).

| Model's answer | Result | Points |
|---|---|---|
| "The Transformer needs paired source/target sentences from WMT 2014. vec2vec needs no paired data and learns the mapping unsupervised." | Correct | +1 |
| "vec2vec doesn't need paired data." | Partially correct (only used Doc B) | +0.5 |
| "The documents don't give enough information to compare them." | Abstained | 0 |
| "Both need paired data, since both are translation methods." | Hallucinated | −0.5 |

**Unanswerable question**

**Q:** What BLEU score does vec2vec achieve on WMT 2014 English–German?
**Ground truth:** Not in the documents. vec2vec translates embeddings between models, not text between languages, and isn't evaluated with BLEU.

| Model's answer | Result | Points |
|---|---|---|
| "vec2vec isn't evaluated on WMT or with BLEU. It translates embeddings between model spaces, not languages." | Correct | +1 |
| "vec2vec achieves 28.4 BLEU on English–German." | Hallucinated (borrowed the Transformer's number) | −0.5 |

## Final Score

- **60%:** final evaluation score
- **40%:** Q&A performance across all meetings, judged by Huy on how well each member explains their own system

## Timeline

- **Start:** Sunday, October 11
- **Weekly meetings:** October 18, October 25, November 1
- **Final evaluation and presentation:** Sunday, November 8

## Final Deliverables (Nov 8)

- The team's folder in the shared repo, with a README explaining the architecture
- Live demo
- Final evaluation score
- Final presentation and Q&A