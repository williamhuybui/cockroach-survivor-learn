"""Fast offline regression checks; no Ollama, Chroma, or API needed."""
import unittest
from unittest.mock import patch
import sys
import types
sys.modules.setdefault("app.llm", types.SimpleNamespace(generate=lambda *args, **kwargs: None))
from app.grounding import DraftAnswer, SupportedClaim, verify_claims, needs_numerical_consistency, format_answer, draft_answer

class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.evidence=[{"id":"S1", "content":"The evaluated models are MPT-30B-Instruct and LongChat-13B (16K).", "task_ids":[0], "title":"paper.pdf"}]
        self.tasks=[{"question":"Which models?", "paper":None}]

    def test_quote_verification(self):
        draft=DraftAnswer(status="answered",claims=[SupportedClaim(text="The models include MPT-30B-Instruct and LongChat-13B (16K).",source_id="S1",quote="MPT-30B-Instruct and LongChat-13B (16K)",task_ids=[0])])
        self.assertEqual(len(verify_claims(draft,self.evidence,self.tasks,question="Which models?")["accepted"]),1)

    def test_reject_fake_quote(self):
        draft=DraftAnswer(status="answered",claims=[SupportedClaim(text="The models include Claude.",source_id="S1",quote="The evaluated models include Claude.",task_ids=[0])])
        result=verify_claims(draft,self.evidence,self.tasks,question="Which models?")
        self.assertEqual(result["rejected_count"],1)
        self.assertIn("quote_not_found",result["rejection_details"][0]["reasons"])

    def test_non_numeric_skips_consistency(self):
        self.assertFalse(needs_numerical_consistency("Which language models are evaluated?"))
        self.assertTrue(needs_numerical_consistency("What BLEU scores were reported?"))

    def test_draft_only_one_call_for_nonnumeric(self):
        draft=DraftAnswer(status="abstained")
        with patch('app.grounding.generate',return_value=draft) as model:
            answer,report=draft_answer("Which models?",[],self.evidence,self.tasks)
        self.assertIsNone(report)
        self.assertEqual(model.call_count,1)

if __name__=='__main__': unittest.main()
