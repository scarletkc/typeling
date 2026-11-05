# /// script
# requires-python = ">=3.13"
# dependencies = ["torch>=2.7", "transformers>=5", "huggingface_hub", "numpy"]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///
"""
Language model decoder tests (typeling.lm). Skipped unless PyTorch and Transformers are installed and TYPELING_LM
names a model (Hugging Face repo id or local directory); the expectations hold for Typeling-25M.

    TYPELING_LM=data/lm/runs/cpt-150m uv run tests/test_lm.py
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = os.environ.get("TYPELING_LM")
READY = bool(MODEL) and all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


@unittest.skipUnless(READY, 'set TYPELING_LM to a model and install "typeling[lm]"')
class LMPredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from typeling.lm import LMPredictor
        cls.p = LMPredictor(MODEL, device="cpu")

    def words(self, text, k=3):
        return [w.lower() for w, _ in self.p.topk(text, k)]

    def test_completion_and_next_word(self):
        self.assertEqual(self.p.suggest("Let me kn"), "ow")
        self.assertEqual(self.p.suggest("Thank you so "), "much")
        self.assertEqual(self.words("I don't kn")[0], "know")

    def test_candidates(self):
        cands = self.p.topk("I'm going to ", 5)
        self.assertEqual(len(cands), 5)
        self.assertEqual(len({w.lower() for w, _ in cands}), 5)
        probs = [c for _, c in cands]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertLessEqual(sum(probs), 1.0 + 1e-6)
        self.assertEqual(self.p.topk("I'm going to ", 3), cands[:3])   # served from the cached search

    def test_typo_repair(self):
        for typed, word in (("See you tomm", "tomorrow"), ("Let me kbow", "know"), ("What do you thnik", "think"),
                            ("Thank you so mich", "much")):
            self.assertIn(word, self.words(typed), typed)
        self.assertEqual(self.p.suggest("See you tomm"), "")   # ghost text only extends the typed letters

    def test_misspellings_are_not_suggested(self):
        self.assertNotIn("recieve", self.words("I will recie", 8))

    def test_blocked_words_are_not_suggested(self):
        lm = self.p.ngram.lm
        for i in sorted(lm.blocked)[:10]:
            w = lm.words[i]
            self.assertNotIn(w, self.words(w[:-1], 8))

    def test_no_suggestion_after_punctuation(self):
        self.assertEqual(self.p.topk("Hello,"), [])
        self.assertEqual(self.p.suggest("Hello,"), "")

    def test_learning(self):
        from typeling.lm import LMPredictor
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "user.json"
            p = LMPredictor(MODEL, path, device="cpu")
            self.assertNotEqual(self.words("I love Zorb")[:1], ["zorblaxa"])
            for _ in range(3):
                p.learn("Our project is called Zorblaxa, and Zorblaxa ships next week.")
            self.assertEqual(p.topk("I love Zorb", 1)[0][0], "Zorblaxa")
            self.assertEqual(self.p.topk("Let me kn", 1)[0][0], p.topk("Let me kn", 1)[0][0])   # common words
            p.save()
            q = LMPredictor(MODEL, path, device="cpu")
            self.assertEqual(q.topk("I love Zorb", 1)[0][0], "Zorblaxa")


if __name__ == "__main__":
    unittest.main()
