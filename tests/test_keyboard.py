"""Keyboard error model and demo editor tests: python -m unittest discover tests"""

import unittest

from typeling import Predictor
from typeling.demo import LineEditor
from typeling.keyboard import NEIGHBORS, corrections


class CorrectionsTest(unittest.TestCase):
    def fixes(self, typed):
        return {q for q, _ in corrections(typed)}

    def test_single_errors_are_undone(self):
        self.assertIn("know", self.fixes("kbow"))     # adjacent key
        self.assertIn("tom", self.fixes("tomm"))      # doubled letter
        self.assertIn("think", self.fixes("thnik"))   # swapped letters
        self.assertIn("recei", self.fixes("recie"))
        self.assertIn("don't", self.fixes("dont"))    # left-out apostrophe
        self.assertIn("I", self.fixes("i"))           # missing capital

    def test_no_letter_is_added_at_the_end(self):
        self.assertNotIn("know", self.fixes("kno"))

    def test_extra_keys_must_be_near_their_neighbors(self):
        self.assertIn("ab", self.fixes("abb"))
        self.assertNotIn("ab", self.fixes("apb"))     # p is not next to a or b

    def test_neighbors_are_symmetric(self):
        for c, nb in NEIGHBORS.items():
            for d in nb:
                self.assertIn(c, NEIGHBORS[d])


class FakePredictor:
    def topk(self, text, k=3):
        return [("tomorrow", 0.9), ("tomato", 0.05), ("Tom", 0.01)][:k] if text.endswith("tomm") else []

    def suggest(self, text):
        return ""


class LineEditorTest(unittest.TestCase):
    def type(self, ed, text):
        for ch in text:
            ed.feed(ch)

    def test_ghost_text(self):
        ed = LineEditor(Predictor())
        self.type(ed, "Let me kn")
        ed.feed("TAB")
        self.assertEqual(ed.text, "Let me know")

    def test_candidates_replace_the_typed_letters(self):
        ed = LineEditor(FakePredictor())
        self.type(ed, "See you tomm")
        self.assertEqual(ed.view(), ("", ["tomorrow", "tomato", "Tom"]))
        ed.feed("TAB")
        self.assertEqual(ed.text, "See you tomorrow")

    def test_highlight(self):
        ed = LineEditor(FakePredictor())
        self.type(ed, "See you tomm")
        for key in ("DOWN", "DOWN", "DOWN", "UP"):   # wraps around
            ed.feed(key)
        ed.feed("TAB")
        self.assertEqual(ed.text, "See you Tom")


if __name__ == "__main__":
    unittest.main()
