"""Kernel unit tests: python -m unittest discover tests"""

import tempfile
import unittest
from pathlib import Path

from typeling import Predictor
from typeling.model import read_model, write_model
from typeling.text import normalize, split_input, tokenize


class TextTest(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize(normalize("I\u2019m here, don't go!! 42 (ok)")), [
            ("i'm", "I'm"), ("here", "here"), (",", None), ("don't", "don't"), ("go", "go"),
            ("<s>", None), ("<num>", None), ("<p>", None), ("ok", "ok"), ("<p>", None)])

    def test_split_input(self):
        self.assertEqual(split_input("hello wor"), ("hello ", "wor"))
        self.assertEqual(split_input("hello "), ("hello ", ""))
        self.assertEqual(split_input("I don'"), ("I ", "don'"))
        self.assertEqual(split_input("(beaut"), ("(", "beaut"))
        self.assertIsNone(split_input("hello,"))
        self.assertIsNone(split_input("x" * 40))


class FormatTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.bin"
            write_model(path, {"a": 1}, {"TEXT": ("s", "hé".encode()), "NUMS": ("H", bytes([1, 0, 2, 1]))})
            meta, s = read_model(path)
        self.assertEqual(meta, {"a": 1})
        self.assertEqual(s["TEXT"], "hé")
        self.assertEqual(list(s["NUMS"]), [1, 258])


class PredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Predictor()

    def test_completion(self):
        self.assertEqual(self.p.suggest("Let me kn"), "ow")
        self.assertEqual(self.p.suggest("I don"), "'t")
        self.assertEqual(self.p.suggest("Thank you so "), "much")

    def test_case(self):
        self.assertEqual(self.p.suggest("WHAT ARE YOU DOI"), "NG")
        self.assertTrue(self.p.topk("It works. ", 1)[0][0][0].isupper())  # capitalized at sentence start
        self.assertEqual(self.p.topk("Yesterday i", 1)[0][0][:1], "i")    # keeps the letters the user typed
        for word, _ in self.p.topk("the ng", 8):                         # no mixed case like "ngOs"
            self.assertTrue(word.islower(), word)

    def test_no_suggestion_after_punctuation(self):
        self.assertEqual(self.p.suggest("Hello,"), "")
        self.assertEqual(self.p.topk("Hello,"), [])

    def test_blocked_words_are_not_suggested(self):
        lm = self.p.lm
        for i in sorted(lm.blocked)[:20]:
            w = lm.words[i]
            self.assertNotIn(w, [c.lower() for c, _ in self.p.topk(w[:-1], 8)])

    def test_misspellings_are_not_suggested(self):
        for typed, bad in (("I will reci", "recieve"), ("definat", "definately"), ("Let me knw", "knwo")):
            self.assertNotIn(bad, [c.lower() for c, _ in self.p.topk(typed, 8)])
        self.assertEqual(self.p.suggest("I will rece"), "ive")
        self.assertIn("lol", [c for c, _ in self.p.topk("lo", 8)])  # slang is not treated as a misspelling

    def test_learning(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "user.json"
            p = Predictor(user=path)
            for _ in range(3):
                p.learn("Our project is called Zorblaxa, and Zorblaxa ships next week.")
            self.assertEqual(p.topk("I love Zorb", 1)[0][0], "Zorblaxa")
            p.save()
            q = Predictor(user=path)
            self.assertEqual(q.topk("I love Zorb", 1)[0][0], "Zorblaxa")


if __name__ == "__main__":
    unittest.main()
