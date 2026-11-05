"""Tests of the Windows input method's prediction engine (app/windows/engine.py)."""

import importlib.util
import os
import struct
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from typeling import Predictor

_PATH = Path(__file__).resolve().parents[1] / "app" / "windows" / "engine.py"
_SPEC = importlib.util.spec_from_file_location("typeling_windows_engine", _PATH)
engine = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(engine)


class RespondTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Predictor()

    def test_completion_reports_partial_length(self):
        n, words = engine.respond(self.p, "Let me kn")
        self.assertEqual(n, 2)
        self.assertIn("know", words)
        self.assertLessEqual(len(words), engine.CANDIDATES)

    def test_next_word_has_empty_partial(self):
        n, words = engine.respond(self.p, "Thank you so ")
        self.assertEqual(n, 0)
        self.assertIn("much", words)

    def test_no_candidates_right_after_punctuation(self):
        self.assertEqual(engine.respond(self.p, "Hello,"), (0, []))

    def test_word_already_typed_stays_a_candidate(self):
        n, words = engine.respond(self.p, "I like the")
        self.assertEqual(n, 3)
        self.assertEqual(words[0], "the")

    def test_curly_apostrophe_is_treated_as_straight(self):
        n, words = engine.respond(self.p, "I don\u2019")
        self.assertEqual(n, 4)
        self.assertIn("don't", words)

    def test_language_model_goes_through_the_bar_rules(self):
        class Lexicon:
            words = ["really", "realm"]
            p1 = [0.002, 0.0001]

        ranked = [("really", 0.9), ("reallyoo", 0.05), ("Realm", 0.03), ("reallyon", 0.02)]
        lm = SimpleNamespace(ngram=SimpleNamespace(lm=Lexicon()), topk=lambda text, k: ranked[:k])
        self.assertEqual(engine.respond(lm, "It was realy"), (5, ["really", "Realm"]))

    def test_reply_format(self):
        self.assertEqual(engine.encode_reply(2, ["know", "knew"]), b"2\nknow\nknew")
        self.assertEqual(engine.encode_reply(0, []), b"0")


class ChooseCandidatesTest(unittest.TestCase):
    RANKS = {"very": 100, "version": 1500, "verified": 3000, "verifiable": 20000, "the": 0, "hi": 4074, "his": 40,
             "play": 200, "playing": 500, "playlists": 15000, "am": 2138, "made": 300, "amusing": 9000, "think": 150,
             "thinking": 500}

    def choose(self, ranked, partial="ve"):
        return engine.choose_candidates(ranked, partial, self.RANKS)

    def test_other_places_prefer_common_words(self):
        ranked = [("very", 0.9), ("verifiable", 0.05), ("verified", 0.04), ("version", 0.03)]
        self.assertEqual(self.choose(ranked), ["very", "version", "verified"])

    def test_first_place_stays_the_models_best_word(self):
        self.assertEqual(self.choose([("verifiable", 0.5), ("very", 0.3)]), ["verifiable", "very"])

    def test_typed_word_keeps_its_place(self):
        self.assertEqual(self.choose([("hi", 0.92), ("his", 0.02)], partial="hi"), ["hi", "his"])
        ranked = [("play", 0.99), ("playing", 0.003), ("playlists", 0.001)]
        self.assertEqual(self.choose(ranked, partial="play"), ["play", "playing", "playlists"])

    def test_unlikely_words_keep_the_bar_full_if_related(self):
        self.assertEqual(self.choose([("very", 0.99), ("verified", 0.001)]), ["very", "verified"])
        ranked = [("think", 0.9), ("thinking", 0.001), ("the", 0.001)]
        self.assertEqual(self.choose(ranked, partial="thnik"), ["think", "thinking"])

    def test_unrelated_corrections_are_dropped(self):
        ranked = [("am", 0.99), ("made", 0.0001), ("amusing", 0.0001)]
        self.assertEqual(self.choose(ranked, partial="am"), ["am", "amusing"])

    def test_next_word_places_have_no_minimum(self):
        self.assertEqual(self.choose([("the", 0.5), ("very", 0.001)], partial=""), ["the", "very"])

    def test_non_words_are_dropped(self):
        self.assertEqual(self.choose([("veryoo", 0.9), ("very", 0.05)]), ["very"])
        self.assertEqual(self.choose([("veryoo", 0.9)]), [])

    def test_case_is_ignored_when_looking_up_a_word(self):
        self.assertEqual(self.choose([("Very", 0.9), ("Version", 0.05)]), ["Very", "Version"])


MODEL = os.environ.get("TYPELING_LM")
LM_READY = bool(MODEL) and all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


@unittest.skipUnless(LM_READY, 'set TYPELING_LM to a model and install "typeling[lm]"')
class LMRespondTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from typeling.lm import LMPredictor
        cls.p = LMPredictor(MODEL, device="cpu")

    def test_typo_repair_replaces_the_typed_letters(self):
        n, words = engine.respond(self.p, "See you tomm")
        self.assertEqual(n, 4)
        self.assertIn("tomorrow", words)
        n, words = engine.respond(self.p, "What do you thnik")
        self.assertEqual(n, 5)
        self.assertIn("think", words)

    def test_a_complete_word_is_not_pushed_out_by_unlikely_ones(self):
        self.assertEqual(engine.respond(self.p, "what do you like to play")[1][0], "play")
        self.assertEqual(engine.respond(self.p, "hi")[1][0].lower(), "hi")
        self.assertEqual(engine.respond(self.p, "I am")[1][0], "am")
        self.assertNotIn("made", engine.respond(self.p, "I am")[1])

    def test_the_bar_is_full_when_related_words_exist(self):
        for text in ("Can you he", "It is ve", "We ne", "What do you thnik"):
            self.assertEqual(len(engine.respond(self.p, text)[1]), engine.CANDIDATES, text)

    def test_non_words_never_reach_the_bar(self):
        for text, junk in [("It was realy", "reallyoo"), ("We need to seperate", "seperateinate"),
                           ("I would like to discus", "discuse"), ("What do you thnik", "think's")]:
            _, words = engine.respond(self.p, text)
            self.assertNotIn(junk, words)
            self.assertTrue(all(w.lower() in self.p.ngram.lm.id for w in words), words)

    def test_next_word_and_limit(self):
        n, words = engine.respond(self.p, "Thank you so ")
        self.assertEqual(n, 0)
        self.assertIn("much", words)
        self.assertLessEqual(len(words), engine.CANDIDATES)


@unittest.skipUnless(os.name == "nt", "named pipes of this kind exist on Windows only")
class PipeTest(unittest.TestCase):
    @staticmethod
    def _connect(name):
        # The server makes a new pipe instance right after each connection; in between, clients must retry.
        for _ in range(100):
            try:
                return open(name, "r+b", buffering=0)
            except OSError:
                time.sleep(0.01)
        return open(name, "r+b", buffering=0)

    @staticmethod
    def _exchange(f, text):
        body = text.encode("utf-8")
        f.write(struct.pack("<I", len(body)) + body)
        size = struct.unpack("<I", f.read(4))[0]
        return f.read(size).decode("utf-8").split("\n")

    def test_refuses_to_run_elevated(self):
        with mock.patch.object(engine, "_is_elevated", return_value=True), self.assertRaises(SystemExit) as caught:
            engine.main()
        self.assertIn("normal terminal", str(caught.exception))

    def test_clients_are_answered_independently(self):
        name = rf"\\.\pipe\LOCAL\typeling-test-{uuid.uuid4().hex}"
        ready = threading.Event()
        threading.Thread(target=engine.serve, args=(Predictor(), name, ready), daemon=True).start()
        self.assertTrue(ready.wait(10))
        with self._connect(name) as a, self._connect(name) as b:
            first = self._exchange(a, "Let me kn")
            second = self._exchange(b, "Thank you so ")
            again = self._exchange(a, "Let me kn")
        self.assertEqual(first[0], "2")
        self.assertIn("know", first[1:])
        self.assertEqual(second[0], "0")
        self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
