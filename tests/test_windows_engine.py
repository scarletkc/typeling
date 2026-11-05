"""Tests of the Windows input method's prediction engine (app/windows/engine.py)."""

import importlib.util
import os
import struct
import threading
import time
import unittest
import uuid
from pathlib import Path
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

    def test_word_already_typed_is_not_a_candidate(self):
        n, words = engine.respond(self.p, "I like the")
        self.assertEqual(n, 3)
        self.assertNotIn("the", words)
        self.assertTrue(words)

    def test_curly_apostrophe_is_treated_as_straight(self):
        n, words = engine.respond(self.p, "I don\u2019")
        self.assertEqual(n, 4)
        self.assertIn("don't", words)

    def test_reply_format(self):
        self.assertEqual(engine.encode_reply(2, ["know", "knew"]), b"2\nknow\nknew")
        self.assertEqual(engine.encode_reply(0, []), b"0")


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
