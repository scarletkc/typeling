"""
Prediction engine for the Windows input method prototype: answers the text service DLL over a named pipe.

    python app/windows/engine.py                # the n-gram model; keep it running while typing, Ctrl+C stops it
    python app/windows/engine.py --lm           # Typeling-25M with typo repair (needs typeling[lm]); --lm DIR for a
                                                # local model, which also means no request to Hugging Face at start

Each message is a little-endian u32 length followed by that many bytes of UTF-8.
    Request:  the text before the cursor
    Reply:    lines: the length of the partly typed word in UTF-16 units, then up to 3 candidate words

Typed text stays in memory for the length of one request: it is not logged, learned or written to disk.
Standard library only, like the rest of the runtime, except for --lm.
"""

import argparse
import ctypes
import functools
import os
import struct
import sys
import threading
import time
from pathlib import Path

try:
    from typeling import Predictor
    from typeling.text import normalize, split_input
except ImportError:  # running from a checkout without the package installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from typeling import Predictor
    from typeling.text import normalize, split_input

PIPE_NAME = r"\\.\pipe\LOCAL\typeling-engine"  # LOCAL: visible to this logon session only
CANDIDATES = 3
SEARCHED = 12  # words asked of the predictor, to still have enough after dropping some
# The language model's second and third places (tuned on the dev split): see choose_candidates().
MIN_PROBABILITY = 0.02  # less likely words must be related to the typed letters or the first word
RARITY_POWER = 0.8      # how strongly they prefer common words
RANK_SMOOTHING = 50     # the most common words count as about equally common
BUFFER_SIZE = 64 * 1024
MAX_REQUEST = 64 * 1024  # the DLL sends at most 1,000 characters


@functools.cache
def word_ranks(lexicon):
    """Lowercase word -> its frequency rank in the n-gram model's vocabulary, the most common word first."""
    order = sorted(range(len(lexicon.words)), key=lambda i: -lexicon.p1[i])
    return {lexicon.words[i]: rank for rank, i in enumerate(order)}


def choose_candidates(ranked, partial, ranks, k=CANDIDATES):
    """The language model's (word, probability) list, most probable first -> the words for the bar.
    The first place is the model's best word, which is right far more often than the n-gram model's, and that may be
    what is already typed: removing it would promote a word the model finds very unlikely (play -> playlists).
    Beyond the first place the model has little to go on and tends toward specific or rare words, so the other places
    prefer common words. Unlikely words must extend the typed letters or the first word, which keeps the bar full but
    drops unrelated corrections (am -> made). The model builds words from subword pieces and also reaches non-words
    (seperateinate, reallyoo); people use the bar to check their spelling, so it offers only words in the n-gram
    model's vocabulary."""
    known = [(w, p) for w, p in ranked if w.lower() in ranks]
    if not known:
        return []
    stems = (partial.lower(), known[0][0].lower())
    rest = [e for e in known[1:] if e[1] >= MIN_PROBABILITY or e[0].lower().startswith(stems)]
    rest.sort(key=lambda e: e[1] / (ranks[e[0].lower()] + RANK_SMOOTHING) ** RARITY_POWER, reverse=True)
    return [known[0][0], *(w for w, _ in rest[:k - 1])]


def respond(predictor, text, k=CANDIDATES):
    """Text before the cursor -> (length of the partial word in UTF-16 units, candidate words)."""
    text = normalize(text)
    sp = split_input(text)
    if sp is None:
        return 0, []
    partial = sp[1]
    ranked = predictor.topk(text, SEARCHED)
    ngram = getattr(predictor, "ngram", None)
    if ngram is None:
        words = [w for w, _ in ranked][:k]
    else:
        words = choose_candidates(ranked, partial, word_ranks(ngram.lm), k)
    return len(partial.encode("utf-16-le")) // 2, words


def encode_reply(partial_len, words):
    return "\n".join([str(partial_len), *words]).encode("utf-8")


if os.name == "nt":
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateNamedPipeW.restype = wintypes.HANDLE
    _k32.CreateNamedPipeW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID]
    _k32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
    _k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, wintypes.LPDWORD, wintypes.LPVOID]
    _k32.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD, wintypes.LPDWORD, wintypes.LPVOID]
    _k32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]

    _INVALID_HANDLE = wintypes.HANDLE(-1).value
    _PIPE_ACCESS_DUPLEX = 0x00000003
    _FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
    _PIPE_REJECT_REMOTE_CLIENTS = 0x00000008  # with PIPE_TYPE_BYTE, PIPE_READMODE_BYTE and PIPE_WAIT, all 0
    _PIPE_UNLIMITED_INSTANCES = 255
    _ERROR_PIPE_CONNECTED = 535


    def _is_elevated():
        return bool(ctypes.windll.shell32.IsUserAnAdmin())


    def _create_pipe(name, first):
        flags = _PIPE_ACCESS_DUPLEX | (_FILE_FLAG_FIRST_PIPE_INSTANCE if first else 0)
        handle = _k32.CreateNamedPipeW(name, flags, _PIPE_REJECT_REMOTE_CLIENTS, _PIPE_UNLIMITED_INSTANCES,
                                       BUFFER_SIZE, BUFFER_SIZE, 0, None)
        if handle == _INVALID_HANDLE:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle


    def _read_exact(handle, n):
        """-> n bytes, or None when the client has gone away."""
        out = bytearray()
        while len(out) < n:
            buf = ctypes.create_string_buffer(n - len(out))
            got = wintypes.DWORD()
            if not _k32.ReadFile(handle, buf, len(buf), ctypes.byref(got), None) or not got.value:
                return None
            out += buf.raw[:got.value]
        return bytes(out)


    def _write_all(handle, data):
        sent = wintypes.DWORD()
        return _k32.WriteFile(handle, data, len(data), ctypes.byref(sent), None) and sent.value == len(data)


    def _handle_client(handle, predictor, lock):
        try:
            while True:
                head = _read_exact(handle, 4)
                if head is None:
                    return
                size = struct.unpack("<I", head)[0]
                if size > MAX_REQUEST:
                    return
                body = _read_exact(handle, size) if size else b""
                if body is None:
                    return
                text = body.decode("utf-8", "replace")
                with lock:  # Predictor caches its last context and is not thread-safe
                    reply = encode_reply(*respond(predictor, text))
                if not _write_all(handle, struct.pack("<I", len(reply)) + reply):
                    return
        finally:
            _k32.DisconnectNamedPipe(handle)
            _k32.CloseHandle(handle)


    def serve(predictor, name=PIPE_NAME, ready=None):
        """Accept clients forever, one thread per client (each UI thread of each application is one client)."""
        lock = threading.Lock()
        handle = _create_pipe(name, first=True)
        if ready is not None:
            ready.set()
        while True:
            if not _k32.ConnectNamedPipe(handle, None) and ctypes.get_last_error() != _ERROR_PIPE_CONNECTED:
                _k32.CloseHandle(handle)
            else:
                threading.Thread(target=_handle_client, args=(handle, predictor, lock), daemon=True).start()
            handle = _create_pipe(name, first=False)


def load_predictor(args):
    """-> (predictor, description) for the command-line options."""
    if args.lm is None:
        return Predictor(), "n-gram model"
    from typeling.lm import LMPredictor  # needs typeling[lm]
    import torch
    torch.set_num_threads(args.threads)
    predictor = LMPredictor(args.lm, device=args.device)
    word_ranks(predictor.ngram.lm)  # now rather than on the first keystroke
    return predictor, f"{args.lm} on {args.device}, {args.threads} threads"


def main():
    if os.name != "nt":
        sys.exit("The input method prototype runs on Windows only.")
    if _is_elevated():
        # A pipe made by an elevated process refuses the normal applications that need to reach it.
        sys.exit("Start the engine from a normal terminal, not an elevated one: applications cannot connect to an "
                 "elevated engine.")
    ap = argparse.ArgumentParser(description="Prediction engine for the Windows input method prototype")
    ap.add_argument("--lm", metavar="MODEL", nargs="?", const="ScarletKc/Typeling-25M",
                    help="use the language model: Typeling-25M from Hugging Face, or another repo id or directory")
    ap.add_argument("--device", default="cpu", help="language model device: cpu (default) or cuda")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the language model (default 4)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    predictor, description = load_predictor(args)
    ready = threading.Event()
    failed = []

    def run():
        try:
            serve(predictor, ready=ready)
        except OSError as e:
            failed.append(e)
            ready.set()

    # A daemon thread: Ctrl+C cannot interrupt the blocking ConnectNamedPipe call on the main thread.
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait()
    if failed:
        sys.exit(f"Could not create {PIPE_NAME}; is another engine already running? ({failed[0]})")
    print(f"Typeling engine ready in {(time.perf_counter() - t0) * 1000:.0f} ms ({description}). "
          "Press Ctrl+C to stop.", flush=True)
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
