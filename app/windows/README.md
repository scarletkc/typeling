# Typeling for Windows (prototype)

A Windows input method that shows English word candidates next to the cursor, in the applications you type in. It
exists to test whether this works across applications and helps in real typing; it is not a finished product.

- `tip/`: the text service, a Rust DLL built on the Text Services Framework (TSF). It is loaded into each application,
  reads the text before the cursor and draws the candidate bar. Letters reach the application untouched.
- `engine.py`: the prediction engine, a separate process running the bundled n-gram model or Typeling-25M. The DLL
  talks to it over a named pipe from a worker thread, so a slow or stopped engine never holds up typing: the bar just
  does not appear.

## Keys

| Key | Action |
|---|---|
| Tab | accept the first candidate |
| Alt+1, Alt+2, Alt+3 | accept that candidate |
| Esc | hide the bar until the next typed key |

Clicking a candidate also accepts it. These keys are taken only while the bar is visible. The bar does not appear on
a blank line, so Tab still indents and moves between empty fields, and it goes away when the cursor moves.

## Install

Building needs Rust with the MSVC toolchain; the engine needs Python 3.13.

```bash
cargo build --release --manifest-path app/windows/tip/Cargo.toml
```

From an elevated PowerShell, because this registers the DLL for the whole machine. It opens in `C:\Windows\System32`,
so change to the repository folder first:

```bash
pwsh -File app/windows/install.ps1
```

Keep the engine running in a normal terminal. It refuses to start in an elevated one, because applications could not
connect to it, and without an engine no candidates appear:

```bash
python app/windows/engine.py
```

Then press Win+Space and choose Typeling, which is listed under English (United States). If it is not listed, add
that language in Settings > Time & language > Language & region. Applications that were already open need a restart.

To remove it, from an elevated PowerShell in the repository folder:

```bash
pwsh -File app/windows/uninstall.ps1
```

## Typo repair

Typeling-25M also repairs typing slips: `See you tomm` offers *tomorrow*, and accepting it replaces the typed letters.
It needs PyTorch and Transformers, which uv can provide without installing them. Stop the running engine first, because
only one can run at a time:

```bash
uv run --no-project --python 3.13 --with "transformers>=5" --with "torch>=2.7" --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match python app/windows/engine.py --lm
```

`--lm` downloads the model from Hugging Face once and asks for its latest revision at each start; `--lm DIR` uses a local
copy and makes no such request. Typed text is never sent. It runs on the CPU with 4 threads by default (`--device`,
`--threads`).

The bar mixes the two models. The first place is the language model's best word, fused with the n-gram model as usual;
that can be the word already typed, in which case accepting it changes nothing. The other places prefer common words,
because beyond its best guess the language model tends toward specific or rare words. A word under 2% likely must extend
the typed letters or the first word, which keeps the bar full but drops unrelated corrections. Only words in the n-gram
model's vocabulary are offered, because the language model also reaches non-words through its subword pieces.

Measured once on the development laptop through the pipe, typing 576 keystrokes: 23 ms per keystroke on average, 50 ms at
p99, and about 490 MB of memory. A release would run the model with llama.cpp instead of PyTorch.

## Privacy

No field gets predictions if it is marked as a password or PIN or is read-only. Windows of elevated applications get
none either, so their text never reaches the engine, which runs without elevation. The engine holds the text before
the cursor (at most 1,000 characters) for the length of one request: it does not log, learn or store anything, and the
pipe accepts local clients only.

## Tests

```bash
cargo test --manifest-path app/windows/tip/Cargo.toml
```

```bash
python -m unittest tests.test_windows_engine
```

`tests/tsf_e2e.rs` runs a typing session against the real engine and a real TSF thread manager, with an in-memory text
store as the application, and installs nothing. TSF accepts key sinks only from installed input methods, so that test
delivers keys to the service itself. Key routing, and how each application behaves, can be checked only after
installing.

## Known limits

- Typo repair needs `--lm`, and it covers typing slips (adjacent keys; missing, extra and swapped letters), not
  misspellings by sound: `definately` and `pleese` get no useful candidates, and a misspelling that made it into the
  vocabulary can still be offered (`seperate` offers *seperately*).
- The engine is started by hand, and nothing is learned from what you type.
- Only 64-bit desktop applications: 32-bit applications cannot load the DLL, and sandboxed (AppContainer) Store apps
  cannot reach the DLL or the pipe. They keep working, without candidates.
- Text can be read and changed only where the application supports TSF. Terminals are not expected to work.
- The DLL is not code-signed and there is no installer, tray icon or settings.

## Roadmap

In this order; none of it has started.

1. A standalone engine that needs neither Python nor PyTorch and starts by itself. It would run the Q8_0 GGUF file with
   llama.cpp and port the decoder in `typeling/lm.py`, including typo repair. Keystroke savings, typo repair, latency
   and memory must hold up before it replaces the current engine.
2. Repair of misspellings by sound (`definately`, `pleese`), which matters to non-native writers, the first target
   users. It comes after the engine so that it is written only once.
3. An installer and code signing, with a tray icon and settings.
