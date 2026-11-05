# Typeling for Windows (prototype)

A Windows input method that shows English word candidates next to the cursor, in the applications you type in. It
exists to test whether this works across applications and helps in real typing; it is not a finished product.

- `tip/`: the text service, a Rust DLL built on the Text Services Framework (TSF). It is loaded into each application,
  reads the text before the cursor and draws the candidate bar. Letters reach the application untouched.
- `engine.py`: the prediction engine, a separate process running the bundled n-gram model. The DLL talks to it over a
  named pipe from a worker thread, so a slow or stopped engine never holds up typing: the bar just does not appear.

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

From an elevated PowerShell, because this registers the DLL for the whole machine:

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

To remove it, from an elevated PowerShell:

```bash
pwsh -File app/windows/uninstall.ps1
```

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

- Candidates come from the n-gram model, so there is no typo repair; Typeling-25M is not connected yet.
- The engine is started by hand, and nothing is learned from what you type.
- Only 64-bit desktop applications: 32-bit applications cannot load the DLL, and sandboxed (AppContainer) Store apps
  cannot reach the DLL or the pipe. They keep working, without candidates.
- Text can be read and changed only where the application supports TSF. Terminals are not expected to work.
- The DLL is not code-signed and there is no installer, tray icon or settings.
