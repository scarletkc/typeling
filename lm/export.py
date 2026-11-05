# /// script
# requires-python = ">=3.13"
# dependencies = ["torch>=2.7", "transformers>=5", "huggingface_hub", "numpy", "sentencepiece"]
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
Release export (milestone 3 of docs/model-plan.md): writes the files of the Hugging Face repositories into
data/release/<name>/, ready for `hf upload`.

    uv run lm/export.py --model data/lm/runs/cpt-150m                              # Typeling-25M
    uv run lm/export.py --model data/lm/runs/cpt-150m --llama-cpp PATH/TO/llama.cpp   # also Typeling-25M-GGUF
    uv run lm/export.py --model data/lm/runs/cpt-150m --llama-cpp PATH/TO/llama.cpp --f16-gguf   # plus an F16 check file

Typeling-25M: fp16 safetensors, config, tokenizer, the model card lm/cards/Typeling-25M.md as README.md, LICENSE and
NOTICE. Typeling-25M-GGUF: Q8_0 written directly by llama.cpp's convert_hf_to_gguf.py (from a llama.cpp checkout, used
with build b11953) from the exported fp16 files, with the card lm/cards/Typeling-25M-GGUF.md, LICENSE and NOTICE.
Nothing is uploaded.
"""

import argparse
import json
import runpy
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CARDS = Path(__file__).resolve().parent / "cards"
NAME = "Typeling-25M"


def copy_docs(out, card):
    if (CARDS / f"{card}.md").exists():
        shutil.copyfile(CARDS / f"{card}.md", out / "README.md")
    else:
        print(f"warning: no model card lm/cards/{card}.md")
    for f in ("LICENSE", "NOTICE"):
        shutil.copyfile(ROOT / f, out / f)


def convert_gguf(llama_cpp, src, target, outtype="q8_0"):
    """Run llama.cpp's convert_hf_to_gguf.py in this process. Its pre-tokenizer detection hashes the tokenization of
    a test string and knows only published vocabularies; Typeling's tokenizer uses GPT-2's byte-level regex, so the
    unknown hash is mapped to "gpt-2". (The tokenizer also adds a space before every text, which llama.cpp's BPE
    tokenizer does not: GGUF users prepend it themselves.)"""
    sys.path[:0] = [str(llama_cpp), str(llama_cpp / "gguf-py")]
    from conversion.base import TextModel

    detect = TextModel.get_vocab_base_pre

    def get_vocab_base_pre(self, tokenizer):
        try:
            return detect(self, tokenizer)
        except NotImplementedError:
            return "gpt-2"

    TextModel.get_vocab_base_pre = get_vocab_base_pre
    sys.argv = ["convert_hf_to_gguf.py", str(src), "--outtype", outtype, "--outfile", str(target)]
    try:
        runpy.run_path(str(llama_cpp / "convert_hf_to_gguf.py"), run_name="__main__")
    finally:
        TextModel.get_vocab_base_pre = detect


def main():
    ap = argparse.ArgumentParser(description="Export the release files")
    ap.add_argument("--model", required=True, help="trained model directory")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "release")
    ap.add_argument("--llama-cpp", type=Path, help="llama.cpp checkout, for the GGUF file")
    ap.add_argument("--f16-gguf", action="store_true",
                    help="also write data/release/check/Typeling-25M-F16.gguf (not released), to measure Q8_0's loss")
    args = ap.parse_args()

    out = args.out / NAME
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out)
    # Transformers 5 writes "TokenizersBackend", which Transformers 4 cannot load; both know the fast tokenizer class.
    # Transformers 4 also resets the pre-tokenizer's add_prefix_space unless the config sets it.
    path = out / "tokenizer_config.json"
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")) | {
        "tokenizer_class": "PreTrainedTokenizerFast", "add_prefix_space": True}, indent=2) + "\n", encoding="utf-8")
    copy_docs(out, NAME)
    print(f"{out}: " + ", ".join(f"{p.name} ({p.stat().st_size / 1e6:.2f} MB)" for p in sorted(out.iterdir())))

    if args.llama_cpp:
        gguf = args.out / f"{NAME}-GGUF"
        if gguf.exists():
            shutil.rmtree(gguf)
        gguf.mkdir(parents=True)
        convert_gguf(args.llama_cpp.resolve(), out, gguf / f"{NAME}-Q8_0.gguf")
        copy_docs(gguf, f"{NAME}-GGUF")
        print(f"{gguf}: " + ", ".join(f"{p.name} ({p.stat().st_size / 1e6:.2f} MB)" for p in sorted(gguf.iterdir())))
        if args.f16_gguf:
            (args.out / "check").mkdir(exist_ok=True)
            convert_gguf(args.llama_cpp.resolve(), out, args.out / "check" / f"{NAME}-F16.gguf", "f16")


if __name__ == "__main__":
    main()
