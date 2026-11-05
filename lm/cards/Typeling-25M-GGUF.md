---
license: apache-2.0
language:
- en
base_model: ScarletKc/Typeling-25M
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- typing
- autocomplete
- word-prediction
- next-word-prediction
- keyboard
---

# Typeling-25M-GGUF

[Typeling-25M](https://huggingface.co/ScarletKc/Typeling-25M), a 25M-parameter English language model for typing
prediction, as a GGUF file for [llama.cpp](https://github.com/ggml-org/llama.cpp).

| File | Size | Quality against 16-bit weights |
|---|---|---|
| `Typeling-25M-Q8_0.gguf` | 27.6 MB | perplexity +0.08%, mean KL divergence 0.0009, same top token 97.9% of the time |

Measured with `llama-perplexity --kl-divergence` on 380k characters of held-out chat and web text (190 chunks of 512
tokens), against an F16 conversion of the same weights. There is no 4-bit file: at this size Q4 would save about 10
MB and small models lose more to 4-bit quantization. You can make one from the
[16-bit weights](https://huggingface.co/ScarletKc/Typeling-25M) with llama.cpp's `convert_hf_to_gguf.py` and
`llama-quantize`.

## What the model is for

Typeling-25M predicts the rest of the word being typed and the next word, for ghost text and candidate bars in chat
and everyday writing. It is a small base model, not a chat or instruction model.

The model predicts tokens, not words. The word-level suggestions, typo repair and benchmark results on the
[main model card](https://huggingface.co/ScarletKc/Typeling-25M) come from the Typeling decoder
([`typeling/lm.py`](https://github.com/scarletkc/typeling/blob/master/typeling/lm.py), `pip install "typeling[lm]"`),
which runs the 16-bit weights with PyTorch. With llama.cpp you get raw next-token predictions; the decoder is the
reference for building word suggestions on top of them.

## Input format

- Texts in training were separated by `</s>` (token 2). Start the prompt with `</s>`.
- The original tokenizer puts a space before every text, so the model expects `" Hello"`, not `"Hello"`, as the first
  word. llama.cpp does not add this space: write it yourself.
- llama.cpp adds no BOS token for this model; none is needed.
- Predictions are best when the text ends right after a word. After a trailing space the model has seen fewer
  examples (1% of its training words).

```sh
llama-completion -m Typeling-25M-Q8_0.gguf -p "</s> Let me know if" -n 6 --temp 0 -no-cnv
#  Let me know if you need any help.
```

`"</s> Thank you so"` tokenizes to `[2, " Thank", " you", " so"]`, the same as the Transformers tokenizer.

## Conversion

Converted with llama.cpp b11953 (`convert_hf_to_gguf.py --outtype q8_0`) by
[`lm/export.py`](https://github.com/scarletkc/typeling/blob/master/lm/export.py). The converter does not recognize this
tokenizer's pre-tokenizer, which uses GPT-2's byte-level regex, so the export maps it to `gpt-2`.

## License

Apache License 2.0, like the base model
[SupraLabs/Supra2-Medium-Base](https://huggingface.co/SupraLabs/Supra2-Medium-Base). The training data and its
attributions are listed on the [main model card](https://huggingface.co/ScarletKc/Typeling-25M) and in `NOTICE`.
