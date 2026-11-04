# 实验：手写权重能否达到训练出来的水平

任务：英文单词补全（可利用同一段落的前文）。在 WikiText-2 上模拟打字，比较不同来源的权重。

## 参赛档位

| 档位 | 权重从哪来 | 可以用的数据 |
|---|---|---|
| baseline | 现有模型（B0：前缀匹配 + wordfreq 词频） | — |
| H0 纯规则 | 按规则手写，不看任何语料统计 | `vocab.txt` |
| H1 公式计算 | 从语料统计直接算出（计数、PMI、SVD…），不做梯度下降 | `vocab.txt`、`train.txt` |
| H2 LLM 写权重 | 由 LLM 按 weight-space programming 的方式直接写出，不用梯度 | `vocab.txt`、开发集评测结果（含出错样例），最多 50 次评测 |
| T 训练 | 梯度下降训练的小神经网络 | `vocab.txt`、`train.txt`，开发集可用于早停和调参 |

注意：H2 的"写手"本身是训练出来的模型，所以 H2 达标说明的是"知识可以不经梯度下降直接写进小网络"，不是"完全不需要训练"。

## 公平规则

- 除 baseline 外，所有模型使用同一份词表 `data/wikitext2/vocab.txt`（训练集出现 ≥2 次的纯字母词，36,300 个，只有词、没有次数）。
- 体积上限 5 MB（加载模型所需文件的总大小，允许压缩）。
- 测试集封存：调试只用开发集；每个模型最终只用 `--final` 在测试集上跑一次。
- 写 H0 / H2 时不读 `train.txt` 原文，也不看测试集。

## 指标

在 `evaluate.py` 开头有完整定义。简述：

- **KSR**：按键节省率。第一个候选就是目标词时按 Tab 接受（算 1 键）。
- **acc@i**：已打 i 个字母时第一个候选正确的比例；acc@0 即"预测下一个词"。
- **top3@0**：下一个词在前 3 个候选里的比例。
- **OOV**：目标词不在模型词表里的比例。
- 体积、加载时间、每键预测耗时。

## 运行

```bash
uv run experiment/prepare_data.py          # 下载并整理数据（一次）
python experiment/evaluate.py B0           # 开发集评测，结果追加到 results.jsonl
python experiment/evaluate.py B0 --final   # 测试集（只在最终比较时）
```

新模型：在 `models.py` 里继承 `Model`，实现 `begin_word / predict / knows / size_bytes`，登记到 `REGISTRY`。

## 结果（开发集）

| 模型 | 档位 | KSR | acc@0 | acc@1 | acc@2 | acc@3 | top3@0 | OOV | 体积 | 每键 |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | 17.13% | 8.53% | 20.99% | 27.22% | 33.63% | 13.99% | 3.69% | 0.88 MB | 22.5 µs |

数据：WikiText-2（Salesforce/wikitext，wikitext-2-raw-v1，CC BY-SA 3.0）。
