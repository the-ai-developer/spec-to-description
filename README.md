# spec-to-description

Structured product specs in, publishable copy out.

A spec arrives as fields: category, material, dimensions, features. A fine-tuned
T5 reads the flattened form of those fields and writes the description an
editor approves before anything reaches a customer.

The production model is HuggingFace T5. The interesting part of this repository
is that the maths is also written out by hand, in about 250 lines you can read
in one sitting, so you can check the architecture against the thing it claims to
implement.

```
spec-to-description/
├── spec2desc/
│   ├── linearise.py     spec fields -> one input string (the wire format)
│   ├── attention.py     multi-head attention, encoder/decoder blocks, causal mask
│   └── beam_search.py   beam search in plain Python, no ML libraries
├── train/
│   └── train_t5_description.py   DDP fine-tune, fp16 AMP, warmup+cosine
├── eval/
│   └── evaluate_generation.py    BLEU, ROUGE, spec fidelity
├── data/
│   └── make_synthetic_dataset.py deterministic corpus generator
├── notebooks/
│   └── 02_seq2seq_attention_from_scratch.ipynb
└── tests/                        43 tests
```

## Quickstart

```bash
make install          # pip install -e ".[dev]"
make test             # 43 tests, ~14s on CPU
make dataset          # 3200 spec/description pairs, deterministic
make train            # fine-tune t5-small, one GPU
make eval             # BLEU / ROUGE / spec fidelity
```

`make install` needs a virtualenv on distributions that mark the system Python
externally managed (PEP 668), which is most of them now:

```bash
python3 -m venv .venv && . .venv/bin/activate
```

`make test` runs without installing anything, because the tests put the
repository on `sys.path` themselves. That is deliberate: the suite should work
on a fresh checkout with one command and no environment setup.

Two GPUs:

```bash
torchrun --nproc_per_node=2 train/train_t5_description.py \
  --train data/generated/train.jsonl --val data/generated/val.jsonl \
  --out checkpoints/t5-description
```

No GPU? `--no-fp16` and a smaller `--model` will run on CPU, slowly.

## Why the hand-written copies exist

T5 is a black box in the ways that matter when something goes wrong. If the
model emits a plausible sentence that contradicts the spec, you want to be able
to point at the attention mask and say whether the decoder was allowed to look
at that field. That is only possible if the reference implementation is in the
repository next to the trained weights.

So `spec2desc/attention.py` is a readable encoder-decoder, and
`spec2desc/beam_search.py` is beam search you can step through with a print
statement. CI checks that the notebook matches its builder, so the two cannot
quietly disagree.

## A bug worth reading about

`beam_search.py` had a pruning loop that only stopped when the live beam set
*and* the finished set both reached `beam_width`. Finished beams accumulate
across the whole search, so on early steps that condition never held, the live
beam set was never trimmed, and the number of expansions grew as
`vocab ** max_len` instead of `beam_width * max_len`.

Every ranking test still passed. The returned top-k was the same either way,
because a wider search only ever finds more candidates, and the best one was
already there. The three existing tests asserted on results and never on work
done.

Measured on a deliberately small case, 10 token vocabulary, `beam_width=4`,
3 steps:

| | expansions |
| --- | --- |
| before | 111 |
| after | 9 |

At a real T5 vocabulary the old code built `32000 ** 32` candidate objects and
never returned. `tests/test_beam_search.py::TestCost` now pins the expansion
count rather than wall-clock time, because timing a unit test just gives you a
suite that fails on a loaded CI machine.

The same class of bug is why this repository's CI includes a data job that
regenerates the corpus twice and diffs it. Silent drift is the expensive kind.

## Tests

```bash
make test
```

```
tests/test_beam_search.py     ranking, termination, and the work budget
tests/test_attention.py       mask blocks the future, weights sum to 1, grads flow
tests/test_linearise_contract.py  the wire format, pinned as exact strings
tests/test_corpus.py          the generator is reproducible
```

Attention tests check properties rather than shapes. One scrambles the last
position of the input and asserts the earlier outputs do not move, which is what
a causal mask is *for*. A shape assertion would have passed on an unmasked
implementation.

## The linearise duplication

`linearise_spec` is 103 lines and it lives in two repositories, this one and
[catalogue-rag-qa](../catalogue-rag-qa), because the serving model server
imports it to compose answers and neither repo should depend on the other at
runtime for a pure string function.

A copy is a drift risk, so `tests/test_linearise_contract.py` pins the output
as exact strings and that same file is kept in both repositories. Change the
format in one place and both test suites fail. That is the intent.

## Relationship to catalogue-rag-qa

[`catalogue-rag-qa`](../catalogue-rag-qa) is the other half: a multimodal
retrieval and Q&A system over the same kind of catalogue. It calls this
model's output to propose descriptions, and it needs an editor approval step
before any of them are published. The two are independent repositories and
neither imports the other at runtime.

## Stack

Python 3.10+, PyTorch 2, HuggingFace Transformers, pytest. No framework, no
config files, no service to run.

## License

MIT. See [LICENSE](LICENSE).
