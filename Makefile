.PHONY: help install test check dataset train train-1gpu eval notebooks lint clean

PY ?= python3
DATA ?= data/generated
CKPT ?= checkpoints/t5-description
MODEL ?= t5-small
EPOCHS ?= 3
N ?= 3200
SEED ?= 42

help:  ## show this help
	@grep -hE '^[a-z][a-z0-9-]*:.*?##' $(MAKEFILE_LIST) \
	 | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## install the package and dev extras
	$(PY) -m pip install -e ".[dev]"

test:  ## run the test suite
	$(PY) -m pytest tests -q

lint:  ## compile every file
	$(PY) -m compileall -q spec2desc train eval data tests

dataset:  ## generate the deterministic synthetic corpus
	$(PY) data/make_synthetic_dataset.py --out $(DATA) --n $(N) --seed $(SEED)

train:  ## fine-tune T5 on one GPU
	torchrun --nproc_per_node=1 train/train_t5_description.py \
	  --train $(DATA)/train.jsonl --val $(DATA)/val.jsonl \
	  --out $(CKPT) --model $(MODEL) --epochs $(EPOCHS) --seed $(SEED)

train-1gpu: train  ## alias for `make train`

eval:  ## score $(CKPT) against the val split (BLEU, ROUGE, spec fidelity)
	$(PY) eval/evaluate_generation.py --val $(DATA)/val.jsonl --model $(CKPT)

notebooks:  ## regenerate the notebook from its builder
	$(PY) notebooks/build_notebooks.py

check: lint test  ## what CI runs

clean:  ## remove generated data, checkpoints and caches
	rm -rf data/generated checkpoints .pytest_cache htmlcov .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
