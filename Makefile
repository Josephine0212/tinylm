.PHONY: setup test lint data tokenizer encode train generate serve bench smoke plot

TOK := artifacts/tokenizer-tinystories-10k
CKPT := runs/tinystories-17m/best.pt

setup:            ## install deps (incl. dev + server)
	uv sync --group dev --extra serve

test:
	uv run pytest -q

lint:
	uv run ruff check tinylm scripts server

data:             ## download TinyStories (~2.2 GB)
	./scripts/download_data.sh

tokenizer:        ## train 10k BPE on TinyStories train
	uv run python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt --vocab-size 10000 --out $(TOK)

encode:           ## tokenize train/valid into uint16 .bin
	uv run python scripts/encode_dataset.py --input data/TinyStoriesV2-GPT4-valid.txt --tokenizer $(TOK) --out data/tinystories_valid.bin
	uv run python scripts/encode_dataset.py --input data/TinyStoriesV2-GPT4-train.txt --tokenizer $(TOK) --out data/tinystories_train.bin

train:            ## ~41M tokens on Apple Silicon (MPS)
	uv run python scripts/train.py --config configs/tinystories_mps.json

generate:
	uv run python scripts/generate.py --ckpt $(CKPT) --prompt "Once upon a time"

serve:            ## web demo at http://localhost:8000
	TINYLM_CKPT=$(CKPT) uv run uvicorn server.app:app --port 8000

bench:
	uv run python scripts/benchmark.py bpe --input tests/fixtures/tinystories_sample_5M.txt --vocab-size 1000
	uv run python scripts/benchmark.py decode --ckpt $(CKPT)

plot:
	uv run python scripts/plot_curves.py runs/tinystories-17m --out docs/loss.png

smoke:            ## tiny end-to-end run on the bundled 5MB sample (CPU, ~1 min)
	uv run python scripts/train_tokenizer.py --input tests/fixtures/tinystories_sample_5M.txt --vocab-size 512 --out artifacts/tokenizer-smoke
	uv run python scripts/encode_dataset.py --input tests/fixtures/tinystories_sample_5M.txt --tokenizer artifacts/tokenizer-smoke --out data/smoke_train.bin --workers 2
	uv run python scripts/encode_dataset.py --input tests/fixtures/tinystories_sample.txt --tokenizer artifacts/tokenizer-smoke --out data/smoke_valid.bin --workers 1
	uv run python scripts/train.py --config configs/smoke.json --device cpu
	uv run python scripts/generate.py --ckpt runs/smoke/best.pt --max-new-tokens 40 --device cpu --seed 0
