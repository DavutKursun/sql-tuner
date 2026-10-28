.PHONY: setup setup-dev test data splits smoke

PY ?= python

setup:
	$(PY) -m pip install torch && $(PY) -m pip install -r requirements.txt
setup-dev:
	$(PY) -m pip install torch && $(PY) -m pip install -r requirements-dev.txt
test:
	$(PY) -m pytest -q
data:             ## Spider from the Hugging Face mirror (CC BY-SA 4.0), verified and unzipped into data/raw/
	$(PY) scripts/download.py --fetch hf
splits:           ## train / val (held-out training databases) / dev / test, gold queries executed once
	$(PY) scripts/build_splits.py
smoke:            ## the whole chain on a miniature Spider and a tiny random model (no download, no GPU)
	$(PY) -m pytest -q tests/test_scripts.py
