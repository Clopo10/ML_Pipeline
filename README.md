# ML_Pipeline

An end-to-end machine learning pipeline for preprocessing, model training/tuning, and inference. This repository collects data artifacts, notebooks, a small inference module, and a lightweight API for serving predictions.

## Features
- Organized data folders for raw, processed, engineered and final datasets
- Reproducible Jupyter notebooks demonstrating baseline and optimized models
- Lightweight inference module and a small API for serving predictions
- Tests for the inference functionality

## Repository Structure

- api/: small app for serving the model ([api/app.py](api/app.py))
- data/: datasets and preprocessing outputs
	- raw/: original source files (physionet training sets)
	- processed/: split data between 2 hospitals
    - cleaned/: intermediate data
	- engineered/: feature-engineered outputs
	- final/: final datasets used for training/evaluation
- models/: exported or tuned model artifacts ([models/tuned_model.json](models/tuned_model.json))
- notebooks/: exploratory and reproducible notebooks
- src/: core inference code ([src/inference.py](src/inference.py))
- tests/: unit tests ([tests/test_inference.py](tests/test_inference.py))
- requirements.txt: Python dependencies

## Quickstart

1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests:

```bash
PYTHONPATH=. pytest
```

4. Run the API:

```bash
uvicorn api.app:app --reload
```


## Notebooks
Open the notebooks in `notebooks/` to reproduce experiments and model tuning steps. They include a baseline pipeline and an optimized pipeline used to produce `models/tuned_model.json`.

## Data
The `data/raw/physionet.org/training` folder contains original dataset files. Downstream folders contain cleaned and engineered data used by the notebooks and training scripts.