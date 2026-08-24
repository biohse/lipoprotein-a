# Molar versus mass units in machine learning-based lipoprotein(a) screening

This repository contains code for the paper “Molar versus mass units in machine learning-based lipoprotein(a) screening”.

## Installation

```bash
git clone https://github.com/biohse/lipoprotein-a && cd lipoprotein-a
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

Install all optional model backends in the same environment when reproducing the complete analysis:

```bash
uv pip install -e ".[dnn,h2o,lightautoml,pysr,xgboost]"
```

H2O AutoML was run with Java 17, and PySR with Julia 1.10.3.

## Usage

Scripts use [Hydra](https://hydra.cc/docs/intro/) for configuration. Select a cohort and model with command-line
overrides:

```bash
python scripts/train.py cohort=mass_f1 model=logistic
```

Use `-m` for a multirun:

```bash
python scripts/train.py -m cohort=mass_f1,mass_p1,molar_p2 model=linear,logistic
```

## Reproduction

Download the Zenodo archive into `data/`.

If needed, regenerate the fixed splits from Tables S2–S4:

```bash
python scripts/split.py -m cohort=mass_f1,mass_p1,molar_p2
```

Train all models:

```bash
MODELS=linear,quadratic,lasso,rf_regressor,svr,logistic,rf_classifier,svc,xgboost,dnn,h2o_automl,lightautoml,pysr
python scripts/train.py -m cohort=mass_f1,mass_p1,molar_p2 model="$MODELS"
```

After all predictions have been generated, evaluate every cohort:

```bash
python scripts/evaluate.py -m cohort=mass_f1,mass_p1,molar_p2
```

## Data Availability

Raw Tables S2–S4 and the exact splits used in the paper are available on
[Zenodo](https://zenodo.org/records/22078402).
