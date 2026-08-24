from pathlib import Path
from shutil import copy2
from typing import Any

import numpy as np
import pandas as pd
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import mean_absolute_error

from lpa.data import FEATURES
from lpa.training import PREDICTION_COLUMNS, load_data


def _target(values: np.ndarray, transform: str) -> np.ndarray:
    return np.log1p(values) if transform == "log1p" else values


def _prediction(values: np.ndarray, transform: str) -> np.ndarray:
    return np.expm1(values) if transform == "log1p" else values


def _model(config: DictConfig, output: Path, seed: int) -> Any:
    return instantiate(
        config.estimator,
        output_directory=str(output),
        random_state=seed,
        _convert_="all",
    )


def _save_julia_environment(output: Path) -> None:
    from juliacall import Main as julia  # pyright: ignore[reportMissingImports]

    project = Path(str(julia.seval("Base.active_project()")))
    copy2(project, output / "Project.toml")
    copy2(project.with_name("Manifest.toml"), output / "Manifest.toml")


def train(config: DictConfig) -> pd.DataFrame:
    data = load_data(config)
    train_rows = data["__split__"] == "train"
    features = data.loc[train_rows, FEATURES]
    lpa = data.loc[train_rows, "LPA"].to_numpy()
    score_data = data.loc[data["__split__"].isin(["validation", "test"])]
    output = Path(to_absolute_path(config.results_dir))
    fit_path = output / "models" / config.cohort.name / config.model.name
    search_path = fit_path / "searches"
    fit_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for transform in config.model.target_transforms:
        for fold, (fit_indices, score_indices) in enumerate(instantiate(config.cv.regression).split(features)):
            seed = config.seed + fold
            model = _model(config.model, search_path, seed)
            model.fit(features.iloc[fit_indices], _target(lpa[fit_indices], transform))
            score = _prediction(np.asarray(model.predict(features.iloc[score_indices])), transform)
            rows.append(
                {
                    "target_transform": transform,
                    "fold": fold,
                    "seed": seed,
                    "mae": mean_absolute_error(lpa[score_indices], score),
                    "equation": str(model.sympy()),
                    "run_id": model.run_id_,
                }
            )

    cv_results = pd.DataFrame(rows)
    cv_mae = cv_results.groupby("target_transform", sort=False)["mae"].mean()
    selected = str(cv_mae.idxmin())
    model = _model(config.model, search_path, config.seed)
    model.fit(features, _target(lpa, selected))
    score = _prediction(np.asarray(model.predict(score_data.loc[:, FEATURES])), selected)

    cv_results.to_csv(fit_path / "cv_results.csv", index=False)
    model.equations_.to_csv(fit_path / "equations.csv", index=False)
    (fit_path / "equation.txt").write_text(str(model.sympy()), encoding="utf-8")
    (fit_path / "equation.tex").write_text(str(model.latex()), encoding="utf-8")
    _save_julia_environment(fit_path)
    run_directory = Path("searches") / model.run_id_
    selection = {
        "target_transform": selected,
        "cv_neg_mae": -float(cv_mae.at[selected]),
        "run_id": model.run_id_,
        "run_directory": str(run_directory),
        "checkpoint": str(run_directory / "checkpoint.pkl"),
    }
    (fit_path / "selection.yaml").write_text(OmegaConf.to_yaml(selection), encoding="utf-8")
    (fit_path / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")

    predictions = []
    for cutoff in config.cohort.cutoffs:
        predictions.append(
            pd.DataFrame(
                {
                    "cohort": config.cohort.name,
                    "model": config.model.name,
                    "task": "regression",
                    "cutoff": cutoff,
                    "pID": score_data["pID"],
                    "__split__": score_data["__split__"],
                    "lpa": score_data["LPA"],
                    "target": (score_data["LPA"] >= cutoff).astype(int),
                    "score": score,
                }
            )
        )

    prediction = pd.concat(predictions, ignore_index=True).loc[:, PREDICTION_COLUMNS]
    prediction = prediction.sort_values(["cutoff", "__split__", "pID"], ignore_index=True)
    path = output / "predictions" / config.cohort.name / f"{config.model.name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(path, index=False)
    return prediction
