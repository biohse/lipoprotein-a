from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from lpa.data import FEATURES, load_cohort
from lpa.evaluation import model_score

PREDICTION_COLUMNS = ("cohort", "model", "task", "cutoff", "pID", "__split__", "lpa", "target", "score")


def _target_transformer(name: str, standardize: bool) -> Any | None:
    steps = []
    if name == "log1p":
        steps.append(("log1p", FunctionTransformer(np.log1p, inverse_func=np.expm1)))
    if standardize:
        steps.append(("scale", StandardScaler()))
    if not steps:
        return None
    return steps[0][1] if len(steps) == 1 else Pipeline(steps)


def fit_regressor(
    features: pd.DataFrame,
    target: pd.Series,
    preprocessor: Any,
    estimator: Any,
    param_grid: Any,
    cv: Any,
    target_transforms: Sequence[str],
    standardize_target: bool,
    n_jobs: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    searches = []
    results = []
    for target_transform in target_transforms:
        pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])
        model = TransformedTargetRegressor(
            regressor=pipeline,
            transformer=_target_transformer(target_transform, standardize_target),
        )
        search = GridSearchCV(
            model,
            param_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=n_jobs,
            refit=False,
            error_score="raise",  # pyright: ignore[reportArgumentType]
        ).fit(features, target)
        frame = pd.DataFrame(search.cv_results_)
        frame.insert(0, "target_transform", target_transform)
        searches.append((target_transform, search))
        results.append(frame)

    target_transform, search = max(searches, key=lambda item: item[1].best_score_)
    selection = {
        "target_transform": target_transform,
        "best_params": search.best_params_,
        "cv_neg_mae": float(search.best_score_),
    }
    model = (
        clone(search.estimator)
        .set_params(  # pyright: ignore[reportAttributeAccessIssue]
            **search.best_params_
        )
        .fit(features, target)
    )
    return model, pd.concat(results, ignore_index=True), selection


def fit_classifier(
    features: pd.DataFrame,
    target: pd.Series,
    preprocessor: Any,
    estimator: Any,
    param_grid: Any,
    cv: Any,
    n_jobs: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",  # pyright: ignore[reportArgumentType]
    ).fit(features, target)
    selection = {"best_params": search.best_params_, "cv_roc_auc": float(search.best_score_)}
    return search.best_estimator_, pd.DataFrame(search.cv_results_), selection


def prediction_rows(
    data: pd.DataFrame,
    model: Any,
    *,
    cohort: str,
    model_name: str,
    task: str,
    cutoff: float,
) -> pd.DataFrame:
    rows = data.loc[data["__split__"].isin(["validation", "test"])]
    return pd.DataFrame(
        {
            "cohort": cohort,
            "model": model_name,
            "task": task,
            "cutoff": cutoff,
            "pID": rows["pID"],
            "__split__": rows["__split__"],
            "lpa": rows["LPA"],
            "target": (rows["LPA"] >= cutoff).astype(int),
            "score": model_score(model, rows.loc[:, FEATURES]),
        }
    )


def _save_fit(path: Path, model: Any, cv_results: pd.DataFrame, selection: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.joblib")
    cv_results.to_csv(path / "cv_results.csv", index=False)
    (path / "selection.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(selection)), encoding="utf-8")


def load_data(config: DictConfig) -> pd.DataFrame:
    cohort = load_cohort(Path(to_absolute_path(config.cohort.data_path)))
    split = pd.read_csv(to_absolute_path(config.cohort.split_path))
    return cohort.merge(split, on="pID", validate="one_to_one")


def train(config: DictConfig) -> pd.DataFrame:
    data = load_data(config)
    train_rows = data["__split__"] == "train"
    features = data.loc[train_rows, FEATURES]
    lpa = data.loc[train_rows, "LPA"]
    output = Path(to_absolute_path(config.results_dir))
    fit_path = output / "models" / config.cohort.name / config.model.name
    fit_path.mkdir(parents=True, exist_ok=True)
    (fit_path / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")

    param_grid = OmegaConf.to_container(config.model.param_grid, resolve=True)
    predictions = []
    if config.model.task == "regression":
        model, cv_results, selection = fit_regressor(
            features,
            lpa,
            instantiate(config.preprocessing, _convert_="all"),
            instantiate(config.model.estimator),
            param_grid,
            instantiate(config.cv.regression),
            config.model.target_transforms,
            config.model.standardize_target,
            config.n_jobs,
        )
        for cutoff in config.cohort.cutoffs:
            rows = prediction_rows(
                data,
                model,
                cohort=config.cohort.name,
                model_name=config.model.name,
                task="regression",
                cutoff=cutoff,
            )
            predictions.append(rows)
        _save_fit(fit_path, model, cv_results, selection)
    else:
        for cutoff in config.cohort.cutoffs:
            target = (lpa >= cutoff).astype(int)
            model, cv_results, selection = fit_classifier(
                features,
                target,
                instantiate(config.preprocessing, _convert_="all"),
                instantiate(config.model.estimator),
                param_grid,
                instantiate(config.cv.classification),
                config.n_jobs,
            )
            rows = prediction_rows(
                data,
                model,
                cohort=config.cohort.name,
                model_name=config.model.name,
                task="classification",
                cutoff=cutoff,
            )
            _save_fit(fit_path / f"{cutoff:g}", model, cv_results, selection)
            predictions.append(rows)

    prediction = pd.concat(predictions, ignore_index=True).loc[:, PREDICTION_COLUMNS]
    prediction = prediction.sort_values(["cutoff", "__split__", "pID"], ignore_index=True)
    prediction_path = output / "predictions" / config.cohort.name / f"{config.model.name}.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(prediction_path, index=False)
    return prediction
