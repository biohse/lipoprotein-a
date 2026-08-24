# pyright: reportArgumentType=false

import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from lpa.data import FEATURES
from lpa.training import PREDICTION_COLUMNS, load_data


def _set_seed(seed: int) -> None:
    import torch  # pyright: ignore[reportMissingImports]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _structure(model: Any) -> pd.DataFrame:
    rows = []
    weights = iter(np.asarray(model.blender.wts).tolist())
    for level, pipelines in enumerate(model.levels, start=1):
        for pipeline in pipelines:
            for algorithm in pipeline.ml_algos:
                rows.append(
                    {
                        "level": level,
                        "model": getattr(algorithm, "_name", type(algorithm).__name__),
                        "fold_models": len(algorithm.models),
                        "weight": next(weights),
                    }
                )
    return pd.DataFrame(rows)


def _assert_complete(model: Any, oof_score: np.ndarray, config: DictConfig) -> tuple[pd.DataFrame, int]:
    structure = _structure(model)
    trials = [
        trial
        for pipelines in model.levels
        for pipeline in pipelines
        for tuner in pipeline.params_tuners
        for trial in ([] if getattr(tuner, "study", None) is None else tuner.study.trials)
    ]
    expected_trials = 2 * config.tuning_params.max_tuning_iter
    complete = (
        not bool(model.timer.child_out_of_time)
        and not np.isnan(oof_score).any()
        and len(trials) == expected_trials
        and all(trial.state.name == "COMPLETE" for trial in trials)
        and bool(structure["fold_models"].eq(config.reader_params.cv).all())
    )
    if not complete:
        raise RuntimeError("LightAutoML did not complete the fixed search plan")
    return structure, len(trials)


def train(config: DictConfig) -> pd.DataFrame:
    from lightautoml.automl.presets.tabular_presets import (  # pyright: ignore[reportMissingImports]
        TabularAutoML,
    )
    from lightautoml.tasks import Task  # pyright: ignore[reportMissingImports]

    data = load_data(config)
    train_rows = data["__split__"] == "train"
    train_data = data.loc[train_rows]
    score_data = data.loc[data["__split__"].isin(["validation", "test"])]
    output = Path(to_absolute_path(config.results_dir))
    predictions = []

    for cutoff in config.cohort.cutoffs:
        _set_seed(config.seed)
        target = (train_data["LPA"] >= cutoff).astype(int)
        frame = train_data.loc[:, FEATURES].copy()
        frame["__target__"] = target.to_numpy()
        fit_path = output / "models" / config.cohort.name / config.model.name / f"{cutoff:g}"
        fit_path.mkdir(parents=True, exist_ok=True)

        model = TabularAutoML(
            task=Task("binary", metric=config.model.metric),
            timeout=config.model.timeout,
            memory_limit=config.model.memory_limit,
            cpu_limit=config.model.cpu_limit,
            gpu_ids=config.model.gpu_ids,
            debug=config.model.debug,
            timing_params=OmegaConf.to_container(config.model.timing_params, resolve=True),
            general_params=OmegaConf.to_container(config.model.general_params, resolve=True),
            reader_params=OmegaConf.to_container(config.model.reader_params, resolve=True),
            tuning_params=OmegaConf.to_container(config.model.tuning_params, resolve=True),
            selection_params=OmegaConf.to_container(config.model.selection_params, resolve=True),
            lgb_params=OmegaConf.to_container(config.model.lgb_params, resolve=True),
            cb_params=OmegaConf.to_container(config.model.cb_params, resolve=True),
        )
        oof_score = model.fit_predict(
            frame,
            roles={"target": "__target__"},
            verbose=config.model.verbose,
        ).data[:, 0]
        structure, tuning_trials = _assert_complete(model, oof_score, config.model)

        joblib.dump(model, fit_path / "model.joblib")
        structure.to_csv(fit_path / "ensemble.csv", index=False)
        pd.DataFrame({"pID": train_data["pID"], "target": target, "score": oof_score}).to_csv(
            fit_path / "oof_predictions.csv", index=False
        )
        resolved = {
            name: getattr(model, name)
            for name in ("general_params", "reader_params", "tuning_params", "lgb_params", "cb_params")
        }
        (fit_path / "resolved_config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(resolved)), encoding="utf-8")
        selection = {
            "cv_roc_auc": float(roc_auc_score(target, oof_score)),
            "ensemble_size": len(structure),
            "tuning_trials": tuning_trials,
        }
        (fit_path / "selection.yaml").write_text(OmegaConf.to_yaml(selection), encoding="utf-8")
        (fit_path / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")

        score = model.predict(score_data.loc[:, FEATURES], n_jobs=1).data[:, 0]
        predictions.append(
            pd.DataFrame(
                {
                    "cohort": config.cohort.name,
                    "model": config.model.name,
                    "task": "classification",
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
