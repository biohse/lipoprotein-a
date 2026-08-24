# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportOptionalMemberAccess=false

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from lpa.data import FEATURES
from lpa.training import PREDICTION_COLUMNS, load_data


def _scores(frame: Any) -> np.ndarray:
    return frame.as_data_frame().iloc[:, -1].to_numpy()


def train(config: DictConfig) -> pd.DataFrame:
    import h2o  # pyright: ignore[reportMissingImports]
    from h2o.automl import H2OAutoML  # pyright: ignore[reportMissingImports]

    data = load_data(config)
    train_rows = data["__split__"] == "train"
    train_data = data.loc[train_rows]
    score_data = data.loc[data["__split__"].isin(["validation", "test"])]
    output = Path(to_absolute_path(config.results_dir))
    predictions = []

    h2o.init(
        nthreads=config.model.nthreads,
        max_mem_size=config.model.max_mem_size,
        verbose=False,
    )
    try:
        for cutoff in config.cohort.cutoffs:
            h2o.remove_all()
            target = (train_data["LPA"] >= cutoff).astype(int)
            frame_data = train_data.loc[:, FEATURES].copy()
            frame_data["__target__"] = target.to_numpy()
            training_frame = h2o.H2OFrame(frame_data)
            training_frame["__target__"] = training_frame["__target__"].asfactor()

            model = H2OAutoML(
                nfolds=config.model.nfolds,
                max_models=config.model.max_models,
                max_runtime_secs=config.model.max_runtime_secs,
                max_runtime_secs_per_model=config.model.max_runtime_secs_per_model,
                seed=config.seed,
                sort_metric=config.model.metric,
                stopping_metric=config.model.metric,
                balance_classes=config.model.balance_classes,
                include_algos=list(config.model.include_algos),
                keep_cross_validation_predictions=True,
                keep_cross_validation_models=False,
                verbosity="info",
            )
            model.train(x=list(FEATURES), y="__target__", training_frame=training_frame)

            fit_path = output / "models" / config.cohort.name / config.model.name / f"{cutoff:g}"
            fit_path.mkdir(parents=True, exist_ok=True)
            leaderboard = model.get_leaderboard(extra_columns="ALL").as_data_frame()
            leaderboard.insert(1, "model_family", leaderboard["model_id"].str.split("_").str[0])
            leaderboard = leaderboard.sort_values(
                ["auc", "logloss", "model_id"], ascending=[False, True, True], ignore_index=True
            )
            leaderboard.to_csv(fit_path / "leaderboard.csv", index=False)
            model.event_log.as_data_frame().to_csv(fit_path / "event_log.csv", index=False)
            leader = h2o.get_model(leaderboard.loc[0, "model_id"])
            h2o.save_model(
                leader,
                path=str(fit_path),
                filename="model",
                force=True,
                export_cross_validation_predictions=True,
            )

            oof_row = leaderboard.loc[leaderboard["model_family"] != "StackedEnsemble"].iloc[0]
            oof_model = h2o.get_model(oof_row["model_id"])
            oof_score = _scores(oof_model.cross_validation_holdout_predictions())
            pd.DataFrame({"pID": train_data["pID"], "target": target, "score": oof_score}).to_csv(
                fit_path / "oof_predictions.csv", index=False
            )
            selection = {
                "leader": leader.model_id,
                "model_family": leaderboard.loc[0, "model_family"],
                "cv_roc_auc": float(leaderboard.loc[0, "auc"]),
                "oof_model": oof_row["model_id"],
            }
            (fit_path / "selection.yaml").write_text(OmegaConf.to_yaml(selection), encoding="utf-8")
            (fit_path / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")

            score_frame = h2o.H2OFrame(score_data.loc[:, FEATURES])
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
                        "score": _scores(leader.predict(score_frame)),
                    }
                )
            )
    finally:
        h2o.cluster().shutdown(prompt=False)

    prediction = pd.concat(predictions, ignore_index=True).loc[:, PREDICTION_COLUMNS]
    prediction = prediction.sort_values(["cutoff", "__split__", "pID"], ignore_index=True)
    path = output / "predictions" / config.cohort.name / f"{config.model.name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(path, index=False)
    return prediction
