import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lpa.data import FEATURES
from lpa.training import PREDICTION_COLUMNS, load_data


def build_model(n_features: int, learning_rate: float) -> Any:
    import keras  # pyright: ignore[reportMissingImports]

    keras.backend.clear_session()
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def _set_seed(seed: int) -> None:
    import keras  # pyright: ignore[reportMissingImports]
    import tensorflow as tf  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def _class_weight(target: np.ndarray, mode: str | None) -> dict[int, float] | None:
    if mode is None:
        return None
    counts = np.bincount(target, minlength=2)
    return {label: len(target) / (2 * count) for label, count in enumerate(counts)}


def _fit_early(
    features: np.ndarray,
    target: np.ndarray,
    config: DictConfig,
    class_weight: str | None,
    seed: int,
) -> tuple[Any, int, pd.DataFrame]:
    import keras  # pyright: ignore[reportMissingImports]

    fit_indices, stop_indices = train_test_split(
        np.arange(len(target)),
        test_size=config.validation_size,
        stratify=target,
        random_state=seed,
    )
    _set_seed(seed)
    model = build_model(features.shape[1], config.learning_rate)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=config.patience,
        min_delta=config.min_delta,
        restore_best_weights=True,
    )
    history = model.fit(
        features[fit_indices],
        target[fit_indices],
        validation_data=(features[stop_indices], target[stop_indices]),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        class_weight=_class_weight(target[fit_indices], class_weight),
        callbacks=[early_stopping],
        verbose=0,
    )
    frame = pd.DataFrame(history.history)
    best_epoch = int(early_stopping.best_epoch + 1)
    return model, best_epoch, frame


def _select_weight(
    features: pd.DataFrame,
    target: pd.Series,
    preprocessor: Any,
    config: DictConfig,
    cv: Any,
    seed: int,
) -> tuple[str | None, pd.DataFrame]:
    rows = []
    for mode in config.class_weights:
        for fold, (fit_indices, score_indices) in enumerate(cv.split(features, target)):
            transform = clone(preprocessor).fit(  # pyright: ignore[reportAttributeAccessIssue]
                features.iloc[fit_indices]
            )
            fit_features = np.asarray(transform.transform(features.iloc[fit_indices]))
            score_features = np.asarray(transform.transform(features.iloc[score_indices]))
            model, best_epoch, _ = _fit_early(
                fit_features,
                target.iloc[fit_indices].to_numpy(),
                config,
                mode,
                seed + fold,
            )
            score = model.predict(score_features, batch_size=config.batch_size, verbose=0).ravel()
            rows.append(
                {
                    "class_weight": mode or "none",
                    "fold": fold,
                    "auroc": roc_auc_score(target.iloc[score_indices], score),
                    "best_epoch": best_epoch,
                }
            )
    results = pd.DataFrame(rows)
    means = results.groupby("class_weight", sort=False)["auroc"].mean()
    selected = str(means.idxmax())
    return (None if selected == "none" else selected), results


def _prediction_rows(
    data: pd.DataFrame,
    preprocessor: Any,
    model: Any,
    cohort: str,
    model_name: str,
    cutoff: float,
    batch_size: int,
) -> pd.DataFrame:
    rows = data.loc[data["__split__"].isin(["validation", "test"])]
    features = np.asarray(preprocessor.transform(rows.loc[:, FEATURES]))
    return pd.DataFrame(
        {
            "cohort": cohort,
            "model": model_name,
            "task": "classification",
            "cutoff": cutoff,
            "pID": rows["pID"],
            "__split__": rows["__split__"],
            "lpa": rows["LPA"],
            "target": (rows["LPA"] >= cutoff).astype(int),
            "score": model.predict(features, batch_size=batch_size, verbose=0).ravel(),
        }
    )


def train(config: DictConfig) -> pd.DataFrame:
    for name, value in config.model.environment.items():
        os.environ[str(name)] = str(value)

    data = load_data(config)
    train_rows = data["__split__"] == "train"
    features = data.loc[train_rows, FEATURES]
    lpa = data.loc[train_rows, "LPA"]
    output = Path(to_absolute_path(config.results_dir))
    predictions = []

    for cutoff in config.cohort.cutoffs:
        target = (lpa >= cutoff).astype(int)
        preprocessor = instantiate(config.preprocessing, _convert_="all")
        selected_weight, cv_results = _select_weight(
            features,
            target,
            preprocessor,
            config.model,
            instantiate(config.cv.classification),
            config.seed,
        )

        preprocessor.fit(features)
        transformed = np.asarray(preprocessor.transform(features))
        _, best_epoch, history = _fit_early(transformed, target.to_numpy(), config.model, selected_weight, config.seed)
        _set_seed(config.seed)
        model = build_model(transformed.shape[1], config.model.learning_rate)
        model.fit(
            transformed,
            target,
            epochs=best_epoch,
            batch_size=config.model.batch_size,
            class_weight=_class_weight(target.to_numpy(), selected_weight),
            verbose=0,
        )

        fit_path = output / "models" / config.cohort.name / config.model.name / f"{cutoff:g}"
        fit_path.mkdir(parents=True, exist_ok=True)
        model.save(fit_path / "model.keras")
        joblib.dump(preprocessor, fit_path / "preprocessor.joblib")
        cv_results.to_csv(fit_path / "cv_results.csv", index=False)
        history.to_csv(fit_path / "history.csv", index=False)
        selection = {"class_weight": selected_weight, "best_epoch": best_epoch}
        (fit_path / "selection.yaml").write_text(OmegaConf.to_yaml(selection), encoding="utf-8")
        (fit_path / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
        predictions.append(
            _prediction_rows(
                data,
                preprocessor,
                model,
                config.cohort.name,
                config.model.name,
                cutoff,
                config.model.batch_size,
            )
        )

    prediction = pd.concat(predictions, ignore_index=True).loc[:, PREDICTION_COLUMNS]
    prediction = prediction.sort_values(["cutoff", "__split__", "pID"], ignore_index=True)
    path = output / "predictions" / config.cohort.name / f"{config.model.name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(path, index=False)
    return prediction
