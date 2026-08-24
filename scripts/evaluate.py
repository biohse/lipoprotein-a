from functools import partial
from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, roc_auc_score

from lpa.evaluation import (
    binary_metrics,
    bootstrap_metrics,
    metric_summary,
    regression_metrics,
    select_threshold,
)

COUNT_METRICS = ["tp", "fp", "fn", "tn"]


def load_predictions(path: Path) -> pd.DataFrame:
    return pd.concat((pd.read_csv(file) for file in sorted(path.glob("*.csv"))), ignore_index=True)


def select_models(validation: pd.DataFrame) -> tuple[str | None, dict[float, str], pd.DataFrame]:
    rows = []
    regression = validation.loc[validation["task"] == "regression"].drop_duplicates(["model", "pID"])
    regression_scores = {
        model: mean_absolute_error(group["lpa"], group["score"]) for model, group in regression.groupby("model")
    }
    selected_regression = min(regression_scores, key=lambda model: (regression_scores[model], model), default=None)
    rows.extend(
        {
            "task": "regression",
            "cutoff": None,
            "model": model,
            "validation_metric": "mae",
            "value": value,
            "selected": model == selected_regression,
        }
        for model, value in regression_scores.items()
    )

    selected_classifiers = {}
    classifiers = validation.loc[validation["task"] == "classification"]
    for cutoff, cutoff_data in classifiers.groupby("cutoff"):
        scores = {
            model: roc_auc_score(group["target"], group["score"]) for model, group in cutoff_data.groupby("model")
        }
        selected_classifiers[cutoff] = min(scores, key=lambda model: (-scores[model], model))
        rows.extend(
            {
                "task": "classification",
                "cutoff": cutoff,
                "model": model,
                "validation_metric": "auroc",
                "value": value,
                "selected": model == selected_classifiers[cutoff],
            }
            for model, value in scores.items()
        )
    return selected_regression, selected_classifiers, pd.DataFrame(rows)


def validation_thresholds(validation: pd.DataFrame, min_specificity: float) -> pd.DataFrame:
    rows = []
    for (model, task, cutoff), group in validation.groupby(  # pyright: ignore[reportGeneralTypeIssues]
        ["model", "task", "cutoff"]
    ):
        threshold = select_threshold(group["target"].to_numpy(), group["score"].to_numpy(), min_specificity)
        metrics = binary_metrics(group["target"].to_numpy(), group["score"].to_numpy(), threshold)
        rows.append(
            {
                "model": model,
                "task": task,
                "cutoff": cutoff,
                "threshold": threshold,
                "validation_auroc": metrics["auroc"],
                "validation_sensitivity": metrics["sensitivity"],
                "validation_specificity": metrics["specificity"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_binary(
    test: pd.DataFrame,
    thresholds: pd.DataFrame,
    selected_regression: str | None,
    selected_classifiers: dict[float, str],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    replicates = []
    threshold_lookup = thresholds.set_index(["model", "task", "cutoff"])["threshold"]
    for (model, task, cutoff), group in test.groupby(  # pyright: ignore[reportGeneralTypeIssues]
        ["model", "task", "cutoff"]
    ):
        group = group.sort_values("pID")
        y_true, score = group["target"].to_numpy(), group["score"].to_numpy()
        threshold = threshold_lookup.loc[(model, task, cutoff)]
        point = binary_metrics(y_true, score, threshold)
        bootstrap = bootstrap_metrics(
            y_true,
            score,
            partial(binary_metrics, threshold=threshold),
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        selected = model == (selected_regression if task == "regression" else selected_classifiers[cutoff])
        summaries.append(
            metric_summary(point, bootstrap).assign(model=model, task=task, cutoff=cutoff, selected=selected)
        )
        replicates.append(bootstrap.drop(columns=COUNT_METRICS).assign(model=model, task=task, cutoff=cutoff))
    return pd.concat(summaries, ignore_index=True), pd.concat(replicates, ignore_index=True)


def evaluate_continuous(
    test: pd.DataFrame,
    selected_model: str | None,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    replicates = []
    regression = test.loc[test["task"] == "regression"].drop_duplicates(["model", "pID"])
    for model, group in regression.groupby("model"):
        group = group.sort_values("pID")
        y_true, score = group["lpa"].to_numpy(), group["score"].to_numpy()
        point = regression_metrics(y_true, score)
        bootstrap = bootstrap_metrics(y_true, score, regression_metrics, n_bootstrap=n_bootstrap, seed=seed)
        summaries.append(metric_summary(point, bootstrap).assign(model=model, selected=model == selected_model))
        replicates.append(bootstrap.assign(model=model))
    return pd.concat(summaries, ignore_index=True), pd.concat(replicates, ignore_index=True)


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(config: DictConfig) -> None:
    predictions = load_predictions(Path(to_absolute_path(config.predictions_dir)))
    expected = {*config.expected_models.regression, *config.expected_models.classification}
    missing = expected - set(predictions["model"])
    if missing:
        raise RuntimeError(f"Missing model predictions: {', '.join(sorted(missing))}")

    validation = predictions.loc[predictions["__split__"] == "validation"]

    selected_regression, selected_classifiers, selection = select_models(validation)
    thresholds = validation_thresholds(validation, config.min_specificity)

    output = Path(to_absolute_path(config.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output / "validation_selection.csv", index=False)
    thresholds.to_csv(output / "validation_thresholds.csv", index=False)

    test = predictions.loc[predictions["__split__"] == "test"]
    binary, binary_bootstrap = evaluate_binary(
        test,
        thresholds,
        selected_regression,
        selected_classifiers,
        config.bootstrap_replicates,
        config.bootstrap_seed,
    )
    continuous, continuous_bootstrap = evaluate_continuous(
        test,
        selected_regression,
        config.bootstrap_replicates,
        config.bootstrap_seed,
    )

    binary.to_csv(output / "binary_metrics.csv", index=False)
    continuous.to_csv(output / "continuous_metrics.csv", index=False)
    binary_bootstrap.to_csv(output / "binary_bootstrap.csv", index=False)
    continuous_bootstrap.to_csv(output / "continuous_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
