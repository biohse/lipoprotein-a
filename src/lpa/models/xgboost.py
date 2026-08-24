from typing import Any, Literal, Self

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBoost classifier with fold-local balanced class weights."""

    def __init__(
        self,
        *,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        n_estimators: int = 100,
        scale_pos_weight: float | Literal["balanced"] = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> Self:
        from xgboost import XGBClassifier as NativeXGBClassifier  # pyright: ignore[reportMissingImports]

        target = np.asarray(y)
        weight = (
            np.count_nonzero(target == 0) / np.count_nonzero(target == 1)
            if self.scale_pos_weight == "balanced"
            else self.scale_pos_weight
        )
        self.model_ = NativeXGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            scale_pos_weight=weight,
            subsample=1.0,
            colsample_bytree=1.0,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(X)
