from typing import Any, ClassVar, Self

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class QuadraticFeatures(PolynomialFeatures):
    """Generate quadratic features with optional pairwise products."""

    _parameter_constraints: ClassVar[dict[str, list[str]]] = {  # pyright: ignore[reportIncompatibleVariableOverride]
        "include_pairwise": ["boolean"]
    }

    def __init__(self, *, include_pairwise: bool = True) -> None:
        super().__init__(degree=2, include_bias=False)
        self.include_pairwise = include_pairwise

    @property
    def powers_(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> np.ndarray[Any, np.dtype[np.intp]]:
        return super().powers_[self._feature_mask_]

    def fit(self, X: Any, y: Any = None) -> Self:
        super().fit(X, y)
        powers = super().powers_
        self._feature_mask_ = (
            np.ones(len(powers), dtype=bool) if self.include_pairwise else np.count_nonzero(powers, axis=1) <= 1
        )
        self.n_output_features_ = int(self._feature_mask_.sum())
        return self

    def transform(self, X: Any) -> Any:
        return super().transform(X)[:, self._feature_mask_]  # pyright: ignore[reportIndexIssue]
