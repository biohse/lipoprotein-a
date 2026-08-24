from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ("Age", "Gender", "HDL", "LDL", "TG", "TC")


def load_cohort(path: Path, header: int = 4) -> pd.DataFrame:
    data = pd.read_excel(path, header=header)
    columns = ["pID", *FEATURES, "LPA"]
    return data.loc[data["Age"].between(18, 80), columns].reset_index(drop=True)


def lpa_bins(lpa: pd.Series, cutoffs: Sequence[float]) -> pd.Series:
    return pd.Series(np.digitize(lpa, cutoffs), index=lpa.index, name="__lpa_bin__")


def make_split(data: pd.DataFrame, cutoffs: Sequence[float], seed: int) -> pd.DataFrame:
    strata = lpa_bins(data["LPA"], cutoffs)  # pyright: ignore[reportArgumentType]
    train, remainder = train_test_split(data.index, train_size=0.7, stratify=strata, random_state=seed)
    validation, test = train_test_split(
        remainder,
        train_size=1 / 3,
        stratify=strata.loc[remainder],
        random_state=seed,
    )

    split = pd.Series(index=data.index, dtype="string", name="__split__")
    split.loc[train] = "train"
    split.loc[validation] = "validation"
    split.loc[test] = "test"
    return pd.concat([data["pID"], split], axis=1).reset_index(drop=True)
