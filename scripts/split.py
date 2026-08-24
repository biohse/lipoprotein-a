from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from lpa.data import load_cohort, make_split


@hydra.main(version_base=None, config_path="configs", config_name="split")
def main(config: DictConfig) -> None:
    cohort = config.cohort
    data = load_cohort(Path(to_absolute_path(cohort.data_path)))
    split = make_split(data, cohort.cutoffs, config.seed)

    output = Path(to_absolute_path(cohort.split_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(output, index=False)


if __name__ == "__main__":
    main()
