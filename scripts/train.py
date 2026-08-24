import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    backend = config.model.get("backend")
    if backend == "dnn":
        from lpa.models.dnn import train
    elif backend == "h2o":
        from lpa.models.h2o_automl import train
    elif backend == "lightautoml":
        from lpa.models.lightautoml import train
    elif backend == "pysr":
        from lpa.models.pysr import train
    else:
        from lpa.training import train

    train(config)


if __name__ == "__main__":
    main()
