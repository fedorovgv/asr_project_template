import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from asr.model import EncDecModel
from asr.logger import logger


@hydra.main(config_path="./asr/configs/", config_name="one_batch_test.yaml")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    asr_model = EncDecModel(cfg.model)

    logger.info(asr_model)

    trainer = pl.Trainer(**cfg.trainer)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()
