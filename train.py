import hydra
from omegaconf import DictConfig, OmegaConf

from asr.trainer import Trainer
from asr.data import get_dataloader, get_dataloaders
from asr.base import BaseTextEncoder, BaseModel
from asr.base.serialization import Serialization

from asr.logger import asr_logger as logger


@hydra.main(config_path="./asr/configs/", config_name="one_batch_test.yaml")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg))

    # setup text encoder
    text_encoder = BaseTextEncoder.from_config(
        cfg.model.text_encoder
    )

    # prepeare loaders
    train_dataloader = get_dataloader(
        cfg.model.train_dataloader, text_encoder,
    )
    val_dataloaders = get_dataloaders(
        cfg.model.val_dataloader, text_encoder,
    )

    # setup model
    asr_model = BaseModel.from_config(cfg.model.architecture)
    asr_model.to(cfg.trainer.device)
    logger.info(asr_model)

    # setup loss, optimizer, scheduler
    asr_loss = Serialization.from_config(cfg.model.loss)

    trainable_params = filter(
        lambda p: p.requires_grad, asr_model.parameters()
    )
    optimizer = Serialization.from_config(
        cfg.model.optimizer, params=trainable_params,
    )
    lr_scheduler = Serialization.from_config(
        cfg.model.lr_scheduler, optimizer=optimizer,
    )

    # initialize trainer
    trainer = Trainer(
        asr_model,
        asr_loss,
        optimizer,
        text_encoder=text_encoder,
        cfg=cfg.trainer,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        lr_scheduler=lr_scheduler,
        len_epoch=cfg.trainer.get("len_epoch", None),
    )
    trainer.train()


if __name__ == "__main__":
    main()
