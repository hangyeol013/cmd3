import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from data.datamodule import CMD3DataModule
from models.model import CMD3Video


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    model = CMD3Video(
        model_name=cfg.model.model_name,
        pretrained_path=cfg.model.pretrained_path,
        feature_extraction=cfg.model.feature_extraction,
        optimizer=cfg.model.optimizer,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        momentum=cfg.model.momentum,
    )
    data_module = CMD3DataModule(
        root_dir=cfg.root_dir,
        modalities=cfg.model.modalities,
        frame_size=cfg.model.frame_size,
        n_samples=cfg.model.n_samples,
        clip_duration=cfg.clip_duration,
        batch_size=cfg.compnode.batch_size,
        num_workers=cfg.compnode.num_workers,
        augmentation=cfg.augmentation,
        normalize=cfg.model.normalize,
    )

    wandb_logger = WandbLogger(
        name=cfg.xp_name,
        project=cfg.dataset.project,
        offline=True,
    )
    checkpoint = ModelCheckpoint(
        monitor="val/precision",
        mode="max",
        save_last=True,
        dirpath="checkpoints",
        filename=cfg.xp_name + "-{epoch}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        gpus=cfg.compnode.num_gpus,
        num_nodes=cfg.compnode.num_nodes,
        accelerator=cfg.compnode.accelerator,
        max_epochs=cfg.num_epochs,
        callbacks=[lr_monitor, checkpoint],
        logger=wandb_logger,
        log_every_n_steps=5,
        precision=16,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
