import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from data.datamodule import CMD3DataModule

from models.model import CMD3Audio


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    # num_class = 3

    # model_urls = {
    #     'vggish': 'https://github.com/harritaylor/torchvggish/'
    #             'releases/download/v0.1/vggish-10086976.pth',
    #     'pca': 'https://github.com/harritaylor/torchvggish/'
    #         'releases/download/v0.1/vggish_pca_params-970ea276.pth'
    # }

    # model = VGGish(urls=model_urls)
    # model.embeddings[4] = nn.Linear(4096, num_class)
    # model.eval()

    model = CMD3Audio(
        model_name=cfg.model.model_name,
        model_path=cfg.model.model_path,
        feature_extraction=cfg.model.feature_extraction,
        optimizer=cfg.model.optimizer,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        momentum=cfg.model.momentum,
    )

    data_module = CMD3DataModule(
        root_dir=cfg.root_dir,
        batch_size=cfg.compnode.batch_size,
        num_workers=cfg.compnode.num_workers,
    )

    wandb_logger = WandbLogger(
        name=cfg.xp_name,
        project=cfg.dataset.project,
        offine=True,
    )
    checkpoint = ModelCheckpoint(
        monitor='val/precision',
        mode='max',
        save_last=True,
        dirpath='checkpoints',
        filename=cfg.xp_name + "-{epoch}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

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
        num_sanity_val_steps=0,
    )

    trainer.fit(model, data_module)



if __name__ == "__main__":
    main()

