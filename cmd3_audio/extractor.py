import pickle

import hydra
from omegaconf import DictConfig
import os
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch

from data.datamodule import CMD3DataModule
from models.model import CMD3Audio


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    feature_extractor = CMD3Audio.load_from_checkpoint(
        cfg.full_checkpoint,
        model_name=cfg.model.model_name,
        model_path = None,
        feature_extraction=cfg.model.feature_extraction,
        optimizer=cfg.model.optimizer,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        momentum=cfg.model.momentum,
    )
    feature_extractor.model.classifier = torch.nn.Identity()

    data_module = CMD3DataModule(
        root_dir=cfg.root_dir,
        batch_size=cfg.compnode.batch_size,
        num_workers=cfg.compnode.num_workers,
    )

    trainer = Trainer(
        gpus=cfg.compnode.num_gpus,
        num_nodes=cfg.compnode.num_nodes,
        accelerator=cfg.compnode.accelerator,
        logger=False,
        precision=16,
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    trainer.test(feature_extractor, data_module)

    features, ids = [], []
    for feature_output in feature_extractor.test_outputs:
        features.append(feature_output["preds"].cpu())
        ids.extend(feature_output["id"])
    features = torch.cat(features)

    if not os.path.exists(cfg.feature_dir):
        os.makedirs(cfg.feature_dir)

    feature_path = os.path.join(cfg.feature_dir, "features.pk")
    with open(feature_path, "wb") as f:
        pickle.dump(features, f)
    id_path = os.path.join(cfg.feature_dir, "ids.pk")
    with open(id_path, "wb") as f:
        pickle.dump(ids, f)


if __name__ == "__main__":
    main()
