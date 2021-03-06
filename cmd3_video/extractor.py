import pickle

import hydra
from omegaconf import DictConfig
import os
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch

from data.datamodule import CMD3DataModule
from models.model import CMD3Video


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    feature_extractor = CMD3Video.load_from_checkpoint(
        cfg.full_checkpoint,
        model_name=cfg.model.model_name,
        model_depth=cfg.model.model_depth,
        pretrained_path=None,
        feature_extraction=cfg.model.feature_extraction,
        optimizer=cfg.model.optimizer,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        momentum=cfg.model.momentum,
    )
    feature_extractor.model.fc = torch.nn.Identity()

    model = CMD3Video.load_from_checkpoint(
        cfg.full_checkpoint,
        model_name=cfg.model.model_name,
        model_depth=cfg.model.model_depth,
        pretrained_path=None,
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

    trainer = Trainer(
        gpus=cfg.compnode.num_gpus,
        num_nodes=cfg.compnode.num_nodes,
        accelerator=cfg.compnode.accelerator,
        logger=False,
        precision=16,
        plugins=DDPPlugin(find_unused_parameters=False),
    )
    trainer.test(feature_extractor, data_module)
    trainer.test(model, data_module)


    features, preds, targets, ids = [], [], [], []
    for feature_output, output in zip(
        feature_extractor.test_outputs, model.test_outputs
    ):
        features.append(feature_output["preds"].cpu())
        preds.append(output["preds"].cpu())
        targets.append(output["targets"].cpu())
        ids.extend(feature_output["id"])
    features = torch.cat(features)
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    if not os.path.exists(cfg.feature_dir):
        os.makedirs(cfg.feature_dir)

    feature_path = os.path.join(cfg.feature_dir, "features.pk")
    with open(feature_path, "wb") as f:
        pickle.dump(features, f)
    pred_path = os.path.join(cfg.feature_dir, "preds.pk")
    with open(pred_path, "wb") as f:
        pickle.dump(preds, f)
    target_path = os.path.join(cfg.feature_dir, "targets.pk")
    with open(target_path, "wb") as f:
        pickle.dump(targets, f)
    id_path = os.path.join(cfg.feature_dir, "ids.pk")
    with open(id_path, "wb") as f:
        pickle.dump(ids, f)


if __name__ == "__main__":
    main()
