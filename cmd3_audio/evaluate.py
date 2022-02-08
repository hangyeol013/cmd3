import hydra
from omegaconf import DictConfig
import os
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch

from data.datamodule import CMD3DataModule
from models.model import CMD3Audio
from utils.metrics import (
    get_class_accuracy,
    get_class_precision,
    get_class_recall,
    get_confusion_matrix,
    get_global_accuracy,
    get_global_precision,
    get_global_recall,
)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    model = CMD3Audio.load_from_checkpoint(
        cfg.full_checkpoint,
        model_name=cfg.model.model_name,
        model_path=None,
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

    trainer = Trainer(
        gpus=cfg.compnode.num_gpus,
        num_nodes=cfg.compnode.num_nodes,
        accelerator=cfg.compnode.accelerator,
        logger=False,
        precision=16,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    trainer.test(model, data_module)

    preds, targets = [], []
    for output in model.test_outputs:
        preds.append(output["preds"].cpu())
        targets.append(output["targets"].cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    global_report = {}
    global_report["global_accuracy"] = get_global_accuracy(preds, targets)
    global_report["global_precision"] = get_global_precision(preds, targets)
    global_report["global_recall"] = get_global_recall(preds, targets)
    global_report_df = pd.DataFrame(global_report, index=["global"])

    class_report = {}
    class_report["class_accuracy"] = get_class_accuracy(preds, targets)
    class_report["class_precision"] = get_class_precision(preds, targets)
    class_report["class_recall"] = get_class_recall(preds, targets)
    class_report_df = pd.DataFrame(class_report)

    confusion_matrix = get_confusion_matrix(preds, targets)
    confusion_matrix_df = pd.DataFrame(confusion_matrix)

    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)

    global_report_df.to_csv(os.path.join(cfg.result_dir, "global_report.csv"))
    class_report_df.to_csv(os.path.join(cfg.result_dir, "class_report.csv"))
    confusion_matrix_df.to_csv(os.path.join(cfg.result_dir, "conf_matrix.csv"))


if __name__ == "__main__":
    main()




