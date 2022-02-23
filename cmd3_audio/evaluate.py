import hydra
from omegaconf import DictConfig
import os
import numpy as np
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

    preds, preds_label, targets, ids = [], [], [], []
    for output in model.test_outputs:
        preds.extend(output["preds"].cpu())
        preds_label.extend(np.argmax(output["preds"].cpu(), axis=1))
        targets.extend(output["targets"].cpu())
        ids.extend(output['id'])

    # pred_files = []
    # id_files = []
    # target_files = []

    # for i in list(set(ids)):
    #     pred_file = 0
    #     target_file = 0
    #     for j in range(len(ids)):
    #         if ids[j] == i:
    #             pred_file += preds[j]
    #             target_file += targets[j]
        
    #     id_files.append(i)
    #     target_files.append(target_file)
    #     pred_files.append(np.argmax(pred_file, axis=0).item())

    # target_files = list(set(target_files))

    # print(pred_files)
    # print(target_files)
    # print(len(pred_files))
    # print(len(target_files))

    # assert False
        
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    preds_label = torch.cat(preds_label)
    ids = np.hstack(ids)

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

    results_report = {}
    results_report["file_path"] = ids
    results_report["targets"] = targets
    results_report["preds"] = preds_label
    results_report_df = pd.DataFrame(results_report)

    # results_file = {}
    # results_file['file_path'] = id_files
    # results_file['targets'] = target_files
    # results_file['preds'] = pred_files

    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)

    global_report_df.to_csv(os.path.join(cfg.result_dir, "global_report.csv"))
    class_report_df.to_csv(os.path.join(cfg.result_dir, "class_report.csv"))
    confusion_matrix_df.to_csv(os.path.join(cfg.result_dir, "conf_matrix.csv"))
    results_report_df.to_csv(os.path.join(cfg.result_dir, "result_reports.csv"))


if __name__ == "__main__":
    main()




