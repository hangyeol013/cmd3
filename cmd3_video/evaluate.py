import hydra
from omegaconf import DictConfig
import os
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch

from data.datamodule import CMD3DataModule
from models.model import CMD3Video
from utils.metrics import (
    get_class_accuracy,
    get_class_precision,
    get_class_recall,
    get_confusion_matrix,
    get_global_accuracy,
    get_global_precision,
    get_global_recall,
    save_conf_matrix,
)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    model = CMD3Video.load_from_checkpoint(
        cfg.full_checkpoint,
        model_name=cfg.model.model_name,
        model_depth=cfg.model.model_depth,
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

    trainer = Trainer(
        gpus=cfg.compnode.num_gpus,
        num_nodes=cfg.compnode.num_nodes,
        accelerator=cfg.compnode.accelerator,
        logger=False,
        precision=16,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    trainer.test(model, data_module)

    preds, preds_label, targets, ids, time_frame = [], [], [], [], []
    for i, output in enumerate(model.test_outputs):
        preds.append(output["preds"].cpu())
        preds_label.append(np.argmax(output["preds"].cpu(), axis=1))
        targets.append(output["targets"].cpu())
        ids.extend(output['id'])
        
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    preds_label = torch.cat(preds_label)
    ids = np.hstack(ids)

    file_results = {}
    vidio_id_set = list(set(ids))

    for clip_id in vidio_id_set:
        file_id = clip_id.split('/clip')[0]
        file_results[file_id] = {'preds': [0, 0, 0], 'preds_f': 0, 'target': 0}
        for video_clip_id, out_preds, out_targets in zip(ids, preds_label, targets):
            video_id = video_clip_id.split('/clip')[0]
            if video_id == file_id:
                if out_preds == 0:
                    file_results[file_id]['preds'][0] += 1
                elif out_preds == 1:
                    file_results[file_id]['preds'][1] += 1
                elif out_preds == 2:
                    file_results[file_id]['preds'][2] += 1
                max_val = max(file_results[file_id]['preds'])
                file_results[file_id]['preds_f'] = file_results[file_id]['preds'].index(max_val)
                # file_results[file_id]['preds_f'] = max(file_results[file_id]['preds'], key=file_results[file_id]['preds'].get)
                file_results[file_id]['target'] = out_targets.item()

    total = 0
    correct = 0
    for i in range(len(file_results.values())):
        file_pred = int(list(file_results.values())[i]['preds_f'])
        file_target = int(list(file_results.values())[i]['target'])
        total += 1
        if file_pred == file_target:
            correct += 1
    file_accuracy = correct / total
    print(file_accuracy)

    global_report = {}
    global_report["global_accuracy"] = get_global_accuracy(preds, targets)
    global_report['file_accuracy'] = file_accuracy
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

    file_report = {}
    file_report['file_path'] = file_results.keys()
    file_report['results'] = file_results.values()
    file_report_df = pd.DataFrame(file_report)

    if not os.path.exists(cfg.result_dir):
        os.makedirs(cfg.result_dir)

    global_report_df.to_csv(os.path.join(cfg.result_dir, "global_report.csv"))
    class_report_df.to_csv(os.path.join(cfg.result_dir, "class_report.csv"))
    confusion_matrix_df.to_csv(os.path.join(cfg.result_dir, "conf_matrix.csv"))
    results_report_df.to_csv(os.path.join(cfg.result_dir, "result_reports.csv"))
    file_report_df.to_csv(os.path.join(cfg.result_dir, "file_reports.csv"))
    save_conf_matrix(os.path.join(cfg.result_dir, "conf_matrix.csv"), os.path.join(cfg.result_dir, "conf_matrix_img.png"))


if __name__ == "__main__":
    main()




