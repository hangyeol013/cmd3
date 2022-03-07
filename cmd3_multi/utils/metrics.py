import numpy as np
import torch
import torchmetrics.functional as F


def get_global_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, num_classes=3
) -> float:
    global_accuracy = F.accuracy(
        preds, targets, average="macro", num_classes=num_classes
    ).item()
    return global_accuracy


def get_global_precision(
    preds: torch.Tensor, targets: torch.Tensor, num_classes=3
) -> float:
    global_precision = F.precision(
        preds, targets, average="macro", num_classes=num_classes
    ).item()
    return global_precision


def get_global_recall(
    preds: torch.Tensor, targets: torch.Tensor, num_classes=3
) -> float:
    global_recall = F.recall(
        preds, targets, average="macro", num_classes=num_classes
    ).item()
    return global_recall


def get_class_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, num_classes=3
) -> np.array:
    class_accuracy = F.accuracy(
        preds, targets, average="none", num_classes=num_classes
    ).numpy()
    return class_accuracy


def get_class_precision(
    preds: torch.Tensor, targets: torch.Tensor, num_classes=3
) -> np.array:
    class_precision = F.precision(
        preds, targets, average="none", num_classes=num_classes
    ).numpy()
    return class_precision


def get_class_recall(
    preds: torch.Tensor, targets: torch.Tensor, num_classes=3
) -> np.array:
    class_recall = F.recall(
        preds, targets, average="none", num_classes=num_classes
    ).numpy()
    return class_recall


def get_confusion_matrix(
    preds: torch.Tensor, targets: torch.Tensor, num_classes=3
) -> np.array:
    confusion_matrix = F.confusion_matrix(
        preds, targets, num_classes=num_classes
    ).numpy()
    return confusion_matrix
