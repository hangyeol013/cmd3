import numpy as np
import torch
import torchmetrics.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns


def save_conf_matrix(conf_path, save_path):

    with open(conf_path) as f:
        lines = f.read().splitlines()

    cf_matrix = []
    for line in lines:
        cf_matrix.append(line.split(',')[1:])    

    cf_matrix = cf_matrix[1:]
    cf_matrix = np.array(cf_matrix).astype(int)

    plt.figure(figsize = (9,9))
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='d')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Stanley Kubrick','David Fincher', 'Joel Coen'])
    ax.yaxis.set_ticklabels(['Stanley Kubrick','David Fincher', 'Joel Coen'])


    ## Display the visualization of the Confusion Matrix.
    plt.savefig(save_path)


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
