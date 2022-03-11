import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
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
from torch.utils import data
from numpy.linalg import norm


NUM_CLASSES = 3


def feature_fuse_eval(pretrained_path, vfeature_path, afeature_path, target_path, id_path, feature_fusion_mode):

    v_features = np.load(vfeature_path, allow_pickle=True)
    a_features = np.load(afeature_path, allow_pickle=True)

    if feature_fusion_mode == 'norm':
        v_features = torch.from_numpy(np.expand_dims(norm(v_features, ord=2, axis=1), axis=1))
        a_features = torch.from_numpy(np.expand_dims(norm(a_features, ord=2, axis=1), axis=1))

    concat_features = torch.cat((v_features, a_features), 1)

    targets = np.load(target_path, allow_pickle=True)
    ids = np.load(id_path, allow_pickle=True)

    print(concat_features.shape)

    model = nn.Linear(concat_features.shape[1], NUM_CLASSES).cuda()
    pretrained_params = torch.load(pretrained_path)
    model.load_state_dict(pretrained_params)

    loader = data.DataLoader(data.TensorDataset(concat_features, targets), batch_size=16)
    model.eval()

    preds = []
    preds_label = []

    for batch in loader:
        inputs = batch[0].float()
        inputs = inputs.cuda()
        target = batch[1].cuda()

        with torch.no_grad():
            pred = model(inputs)
        preds.append(pred.cpu())
        preds_label.append(np.argmax(pred.cpu(), axis=1))

    preds = torch.cat(preds)
    preds_label = torch.cat(preds_label)
    
    if feature_fusion_mode == 'norm':
        save_path = 'results/multimodal/feature_fusion_norm'
    else:
        save_path = 'results/multimodal/feature_fusion'
    
    save_results(preds, preds_label, targets, ids, save_path)



def pred_fuse(vPred_path, aPred_path, targets_path, ids_path, alpha):

    v_preds = np.load(vPred_path, allow_pickle=True)
    a_preds = np.load(aPred_path, allow_pickle=True)

    targets = np.load(targets_path, allow_pickle=True)
    ids = np.load(ids_path, allow_pickle=True)
    
    preds_fusion = alpha * v_preds + (1-alpha) * a_preds
    preds_label = np.argmax(preds_fusion.cpu(), axis=1)

    save_path = 'results/multimodal/pred_fusion_{}'.format(alpha)

    save_results(preds_fusion, preds_label, targets, ids, save_path)



def save_results(preds, preds_label, targets, ids, save_path):

    file_results = {}
    id_set = list(set(ids))

    for file_id in id_set:
        file_results[file_id] = {'preds': [0, 0, 0], 'preds_f': 0, 'target': 0}
        for audio_id, out_preds, out_targets in zip(ids, preds_label, targets):
            if audio_id == file_id:
                if out_preds == 0:
                    file_results[file_id]['preds'][0] += 1
                elif out_preds == 1:
                    file_results[file_id]['preds'][1] += 1
                elif out_preds == 2:
                    file_results[file_id]['preds'][2] += 1
                max_val = max(file_results[file_id]['preds'])
                file_results[file_id]['preds_f'] = file_results[file_id]['preds'].index(max_val)
                file_results[file_id]['target'] = int(out_targets.item())

    total = 0
    correct = 0
    for i in range(len(file_results.values())):
        file_pred = int(list(file_results.values())[i]['preds_f'])
        file_target = int(list(file_results.values())[i]['target'])
        total += 1
        if file_pred == file_target:
            correct += 1
    file_accuracy = correct / total

    global_report = {}
    global_report["global_accuracy"] = get_global_accuracy(preds, targets)
    global_report['file_accuracy'] = file_accuracy
    global_report["global_precision"] = get_global_precision(preds, targets)
    global_report["global_recall"] = get_global_recall(preds, targets)
    global_report_df = pd.DataFrame(global_report, index=["global_clip"])

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


    result_dir = save_path

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    global_report_df.to_csv(os.path.join(result_dir, "global_report.csv"))
    class_report_df.to_csv(os.path.join(result_dir, "class_report.csv"))
    confusion_matrix_df.to_csv(os.path.join(result_dir, "conf_matrix.csv"))
    results_report_df.to_csv(os.path.join(result_dir, "result_reports.csv"))
    file_report_df.to_csv(os.path.join(result_dir, "file_reports.csv"))
    save_conf_matrix(os.path.join(result_dir, "conf_matrix.csv"), os.path.join(result_dir, "conf_matrix_img.png"))


if __name__ == '__main__':
    

    # Evaluation
    feature_fusion_mode = 'norm'

    if feature_fusion_mode == 'norm':
        pretrained_path = 'cmd3_multi/feature_fusion_norm.pth'
    else:
        pretrained_path = 'cmd3_multi/feature_fusion.pth'

    vfeatures_eval_path = 'custom_features/video/last_layer_test/features.pk'
    afeatures_eval_path = 'custom_features/audio/last_layer_test/features.pk'
    targets_eval_path = 'custom_features/audio/last_layer_test/targets.pk'
    ids_eval_path = 'custom_features/audio/last_layer_test/ids.pk'
    
    vpreds_path = 'custom_features/video/last_layer_test/preds.pk'
    apreds_path = 'custom_features/audio/last_layer_test/preds.pk'

    # alpha = 0.5
    feature_fuse_eval(pretrained_path, vfeatures_eval_path, afeatures_eval_path, targets_eval_path, ids_eval_path, feature_fusion_mode)
    # pred_fuse(vpreds_path, apreds_path, targets_eval_path, ids_eval_path, alpha)
    # pred_fuse(vpreds_path, apreds_path, targets_eval_path, ids_eval_path, 0.1)
    # pred_fuse(vpreds_path, apreds_path, targets_eval_path, ids_eval_path, 0.3)
    # pred_fuse(vpreds_path, apreds_path, targets_eval_path, ids_eval_path, 0.5)
    # pred_fuse(vpreds_path, apreds_path, targets_eval_path, ids_eval_path, 0.7)
    # pred_fuse(vpreds_path, apreds_path, targets_eval_path, ids_eval_path, 0.9)