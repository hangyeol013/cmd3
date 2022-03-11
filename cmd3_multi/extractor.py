import pickle

from omegaconf import DictConfig
import numpy as np
import torch
import os


def main():

    vfeature_path = 'custom_features/video/last_layer_test/features.pk'
    afeature_path = 'custom_features/audio/last_layer_test/features.pk'
    v_features = np.load(vfeature_path, allow_pickle=True)
    a_features = np.load(afeature_path, allow_pickle=True)
    concat_features = torch.cat((v_features, a_features), 1)

    feature_directory = 'custom_features/multimodal/feature_fusion'
    feature_path = 'custom_features/multimodal/feature_fusion/features.pk'
    
    if not os.path.exists(feature_directory):
        os.makedirs(feature_directory)

    with open(feature_path, "wb") as f:
        pickle.dump(concat_features, f)


if __name__ == "__main__":
    main()
