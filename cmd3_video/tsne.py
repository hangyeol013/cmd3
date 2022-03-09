import argparse
import torch
import random
import numpy as np
import pandas as pd
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



def fix_random_seeds():
    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def color_movie(labels):
    colors_per_movie = {}

    for movie_name in labels:
        colors_per_movie[movie_name] = list(np.random.randint(256,size=3))
    print(colors_per_movie)
    return colors_per_movie


# colors_per_class = {
#     'Stanley Kubrick' : [177, 106, 57],
#     'David Fincher' : [31, 239, 252],
#     'Joel Coen' : [96, 94, 211]
# }

colors_per_class = {
    0 : [177, 106, 57],
    1 : [31, 239, 252],
    2 : [96, 94, 211]
}


colors_per_color = {
    'Black&White': [150, 150, 150],
    'Color': [177, 106, 57]
}

colors_per_year = {
    '1957': [52, 57, 57],
    '1964': [57, 91, 90],
    '1992': [58, 126, 126],
    '1994': [56, 163, 164],
    '2001': [52, 57, 57],
    # '2001': [49, 200, 207],
    '2008': [31, 239, 252]
}

# colors_per_year = {
#     '1957': [52, 57, 57],
#     '1964': [52, 57, 57],
#     '1992': [52, 57, 57],
#     '1994': [52, 57, 57],
#     '2001': [31, 239, 252],
#     '2008': [31, 239, 252]
# }


colors_per_movieName = {
    'Paths of Glory': [200, 70, 57],
    'Dr. Strangelove': [160, 70, 57],
    'Alien 3': [91, 190, 91],
    'The Curious Case of Benjamin Button': [91, 130, 91],
    "The Man Who Wasn't There": [96, 94, 250],
    'The Hudsucker Proxy': [96, 94, 170]
}

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)

    return starts_from_zero / value_range


def visualize_tsne_points(tx, ty, labels, mode):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if mode == 'file_names':
        colors_movies = color_movie(labels)
        for label in colors_movies:
            indices = [i for i, l in enumerate(labels) if l == label]
            print(label, ': ', len(indices))

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            color = np.array([colors_movies[label][::-1]], dtype=np.float) / 255
            ax.scatter(current_tx, current_ty, c=color, label=label)

    if mode == 'movie_names':
        for label in colors_per_movieName:
            indices = [i for i, l in enumerate(labels) if l == label]
            print(label, ': ', len(indices))

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            color = np.array([colors_per_movieName[label][::-1]], dtype=np.float) / 255
            ax.scatter(current_tx, current_ty, c=color, label=label)

    elif mode == 'years':
        for label in colors_per_year:
            indices = [i for i, l in enumerate(labels) if l == int(label)]
            print(label, ': ', len(indices))

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            color = np.array([colors_per_year[label][::-1]], dtype=np.float) / 255
            ax.scatter(current_tx, current_ty, c=color, label=label)

    elif mode  == 'color_type':
        for label in colors_per_color:
            indices = [i for i, l in enumerate(labels) if l == label]
            print(label, ': ', len(indices))

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            color = np.array([colors_per_color[label][::-1]], dtype=np.float) / 255
            ax.scatter(current_tx, current_ty, c=color, label=label)

    else:
        for label in colors_per_class:
            indices = [i for i, l in enumerate(labels) if l == label]
            print(label, ': ', len(indices))

            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255
            ax.scatter(current_tx, current_ty, c=color, label=label)

    ax.legend(loc='best', prop={'size': 20})

    plt.show()

def visualize_tsne(tsne, labels, mode, plot_size=1000, max_image_size=100):
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    visualize_tsne_points(tx, ty, labels, mode)



def visualize_tsne_points_both(tx, ty, gt_labels, pd_labels, file_name):
    fig = plt.figure(figsize=(8,6), dpi=1200)
    ax = fig.add_subplot(111)

    for label in colors_per_class:
        gt_indices = [i for i, l in enumerate(gt_labels) if l == label]
        pd_indices = [i for i, l in enumerate(pd_labels) if l == label]
        print(label, '(groundTruth): ', len(gt_indices), '|', '(predictions): ', len(pd_indices))
        # print(label, '(predictions): ', len(pd_indices))
        color_correct = np.array([colors_per_class[label][::-1]], dtype=float) / 255

        gt_tx = np.take(tx, gt_indices)
        gt_ty = np.take(ty, gt_indices)
        pd_tx = np.take(tx, pd_indices)
        pd_ty = np.take(ty, pd_indices)

        ax.scatter(gt_tx, gt_ty, c=color_correct, label='{}_gt'.format(label), marker = 'o', alpha=0.3)
        ax.scatter(pd_tx, pd_ty, c=color_correct, label='{}_pd'.format(label), marker = 'x', alpha=0.3)

    # plt.show()
    plt.savefig(file_name)


def visualize_both(tsne, gt_labels, pd_labels, file_name, plot_size=1000, max_image_size=100):
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    visualize_tsne_points_both(tx, ty, gt_labels, pd_labels, file_name)


def tsne_vis(feature_path, result_path, tsne_mode, save_path):

    feature = np.load('{}/features.pk'.format(feature_path), allow_pickle=True)

    # cmd3_results = pd.read_csv('results/cmd3_0222_All_e20/result_reports.csv')
    cmd3_results = pd.read_csv(result_path)

    gt_labels = []
    pd_labels = []
    file_names = []
    movie_names = []
    years = []
    colors = []

    for i in range(len(feature)):
        video_index = i
        gt_label = cmd3_results.iloc[video_index].targets.item()
        pd_label = cmd3_results.iloc[video_index].preds.item()
        gt_labels.append(gt_label)
        pd_labels.append(pd_label)

    features_tsne = feature

    print('features_shape: ', features_tsne.shape)
    # t-SNE part
    tsne = TSNE(n_components=2).fit_transform(features_tsne)
    print('tsne_shape: ', tsne.shape)
    if tsne_mode == 'gt_labels':
        visualize_tsne(tsne, gt_labels, tsne_mode)
    elif tsne_mode == 'pd_labels':
        visualize_tsne(tsne, pd_labels, tsne_mode)
    elif tsne_mode == 'file_names':
        visualize_tsne(tsne, file_names, tsne_mode)
    elif tsne_mode == 'both_labels':
        visualize_both(tsne, gt_labels, pd_labels, save_path)
    elif tsne_mode == 'movie_names':
        visualize_tsne(tsne, movie_names, tsne_mode)
    elif tsne_mode == 'years':
        visualize_tsne(tsne, years, tsne_mode)
    elif tsne_mode == 'color_type':
        visualize_tsne(tsne, colors, tsne_mode)


def main():

    mode = 'multimodal'
    tsne_mode = 'both_labels'

    if mode == 'video':
        xp_name = 'cmd3_video_last_0305'
        feature_path = 'custom_features/video/{}'.format(xp_name)
        result_path = 'results/video/{}/result_reports.csv'.format(xp_name)
        save_path = 't-sne/{}'.format(xp_name)
    elif mode == 'audio':
        xp_name = 'cmd3_audio_last_0305'
        feature_path = 'custom_features/audio/{}'.format(xp_name)
        result_path = 'results/audio/{}/result_reports.csv'.format(xp_name)
        save_path = 't-sne/{}'.format(xp_name)
    elif mode == 'multimodal':
        xp_name = 'pred_fusion'
        feature_path = 'custom_features/multimodal/{}'.format(xp_name)
        result_path = 'results/multimodal/pred_fusion_0.5/result_reports.csv'
        save_path = 't-sne/{}'.format(xp_name)

    fix_random_seeds()
    tsne_vis(feature_path, result_path, tsne_mode, save_path)

if __name__ == '__main__':
    main()
