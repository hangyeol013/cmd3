import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'results/multimodal/pred_fusion_0.9/file_reports.csv'
save_path = path.replace('file_reports.csv', 'conf_matrix_file.png')

with open(path) as f:
        lines = f.read().splitlines()

conf_matrix = np.zeros([3,3])

lines = lines[1:]
for line in lines:
    pred = int(line.split('preds_f')[1][3])
    target = int(line.split('target')[1][3])
    conf_matrix[target][pred] += 1

cf_matrix = np.array(conf_matrix).astype(int)
cf_matrix = cf_matrix/cf_matrix.sum(axis=1, keepdims=True)

plt.figure(figsize = (9,9))
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.2f', annot_kws={"size": 13})

ax.set_xlabel('\nPredicted Values', size=17)
ax.set_ylabel('Actual Values ', size=17)

ax.xaxis.set_ticklabels(['Stanley Kubrick','David Fincher', 'Joel Coen'], size=15)
ax.yaxis.set_ticklabels(['Stanley Kubrick','David Fincher', 'Joel Coen'], size=15)

plt.savefig(save_path)