import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np


mode = 'audio'
layer = 'last_layer'
save_name = mode + '_' + layer
exp_name = os.path.join(mode, layer)
conf_path = 'results/' + exp_name + '/conf_matrix.csv'


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
save_path = 'conf_matrix/{}.png'.format(save_name)
plt.savefig(save_path)