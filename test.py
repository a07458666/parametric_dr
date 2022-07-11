import os
import numpy as np
import matplotlib.pyplot as plt
import torch 

from scipy import spatial
from sklearn.neighbors import KNeighborsClassifier
from parametric_dr.tsne_nn import TSNE_NN
from parametric_dr.evaluation import metric_continuity, metric_trustworthiness, metric_neighborhood_hit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 200
batch_size = 256
data_size = 5000
verbose = True
name = 'sigmoid'
np.random.seed(0)
torch.manual_seed(0)

# 資料路徑
dataset_path = './dataset/mnist_60000.npz'
dataset_name = os.path.basename(dataset_path)
dataset = np.load(dataset_path)
perm = np.random.permutation(len(dataset['data']))


# 分割資料
X_train = dataset['data'][perm][:data_size]
y_train = dataset['target'][perm][:data_size]
X_test = dataset['data'][perm][data_size:data_size * 2]
y_test = dataset['target'][perm][data_size:data_size * 2]

# 設定參數 (perplexity for TSNE, n_neighbors and min_dist for UMAP)
tsne_nn = TSNE_NN(device, n_epochs=epochs, verbose = verbose, batch_size=batch_size)

# Train
X_embedded = tsne_nn.fit(X_train)
X_embedded_test = tsne_nn.fit_val(X_test)

# high dimension data
if not os.path.exists("./results/" + dataset_name + "_D_high.npy"):
    D_high_list = spatial.distance.pdist(X_test, 'euclidean')
    D_high_matrix = spatial.distance.squareform(D_high_list)
    np.save("./results/" + dataset_name + "_D_high.npy", D_high_matrix)
else:
    D_high_matrix = np.load("./results/" + dataset_name + "_D_high.npy")

# low dimension data"
D_low_list = spatial.distance.pdist(X_embedded_test, 'euclidean')
D_low_matrix = spatial.distance.squareform(D_low_list)

print("----------Evaluation----------")
# Continuity
continuity = metric_continuity(D_high_matrix, D_low_matrix, k=7)
print("Continuity = " + str(continuity))
# Trustworthiness
trustworthiness = metric_trustworthiness(D_high_matrix, D_low_matrix, k=7)
print("Trustworthiness = " + str(trustworthiness))
# Neighborhood Hit
neighborhood_hit = metric_neighborhood_hit(X_embedded_test, y_test, k=7)
print("Neighborhood Hit = " + str(neighborhood_hit))
# Average
average = (continuity + trustworthiness + neighborhood_hit) / 3
print("Average = " + str(average))

# Project PNG
x_axis = X_embedded_test[:,0]
y_axis = X_embedded_test[:,1]
c = y_test

plt.figure(figsize=(12, 12))
plt.subplot(221) 
plt.xlim(0,1)
plt.ylim(0,1)
if len(set(c.tolist())) == 10:
    plt.scatter(x_axis, y_axis, c=c, alpha=0.3, cmap='tab10')
        
elif len(set(c.tolist())) == 20:
    plt.scatter(x_axis, y_axis, c=c, alpha=0.3, cmap='tab20')
            
elif len(set(c.tolist())) <= 5:
    cdict = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red', 4: 'tab:purple'}
            
    for color in np.unique(c):
        ix = np.where(c == color)
        plt.scatter(x_axis[ix], y_axis[ix], c=cdict[color], alpha=0.3)
else:
    plt.scatter(x_axis, y_axis, c=c, alpha=0.3)

plt.subplot(222) 
if (len(tsne_nn.max_grads) > 0):
    print("max grad len = ", len(tsne_nn.max_grads))
    print("max grad MAX = ", max(tsne_nn.max_grads))
    max_grads = np.nan_to_num(tsne_nn.max_grads)
    plt.plot(max_grads)

plt.subplot(223) 
if (len(tsne_nn.epoch_losses) > 0):
    epoch_losses = np.nan_to_num(tsne_nn.epoch_losses)
    plt.plot(epoch_losses)

plt.subplot(224) 
table_data=[
    ["Continuity", round(continuity,3)],
    ["Trustworthiness", round(trustworthiness,3)],
    ["Neighborhood Hit", round(neighborhood_hit,3)],
    ["Average", round(average,3)]
]
plt.axis('off')
plt.axis('tight')
the_table = plt.table(cellText=table_data, loc='best')
the_table.auto_set_font_size(False)
the_table.set_fontsize(18)
the_table.scale(1, 2)

plt.savefig('./results/' + name + '.png')