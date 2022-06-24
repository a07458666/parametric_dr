# Parametric-DR
Code for Facilitate the Parametric Dimension Reduction by Gradient Clipping

Arxiv: https://arxiv.org/abs/2009.14373
![Teaser image](./img/teaser.png)

> **Abstract:** *We extend a well-known dimension reduction method, t-distributed stochastic neighbor embedding (t-SNE), from non-parametric to parametric by training neural networks. The main advantage of a parametric technique is the generalization of handling new data, which is particularly beneficial for streaming data exploration. However, training a neural network to optimize the t-SNE objective function frequently fails. Previous methods overcome this problem by pre-training and then fine-tuning the network. We found that the training failure comes from the gradient exploding problem, which occurs when data points distant in high-dimensional space are projected to nearby embedding positions. Accordingly, we applied the gradient clipping method to solve the problem. Since the networks are trained by directly optimizing the t-SNE objective function, our method achieves an embedding quality that is compatible with the non-parametric t-SNE while enjoying the ability of generalization. Due to mini-batch network training, our parametric dimension reduction method is highly efficient. We further extended other non-parametric state-of-the-art approaches, such as LargeVis and UMAP, to the parametric versions. Experiment results demonstrate the feasibility of our method. Considering its practicability, we will soon release the codes for public use.*

## Prerequisites
* pytorch (https://pytorch.org/)
* opentsne (https://opentsne.readthedocs.io/en/latest/api/index.html)
* umap (https://github.com/lmcinnes/umap)

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>
Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.7.

```
git clone https://github.com/a07458666/parametric_dr  # clone
cd parametric_dr
pip install -r requirements.txt  # install
```
</details>

<details open>
<summary>Inference</summary>
You can try it in <a href="https://github.com/a07458666/parametric_dr/blob/main/tutorial.ipynb">jupyter note</a>

### TSNE_NN
```python
import numpy as np
import torch
from parametric_dr.tsne_nn import TSNE_NN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
batch_size = 256

X = np.random.rand(512, 128)
X_embedded = TSNE_NN(device, n_epochs=epochs, batch_size=batch_size).fit(X)
```

### LARGEVIS_NN
```python
import numpy as np
import torch
from parametric_dr.largevis_nn import LARGEVIS_NN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
batch_size = 256

X = np.random.rand(512, 128)
X_embedded = LARGEVIS_NN(device, n_epochs=epochs, batch_size=batch_size).fit(X)
```

### UMAP_NN
```python
import numpy as np
import torch
from parametric_dr.umap_nn import UMAP_NN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
batch_size = 256

X = np.random.rand(512, 128)
X_embedded = UMAP_NN(device, n_epochs=epochs, batch_size=batch_size).fit(X)
```
</details>
