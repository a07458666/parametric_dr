import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from umap.umap_ import fuzzy_simplicial_set, find_ab_params
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from scipy.sparse import save_npz, load_npz
import random
import timeit

from .network import FCEncoder
from .graphutil import make_epochs_per_sample_from_P
from tqdm import tqdm
MIN_DIST=0.1
SPREAD=1.0
EPS = 1e-12
D_GRAD_CLIP = 1e14

class UMAP_NN():
	def __init__(self, device, n_epochs, num_layers=3 ,hidden_dim=256, n_components=2, verbose=True, batch_size=4096, neg_rate=5):
		self.device = device
		self.n_epochs = n_epochs
		self.neg_rate = neg_rate
		self.batch_size = batch_size * neg_rate
		self.verbose = verbose
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.n_components = n_components
	
	def fit(self, data):
		encoder = FCEncoder(data.shape[1], num_layers=self.num_layers, hidden_dim=self.hidden_dim, low_dim=self.n_components)
		batch_size = self.batch_size
		
		device = self.device
		print('Device:', device)
		
		ua, ub = find_ab_params(SPREAD, MIN_DIST)
		print('a:', ua, 'b:', ub)
		
		print('calc V')
		V_csc = fuzzy_simplicial_set(data, n_neighbors=15, random_state=np.random.RandomState(42), metric='euclidean')[0]
		
		print('Make Graph')
		graph, epochs_per_sample, epochs_per_negative_sample = make_epochs_per_sample_from_P(V_csc, self.n_epochs, self.neg_rate)
		epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
		epoch_of_next_sample = epochs_per_sample.copy()
		head = graph.row
		tail = graph.col
		
		print('Trying to put X into GPU')
		X = torch.from_numpy(data).float()
		X = X.to(device)
		self.X = X
		
		init_lr = 1e-3
		encoder = encoder.to(device)
		optimizer = optim.RMSprop(encoder.parameters(), lr=init_lr, weight_decay=0)

		rnd_max_idx = X.shape[0]
		print('optimizing...')
		grad_log = []
		pbar = tqdm(range(1, self.n_epochs))
		for epoch in pbar:
			
			batch_i = []
			batch_j = []

			batch_neg_i = []
			for i in range(epochs_per_sample.shape[0]):
				if epoch_of_next_sample[i] <= epoch:
					i_idx, j_idx = head[i], tail[i]
					batch_i.append(i_idx)
					batch_j.append(j_idx)

					epoch_of_next_sample[i] += epochs_per_sample[i]

					n_neg_samples = int(
						(epoch - epoch_of_next_negative_sample[i])
						/ epochs_per_negative_sample[i]
					)
					for _ in range(n_neg_samples):
						batch_neg_i.append(i_idx)

					epoch_of_next_negative_sample[i] += (
						n_neg_samples * epochs_per_negative_sample[i]
					)
			batch_neg_j = torch.randint(0, rnd_max_idx, (len(batch_neg_i),)).tolist()
			batch_r = torch.zeros(len(batch_i), dtype=torch.long).tolist() + torch.ones(len(batch_neg_i), dtype=torch.long).tolist()


			batch_i += batch_neg_i
			batch_j += batch_neg_j

			rnd_perm = torch.randperm(len(batch_i))
			batch_i = torch.Tensor(batch_i).long()[rnd_perm]
			batch_j = torch.Tensor(batch_j).long()[rnd_perm]
			batch_r = torch.Tensor(batch_r).long()[rnd_perm]

			loss_total = []
			update_time = []
			
			
			for i in range(0, len(batch_i), batch_size):
				start_time = timeit.default_timer()
				bi = batch_i[i:i+batch_size]
				bj = batch_j[i:i+batch_size]
				br = batch_r[i:i+batch_size]

				optimizer.zero_grad()

				Y_bi = encoder(X[bi])
				Y_bj = encoder(X[bj])
				Y_bj[br==1] = Y_bj[br==1].detach()

				d = (Y_bi - Y_bj).pow(2).sum(dim=1)

				def reject_outliers(data, m=2):
					return data[(data - (data.mean())).abs() < m * (data.std())]
				def hook(grad):
					grad_clamp = grad.clamp(min=-D_GRAD_CLIP, max=D_GRAD_CLIP)
					abs_grad = grad_clamp.clone().abs()
					rgrad = reject_outliers(abs_grad)
					grad_log.append([abs_grad.max(), abs_grad.min(), abs_grad.mean(), abs_grad.std()])
					return grad_clamp
				d.register_hook(hook)
				dp = d.pow(ub)
				w = (1/(1+ua*(dp))).clamp(min=0, max=1)
				pw = w[br==0]
				rw = w[br==1]
				loss = - (torch.log(pw + EPS)).sum()
				loss += - (torch.log(1 - rw + EPS)).sum()
				loss.backward()
				loss_total.append(loss.item() / len(bi))

				torch.nn.utils.clip_grad_value_(encoder.parameters(), 4)
				optimizer.step()
				
				elapsed = timeit.default_timer() - start_time
				update_time.append(elapsed)
			
			new_lr = (1 - epoch / self.n_epochs) * init_lr
			for param_group in optimizer.param_groups:
				param_group['lr'] = new_lr
				
			if (self.verbose):
				pbar.set_description("Processing epoch %03d/%03d loss : %.5f time : %.5fs" % (epoch + 1, self.n_epochs, np.mean(loss_total), np.mean(update_time)))
		return encoder(self.X).cpu().detach().numpy()
