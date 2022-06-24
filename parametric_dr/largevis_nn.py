import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from openTSNE import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from scipy.sparse import save_npz, load_npz
import random
import timeit

from tqdm import tqdm
from .network import FCEncoder
from .graphutil import make_epochs_per_sample_from_P

MIN_DIST=0.1
SPREAD=1.0
EPS = 1e-12
D_GRAD_CLIP = 1e14

class LARGEVIS_NN():
	def __init__(self, device, n_epochs, num_layers=3 ,hidden_dim=256, n_components=2, verbose=True, batch_size=256, neg_rate=5, gamma=7):
		self.device = device
		self.n_epochs = n_epochs
		self.neg_rate = neg_rate
		self.batch_size = batch_size * neg_rate
		self.gamma = gamma
		self.verbose = verbose
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.n_components = n_components
	
	def fit(self, data):
		encoder = FCEncoder(data.shape[1], num_layers=self.num_layers, hidden_dim=self.hidden_dim, low_dim=self.n_components)
		batch_size = self.batch_size
		device = self.device
		
		print('calc P')
		pre_embedding = TSNE(perplexity=15).prepare_initial(data)
		P_csc = pre_embedding.affinities.P
		
		print('Make Graph')
		graph, epochs_per_sample, epochs_per_negative_sample = make_epochs_per_sample_from_P(P_csc, self.n_epochs, self.neg_rate)
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
		optimizer = optim.RMSprop(encoder.parameters(), lr=init_lr)
		lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-7)

		rnd_max_idx = X.shape[0]
		gamma = self.gamma

		print('optimizing...')
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
					
					for _ in range(self.neg_rate):
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

				X_i = X[bi]
				X_j = X[bj]
				Y_bi = encoder(X_i)
				Y_bj = encoder(X_j)
				Y_bj[br==1] = Y_bj[br==1].detach()

				d = (Y_bi - Y_bj).pow(2).sum(dim=1)
                #andy
				#d.register_hook(lambda grad: grad.clamp(min=-D_GRAD_CLIP, max=D_GRAD_CLIP))
				if d.requires_grad:
					def hook(grad):
						return grad
				d.register_hook(hook)
                
				w = (1/(1+d)).clamp(min=0, max=1)

				pw = w[br==0]
				rw = w[br==1]
				loss = - (torch.log(pw + EPS)).sum()
				loss += - (gamma * torch.log(1 - rw + EPS)).sum()
				loss /= len(bi)
				loss.backward()
				loss_total.append(loss.item())

				torch.nn.utils.clip_grad_value_(encoder.parameters(), 16)
				optimizer.step()
				elapsed = timeit.default_timer() - start_time
				update_time.append(elapsed)

			lr_sched.step()

			if (self.verbose):
				pbar.set_description("Processing epoch %03d/%03d loss : %.5f time : %.5fs" % (epoch + 1, self.n_epochs, np.mean(loss_total), np.mean(update_time)))
		return encoder(self.X).cpu().detach().numpy()

