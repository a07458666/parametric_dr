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
from functools import partial
import timeit
import math
from tqdm import tqdm
from .network import FCEncoder

EPS = 1e-12
D_GRAD_CLIP = 1e14

class TSNE_NN():
	def __init__(self, device, n_epochs, num_layers=3 ,hidden_dim=256, n_components=2, verbose=True, batch_size=256):
		self.device = device
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.perplexity = 15
		self.test_data = None
		self.max_clipped_grads = []
		self.max_grads = []
		self.loss_total = []
		self.verbose = verbose
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.n_components = n_components
	
	def fit(self, data):
		encoder = FCEncoder(data.shape[1], num_layers=self.num_layers, hidden_dim=self.hidden_dim, low_dim=self.n_components)
		batch_size = self.batch_size
		device = self.device
		print('perplexity:', self.perplexity)
		
		print('calc P')
		pre_embedding = TSNE(perplexity=self.perplexity).prepare_initial(data)
		P_csc = pre_embedding.affinities.P
			
		print('Trying to put X into GPU')
		X = torch.from_numpy(data).float()
		X = X.to(device)
		self.X = X

		encoder = encoder.to(device)
		init_lr = 1e-3
		optimizer = optim.RMSprop(encoder.parameters(), lr=init_lr)
		lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs * math.ceil(len(X)/batch_size), eta_min=1e-7)
		
		def neg_squared_euc_dists(X):
			D = torch.cdist(X, X, p=2)
			return -D

		def w_tsne(Y):
			distances = neg_squared_euc_dists(Y)
			inv_distances = 1. / (1. - (distances)) #1 / (1+d^2)
			return inv_distances
		
		def KLD(P, Q):
			x = P/Q
			if x.requires_grad:
				def hook(grad):
					self.max_grads.append(float(grad.abs().max().cpu().numpy()))
					return grad
				x.register_hook(hook)
			return P * torch.log(x)
		
		iteration = 0
		print('optimizing...')
		pbar = tqdm(range(self.n_epochs))
		for epoch in pbar:
			iteration += 1

			idxs = torch.randperm(len(X))
			
			self.loss_total = []
			update_time = []
			for i in range(0, len(X), batch_size):
				start_time = timeit.default_timer()
				idx = idxs[i:i+batch_size]
				_p = torch.Tensor(P_csc[idx][:, idx].toarray()).float()
				if iteration < 250:
					_p *= 4
				p = (_p+EPS).to(device)
				optimizer.zero_grad()
				y = encoder(X[idx])
				if (torch.sum(torch.isnan(y)) > 0):
					return
				w = w_tsne(y)
				q = w / torch.sum(w)
				loss = KLD(p, q).sum()
				loss.backward()
				self.loss_total.append(loss.item())
				# torch.nn.utils.clip_grad_value_(encoder.parameters(), 4)
				optimizer.step()
				elapsed = timeit.default_timer() - start_time
				update_time.append(elapsed)
			
				lr_sched.step()
			if (self.verbose):
				pbar.set_description("Processing epoch %03d/%03d loss : %.5f time : %.5fs" % (epoch + 1, self.n_epochs, np.mean(self.loss_total), np.mean(update_time)))
				# print('{:03d}/{:03d}'.format(epoch, self.n_epochs), '{:.5f}'.format(np.mean(self.loss_total)), '{:.5f}s'.format(np.mean(update_time)))
		return encoder(self.X).cpu().detach().numpy()
