import operator
from functools import reduce

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def get_activation(act, inplace=True):
	if act == 'lrelu':
		return nn.LeakyReLU(0.01, inplace=inplace)
	elif act == 'relu':
		return nn.ReLU(inplace=inplace)
	raise Exception('unsupported activation function')

class FCEncoder(nn.Module):
	def __init__(self, dim, num_layers=3, hidden_dim=256, low_dim=2, act='lrelu'):
		super(FCEncoder, self).__init__()
		self.dim = dim
		self.num_layers = num_layers
		self.act = partial(get_activation, act=act)
		layers = [
			(nn.Linear(dim, hidden_dim*2)),
			self.act(),
			(nn.Linear(hidden_dim*2, hidden_dim)),
			self.act(),
		]
		layers += [
			(nn.Linear(hidden_dim, hidden_dim)),
			self.act(),
		] * num_layers
		layers += [
			(nn.Linear(hidden_dim, low_dim)),
		]
		self.net = nn.Sequential(*layers)
		
	def forward(self, X):
		return self.net(X)
