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
	elif act == 'sigmoid':
		return nn.Sigmoid()
	raise Exception('unsupported activation function')

class FCEncoder(nn.Module):
	def __init__(self, dim, low_dim=2, act='sigmoid'):
		super(FCEncoder, self).__init__()
		self.dim = dim
		self.act = partial(get_activation, act=act)
        
		# new network (1024 -> 512 -> 256 -> 128 -> 2)
		layers = [
			(nn.Linear(dim, 1024)),
			# (nn.BatchNorm1d(1024)),
			self.act(),
			(nn.Linear(1024, 512)),
			# (nn.BatchNorm1d(512)),
			self.act(),
			(nn.Linear(512, 256)),
			# (nn.BatchNorm1d(256)),
			self.act(),
			(nn.Linear(256, 128)),
			# (nn.BatchNorm1d(128)),
			self.act(),
			(nn.Linear(128, low_dim))
		]
        
		self.net = nn.Sequential(*layers)
		
	def forward(self, X):
		return self.net(X)
