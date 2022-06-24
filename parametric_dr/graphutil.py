import numpy as np

def make_graph(P, n_epochs=-1):
	graph = P.tocoo()
	graph.sum_duplicates()
	n_vertices = graph.shape[1]

	if n_epochs <= 0:
		if graph.shape[0] <= 10000:
			n_epochs = 500
		else:
			n_epochs = 200

	graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
	graph.eliminate_zeros()
	return graph

def make_epochs_per_sample(weights, n_epochs):
	result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
	n_samples = n_epochs * (weights / weights.max())
	result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
	return result

def make_epochs_per_sample_from_P(p, n_epoch, neg_rate=5):
	graph = make_graph(p, n_epoch)
	epochs_per_sample = make_epochs_per_sample(graph.data, n_epoch)
	epochs_per_negative_sample = epochs_per_sample / neg_rate
	return graph, epochs_per_sample, epochs_per_negative_sample