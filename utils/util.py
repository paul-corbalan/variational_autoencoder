import numpy as np
import torch

from matplotlib import pyplot as plt


def prod(x):
	if type(x)==list:
		return np.prod(np.array(x))
	elif type(x)==float or type(x)==int:
		return x
	else:
		return np.prod(np.array(list(x.shape)))

def format_labels_prob(y, nbr_dif_labels=10):
	n=y.shape[0]
	list_y=list(y.numpy())
	y_bis = torch.zeros((n, nbr_dif_labels))
	for i in range(n):
		j = list_y[i]
		y_bis[i][j] = 1.
	return y_bis