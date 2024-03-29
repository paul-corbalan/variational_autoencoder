from matplotlib import pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display


def set_default(figsize=(10, 10), dpi=100):
	plt.style.use(['dark_background', 'bmh'])
	plt.rc('axes', facecolor='k')
	plt.rc('figure', facecolor='k')
	plt.rc('figure', figsize=figsize, dpi=dpi)


def display_images(in_, out, n=1, label=None, count=False):
	for N in range(n):
		if in_ is not None:
			in_pic = in_.data.cpu().view(-1, 28, 28)
			plt.figure(figsize=(18, 4))
			plt.suptitle(label + ' – real test data / reconstructions', color='w', fontsize=16)
			for i in range(4):
				plt.subplot(1,4,i+1)
				plt.imshow(in_pic[i+4*N])
				plt.axis('off')
		out_pic = out.data.cpu().view(-1, 28, 28)
		plt.figure(figsize=(18, 6))
		for i in range(4):
			plt.subplot(1,4,i+1)
			plt.imshow(out_pic[i+4*N])
			plt.axis('off')
			if count: plt.title(str(4 * N + i), color='w')