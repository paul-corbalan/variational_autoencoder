import torch
import torchvision
from torch import nn

import utils


# Defining the model


class Model(nn.Module):
	def __init__(self, d=20, size_input=[28,28], encoder=None, decoder=None):
		super().__init__()

		self.d = d
		self.size_input = size_input
		self.flatten_size_input = utils.prod(self.size_input)
		global flatten_size_input
		flatten_size_input = self.flatten_size_input
		if encoder==None or decoder==None:
			self.encoder = nn.Sequential(
				nn.Linear(self.flatten_size_input, d ** 2),
				nn.ReLU(),
				nn.Linear(d ** 2, d * 2)
			)

			self.decoder = nn.Sequential(
				nn.Linear(d, d ** 2),
				nn.ReLU(),
				nn.Linear(d ** 2, self.flatten_size_input),
				nn.Sigmoid(),
			)
		else:
			self.encoder, self.decoder = encoder, decoder

	def reparameterise(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = std.data.new(std.size()).normal_()
			return eps.mul(std).add_(mu)
		else:
			return mu

	def encode(self,x):
		return self.encoder(x.view(-1, self.flatten_size_input)).view(-1, 2, self.d)

	def decode(self,z):
		return self.decoder(z)

	def forward(self, x):
		mu_logvar = self.encode(x)
		mu = mu_logvar[:, 0, :]
		logvar = mu_logvar[:, 1, :]
		z = self.reparameterise(mu, logvar)
		return self.decode(z), mu, logvar


def optimizer(model, optim=torch.optim.Adam, learning_rate=1e-3):
	return optim(model.parameters(),lr=learning_rate,)

def loss_function(f=nn.functional.binary_cross_entropy, β=1):
	def loss(x_hat, x, mu, logvar):
		Data_Error = f(x_hat, x.view(-1, flatten_size_input), reduction='sum')
		KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))
		return Data_Error + β * KLD
	return loss