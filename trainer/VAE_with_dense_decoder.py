import torch
import torchvision
from torch import nn

import utils

class Trainer:
	def __init__(self, model, optimizer, loss_function, device, train_loader, test_loader, epochs = 10, print_frequency = 10):
		self.model = model
		self.optimizer = optimizer
		self.loss_function = loss_function

		self.device = device
		self.train_loader = train_loader
		self.test_loader = test_loader
		
		self.epochs = epochs
		self.print_frequency = print_frequency

	def _train_epoch(self, epoch):
		# Training
		if epoch > 0:  # test untrained net first
			codes = dict.fromkeys(["μ", "logσ2", "y", "loss", "AR"])
			self.model.train()
			means, logvars, labels, losses, accuracy_rate = list(), list(), list(), list(), list()
			train_loss = 0
			good_pred = 0
			for batch_idx, (x, y) in enumerate(self.train_loader):
				size_batch = list(x.shape)[0] if batch_idx==0 else size_batch
				x = x.to(self.device)
				y = utils.format_labels_prob(y)
				y = y.to(self.device)
				# ===================forward=====================
				y_hat, mu, logvar = self.model(x)
				loss = self.loss_function(y_hat, y, mu, logvar)
				train_loss += loss.item()
				good_pred += (torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).sum().item()
				# ===================backward====================
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				# =====================log=======================
				means.append(mu.detach())
				logvars.append(logvar.detach())
				labels.append(y.detach())
				losses.append(train_loss / ((batch_idx + 1) * size_batch))
				accuracy_rate.append(good_pred / ((batch_idx + 1) * size_batch))
				if batch_idx % self.print_frequency == 0:
					print(f'- Epoch: {epoch} [{batch_idx}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader.dataset) :.0f}%)] Average loss: {train_loss / ((batch_idx + 1) * size_batch):.4f}\t Accuracy rate: {good_pred / ((batch_idx + 1) * size_batch) :.0f}%', end="\r")
			# ===================log========================
			codes['μ'] = torch.cat(means).cpu()
			codes['logσ2'] = torch.cat(logvars).cpu()
			codes['y'] = torch.cat(labels).cpu()
			codes['loss'] = losses
			codes['AR'] = accuracy_rate
			print(f'====> Epoch: {epoch} Average loss: {train_loss / len(self.train_loader.dataset):.4f}\t Accuracy rate: {100 * good_pred / len(self.train_loader.dataset) :.0f}%')
			return codes

	def _test_epoch(self, epoch):
		# Testing
		codes = dict.fromkeys(["μ", "logσ2", "y", "loss", "AR"])
		with torch.no_grad():
			self.model.eval()
			means, logvars, labels, losses, accuracy_rate = list(), list(), list(), list(), list()
			test_loss = 0
			good_pred = 0
			for batch_idx, (x, y) in enumerate(self.test_loader):
				size_batch = list(x.shape)[0] if batch_idx==0 else size_batch
				x = x.to(self.device)
				y = utils.format_labels_prob(y)
				y = y.to(self.device)
				# ===================forward=====================
				y_hat, mu, logvar = self.model(x)
				test_loss += self.loss_function(y_hat, y, mu, logvar).item()
				good_pred += (torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).sum().item()
				# =====================log=======================
				means.append(mu.detach())
				logvars.append(logvar.detach())
				labels.append(y.detach())
				losses.append(test_loss / ((batch_idx + 1) * size_batch))
				accuracy_rate.append(good_pred / ((batch_idx + 1) * size_batch))
		# ===================log========================
		codes['μ'] = torch.cat(means).cpu()
		codes['logσ2'] = torch.cat(logvars).cpu()
		codes['y'] = torch.cat(labels).cpu()
		codes['loss'] = losses
		codes['AR'] = accuracy_rate
		test_loss /= len(self.test_loader.dataset)
		print(f'====> Test set loss: {test_loss:.4f}\t Accuracy rate: {100. * good_pred / len(self.train_loader.dataset) :.0f}%')
		return codes

	def train(self):
		codes_train = dict(μ=list(), logσ2=list(), y=list(), loss=list(), AR=list())
		codes_test = dict(μ=list(), logσ2=list(), y=list(), loss=list(), AR=list())
		for epoch in range(self.epochs + 1):
			codes = self._train_epoch(epoch) if epoch>0 else dict.fromkeys(["μ", "logσ2", "y", "loss", "AR"])
			for key in codes_train:
				codes_train[key].append(codes[key])
			codes = self._test_epoch(epoch)
			for key in codes_test:
				codes_test[key].append(codes[key])
			if epoch != self.epochs:
				print("---")
		return codes_train, codes_test
