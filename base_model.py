import os
import itertools
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import torch
import torch.nn as nn
import logging
import time
import datetime
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
from make_stats import plot_error, plot_comp
import utils
from lamb import Lamb
import wandb

def count_parameter(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_dict(d1, d2):
	for key, value in d2.items():
		if key in d1:
			d1[key] += value
		else:
			d1[key] = value
	return d1

def divide_dict(d, denom):
	return {k: v / denom for k, v in d.items()}

def add_prefix_to_dict(d, prefix):
	return {f'{prefix}_{k}': v for k, v in d.items()}

class BaseModel(torch.nn.Module):
	def __init__(self, in_dim, out_dim, output_dir, use_gpu=True, lr=1e-4, schedule_lr=False):
		assert out_dim == 4, 'out_dim should be 4 for top regression task'
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.output_dir = output_dir
		self.logger = self.get_logger()
		self.saved_models_dir = os.path.join(self.output_dir, 'saved_models')
		if not os.path.exists(self.saved_models_dir):
			os.makedirs(self.saved_models_dir)
		self.history = {}
		self.define_modules()
		self.logger.info(f'Using Lamb optimizer with lr={lr}')
		self.optimizer = Lamb(self.parameters(), lr=lr, weight_decay=0.01)
		self.schedule_lr = schedule_lr
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, threshold=0.01, verbose=True)
		if self.load('min_val_loss_model.pt', warn_when_fail=False):
			pass
		else:
			self.logger.info('Initializing a new model from scratch')
		self.logger.info(f'Model has {count_parameter(self) / 1e6:.2g}M parameters')
		self.log_parameters()
		self.epoch = 1

	def define_modules(self):
		self.out_mlp = Linear(self.in_dim, self.out_dim)

	def log_parameters(self):
		return

	def set_scheduler(self, scheduler):
		self.scheduler = scheduler

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def get_logger(self):
		logger = logging.getLogger('train_logger')
		if (logger.hasHandlers()):
			logger.handlers.clear()
		logger.setLevel(logging.DEBUG)
		fh = logging.FileHandler(f'{self.output_dir}/train.log')
		fh.setLevel(logging.DEBUG)
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		logger.addHandler(fh)
		logger.addHandler(ch)
		return logger

	def save(self, name):
		checkpoint_dict = {
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'history': self.history
		}
		torch.save(checkpoint_dict, os.path.join(self.saved_models_dir, name))
		self.logger.info(f'Saved model at {os.path.join(self.saved_models_dir, name)}')

	def load(self, name, warn_when_fail=True):
		checkpoint_dir = os.path.join(self.saved_models_dir, name)
		if not os.path.exists(checkpoint_dir):
			if warn_when_fail:
				print(f'Checkpoint directory does not exist: {checkpoint_dir}')
			return False
		checkpoint_dict = torch.load(checkpoint_dir)
		self.load_state_dict(checkpoint_dict['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
		self.history = checkpoint_dict['history']
		self.logger.info(f'Loaded model from {os.path.join(self.saved_models_dir, name)}')
		return True

	def forward_and_return_loss(self, data, return_y=False):
		y_target = data['y']
		y_pred = self(data['x'])
		loss, loss_info = self.loss(y_pred, y_target)
		if return_y:
			return loss, loss_info, y_target.detach(), y_pred.detach()
		return loss, loss_info

	def update_model(self, data):
		self.optimizer.zero_grad()
		loss, loss_info = self.forward_and_return_loss(data, return_y=False)
		loss.backward()
		self.optimizer.step()
		return loss_info
		
	def train_epoch(self, train_loader, save_freq=5000):
		torch.cuda.empty_cache()
		self.train()
		sum_loss_info = {}
		num_batch = 0
		y_target_list = []
		y_pred_list = []
		for data in tqdm(train_loader):
			if num_batch > 0 and num_batch % save_freq == 0:
				self.save('most_recent_model.pt')
			loss_info = self.update_model(data)
			sum_loss_info = add_dict(sum_loss_info, loss_info)
			num_batch += 1
		info = divide_dict(sum_loss_info, num_batch)
		self.epoch += 1
		return info

	def validate_model(self, loader, prefix):
		torch.cuda.empty_cache()
		self.eval()
		sum_loss_info = {}
		num_batch = 0
		y_target_list = []
		y_pred_list = []
		with torch.no_grad():
			for data in tqdm(loader):
				# here we are only evaluating the loss, not optimizing
				loss, loss_info, y_target, y_pred = self.forward_and_return_loss(data, return_y=True) 
				###### NO BACKPROP ######
				y_target_list.append(y_target)
				y_pred_list.append(y_pred)
				sum_loss_info = add_dict(sum_loss_info, loss_info)
				num_batch += 1
		info = divide_dict(sum_loss_info, num_batch)
		y_pred = torch.cat(y_pred_list, dim=0)
		y_target = torch.cat(y_target_list, dim=0)
		metrics_info = self.get_metrics(y_pred.numpy(), y_target.numpy())
		info.update(metrics_info)
		self.plot_y(prefix, y_pred, y_target)
		return info

	def get_metrics(self, y_pred, y_target):
		# compute metrics in addition to loss on the predicted 4-vectors, assuming matched and all valid(no placeholder)
		# y_*: (..., D)
		if y_target.shape[-1] != 4: # skip if target doesn't look like a 4 vetor
			return {}
		dy = y_pred - y_target
		if self.detector:
			dy[2] = np.arccos(np.cos(dy[2]))
		mae = np.mean(np.abs(dy), axis=0)
		med_abs_pull = np.median(np.abs(dy) / (np.abs(y_target) + 1e-5), axis=0)

		
		return {'mae': torch.FloatTensor(mae), 'med_abs_pull': torch.FloatTensor(med_abs_pull)}

	def update_history(self, info):
		wandb.log(info, commit=True)
		for key, value in info.items():
			if key in self.history:
				if isinstance(value, list):
					self.history[key].extend(value)
				else:
					self.history[key].append(value)
			else:
				self.history[key] = [value]

	def plot_history(self):
		for key in self.history:
			if 'train' in key and 'loss' in key:
				plt.plot(self.history[key], label=key)
				plt.legend()
				plt.savefig(os.path.join(self.output_dir, 'train_losses.png'))
		keys = set([key.replace('train_', '').replace('val_', '').replace('test_', '') for key in self.history])
		# scalar history
		for key in keys:
			for prefix in ['train', 'val', 'test']:
				if f'{prefix}_{key}' in self.history:
					hist = self.history[f'{prefix}_{key}']
					if isinstance(hist[0], torch.Tensor):
						hist = torch.stack(hist, dim=0)
					else:
						hist = torch.FloatTensor(hist)
					if len(hist.shape) == 1:
						plt.plot(hist, label=prefix)
						plt.ylim(torch.quantile(hist, 0.), torch.quantile(hist, 0.9))
			plt.ylabel(key)
			plt.xlabel('epoch')
			plt.legend()
			plt.savefig(os.path.join(self.output_dir, f'{key}.png'))
			plt.clf()
		# vector history
		for key in keys:
			if f'val_{key}' in self.history:
				hist = self.history[f'val_{key}']
				if isinstance(hist[0], torch.Tensor) and hist[0].shape[0] > 1:
					for i in range(hist[0].shape[-1]):
						for prefix in ['train', 'val', 'test']:
							if f'{prefix}_{key}' in self.history:
								hist = self.history[f'{prefix}_{key}']
								hist = torch.stack(hist, dim=0)[:, i]
								plt.plot(hist, label=f'{prefix}')
								plt.ylim(torch.quantile(hist, 0.), torch.quantile(hist, 0.9))
						plt.ylabel(f'{key}_{self.dim_labels[i]}')
						plt.xlabel('epoch')
						plt.legend()
						plt.savefig(os.path.join(self.output_dir, f'{key}_{self.dim_labels[i]}.png'))
						plt.clf()
	def plot_y(self, prefix, y_pred, y_target):
		if y_target.shape[-1] != 4: # skip if target doesn't look like a 4 vetor
			return
		for i in range(y_pred.shape[-1]):
			y_target_ = y_target[:, i]
			y_pred_ = y_pred[:, i]
			plt.scatter(y_target_, y_pred_, s=1)
			x = torch.linspace(torch.min(y_target_), torch.max(y_target_), 1000)
			plt.plot(x, x)
			plt.title(prefix)
			plt.xlabel(f'$y^{i}' + '_\mathrm{true}$')
			plt.ylabel(f'$y^{i}' + '_\mathrm{pred}$')
			plt.savefig(os.path.join(self.output_dir, f'{prefix}_y_{i}.png'))
			plt.clf()

		y_dim = y_target.shape[-1]
		# xyzm
		if not self.detector:
			_, axs = plt.subplots(2, y_dim, figsize=(4*y_dim, 4*2), constrained_layout=True)
			configs = utils.get_plot_configs(detector=False)
			plot_comp(axs[0], y_pred, None, y_target, configs[0], configs[1], configs[2])
			pulls_reg, med_pull, med_abs_pull = plot_error(axs[1], y_pred, y_target, configs[0], configs[1], configs[2], label=r'$\mathcal{M}$', detector=False)
			plt.savefig(f'{self.output_dir}/{prefix}_pull_xyzm.png')
			plt.close()
		# detector
		if not self.detector:
			y_pred = utils.xyzm_to_detector(y_pred)
			y_target = utils.xyzm_to_detector(y_target)
		configs = utils.get_plot_configs(detector=True)
		_, axs = plt.subplots(2, y_dim, figsize=(4*y_dim, 4*2), constrained_layout=True)
		plot_comp(axs[0], y_pred, None, y_target, configs[0], configs[1], configs[2])
		pulls_reg, med_pull, med_abs_pull = plot_error(axs[1], y_pred, y_target, configs[0], configs[1], configs[2], label=r'$\mathcal{M}$', detector=True)
		plt.savefig(f'{self.output_dir}/{prefix}_pull_detector.png')
		plt.close()

	def train_model(self, num_epochs, train_loader, val_loader, test_loader):
		self.logger.info(f'Traning on {len(train_loader) * train_loader.batch_size} events')
		for epoch in range(num_epochs):
			start = time.time()
			train_info = self.train_epoch(train_loader)
			elapsed = (time.time() - start)
			val_info = self.validate_model(val_loader, prefix='val')
			if test_loader != None:
				test_info = self.validate_model(test_loader, prefix='test')
			else:
				test_info = val_info
			if self.schedule_lr:
				# self.scheduler.step(val_info['loss'])
				self.scheduler.step()
			self.logger.info('Epoch: {:03d}, Time: {:.3f} s'.format(epoch, elapsed))
			self.logger.info('\tTrain loss: {:.3f}, Val loss: {:.3f}, Test loss: {:.3f}'.format(train_info['loss'], val_info['loss'], test_info['loss']))
			if 'acc' in train_info:
				self.logger.info('\tTrain acc: {:.3f}, Val acc: {:.3f}, Test acc: {:.3f}'.format(train_info['acc'], val_info['acc'], test_info['acc']))
			train_info = add_prefix_to_dict(train_info, 'train')
			val_info = add_prefix_to_dict(val_info, 'val')
			test_info = add_prefix_to_dict(test_info, 'test')
			self.update_history({**train_info, **val_info, **test_info})
			# self.plot_history()
			self.save('most_recent_epoch_model.pt')
			if val_info['val_loss'] <= min(self.history['val_loss']):
				self.save('min_val_loss_model.pt')
			if test_info['test_loss'] <= min(self.history['test_loss']):
				self.save('min_test_loss_model.pt')

	def loss(self, y_pred, y_target):
		raise NotImplementedError

	def forward(self, data):
		raise NotImplementedError

	def run_inference(self, test_loader, max_num_batch=float('inf'), version='nominal', **kwargs):
		raise NotImplementedError