import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU, BatchNorm1d, Dropout, LayerNorm
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from transformer import TransformerEncoder, TransformerDecoder, GraphMLP, GlobalAttention
from base_model import BaseModel
import logging
import time
import datetime
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from functools import partial
import numpy as np
tqdm = partial(tqdm, position=0, leave=True)


class CovariantTopFormer(BaseModel):
	def __init__(self, in_dim, hidden_dim, out_dim, max_num_output, output_dir, use_gpu=True, lr=1e-4, schedule_lr=False, num_convs=(3, 3), heads=8, dist_scale=1, beta=0.5, dropout=0., match_scale_factor=1, p_norm=2, mass=172.76, geometric=False, d_space=3, break_eta_covariance=False, uniform_attention=False):
		self.hidden_dim = hidden_dim
		self.num_convs = num_convs
		self.dropout = dropout
		self.max_num_output = max_num_output
		self.heads = heads
		self.geometric = geometric
		self.d_space = d_space
		self.break_eta_covariance = break_eta_covariance
		self.uniform_attention = uniform_attention
		super().__init__(in_dim, out_dim, output_dir, use_gpu, lr, schedule_lr)
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.beta = beta
		self.match_scale_factor = match_scale_factor.to(self.device)
		self.p_norm = p_norm
		self.loss_scale_factor = 1/torch.FloatTensor([100, 100, 1, 5]).to(self.device) # px, py, eta, m
		self.ghost_value = utils.get_ghost_value()
		self.mass = mass
		self.detector = True
		self.dim_labels = utils.get_plot_configs(detector=True)[0]
		self.pi = 3.1415926

	def define_modules(self):	
		self.invariant_encoder = TransformerEncoder(in_dim=self.in_dim - 2, hidden_dim=self.hidden_dim, num_convs=self.num_convs[0], heads=self.heads, dropout=self.dropout, geometric=self.geometric, update_p=False, d_space=self.d_space, uniform_attention=self.uniform_attention)
		self.pre_decoder = TransformerDecoder(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, num_convs=1, heads=self.heads, dropout=self.dropout, geometric=False)
		self.covariant_decoder = TransformerDecoder(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, num_convs=self.num_convs[1], heads=self.heads, dropout=self.dropout, geometric=self.geometric, update_p=True, d_space=self.d_space, uniform_attention=self.uniform_attention)
		if self.break_eta_covariance:
			self.augment_linear_out = Linear(self.hidden_dim + 1, self.hidden_dim)
			self.augment_linear_source = Linear(self.hidden_dim + 1, self.hidden_dim)
			self.post_decoder = TransformerDecoder(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, num_convs=self.num_convs[2], heads=self.heads, dropout=self.dropout, geometric=self.geometric, update_p=True, d_space=self.d_space, uniform_attention=self.uniform_attention)
		self.out_linear = Linear(self.hidden_dim, 2) # pT and mass
		self.pool_attn_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, 1))
		self.pool_value_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim))
		self.pool = GlobalAttention(self.pool_attn_nn, self.pool_value_nn)
		self.readout_key_nn = GraphMLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout, normalize_input=True)
		self.readout_value_nn = GraphMLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout, normalize_input=True)
		self.readout = Set2Set(in_channels=self.hidden_dim, num_outputs=self.max_num_output, num_layers=1)
		self.count_logits_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, self.max_num_output))

	def angle_to_vec(self, p):
		return torch.stack([p[..., 0], p[..., 1], p[..., 2].cos(), p[..., 2].sin(), p[..., 3]], dim=-1) # (..., 5)
	
	def vec_to_angle_target(self, p):
		invalid_index, _ = (p == self.ghost_value).max(-1) #(...)
		new_p = torch.stack([p[..., 0], p[..., 1], torch.atan2(p[..., 3], p[..., 2]), p[..., 4]], dim=-1) # (..., 4)
		# preserve ghost_values
		new_p[invalid_index] = self.ghost_value
		return new_p

	def vec_to_angle_pred(self, p):
		return self.vec_to_angle_target(p)

	def forward_and_return_loss(self, data, return_y=False):
		# Override because y's are in computed by loss() due to matching
		y_target = data['y']
		y_pred = self(data['x'])
		# convert y_target's phi into unit vectors before passing into loss
		y_target['momenta'] = self.angle_to_vec(y_target['momenta'])
		# FIX ME: add unit-norm bonus
		loss, loss_info = self.loss(y_pred, y_target)
		# get matched targets and predictions, with placeholders all removed
		y_target, y_pred = loss_info.pop('y_target'), loss_info.pop('y_pred')
		# convert unit-vectors to angle
		y_pred_angle_phi = self.vec_to_angle_pred(y_pred)
		y_target_angle_phi = self.vec_to_angle_target(y_target)
		if return_y:
			return loss, loss_info, y_target_angle_phi, y_pred_angle_phi
		return loss, loss_info
	
	def forward(self, input, inference=False, force_correct_num_pred=False):
		graph = input['graph']
		x, edge_index, batch = graph.x.to(self.device), graph.edge_index.to(self.device), graph.batch.to(self.device)
		batch_size = batch.max() + 1
		############ encoder ############
		# process input graph
		# assume x = [pT, eta, phi, m, ...one-hot...]
		p, x = x[:, 1:3], torch.cat([x[:, 0].unsqueeze(-1), x[:, 3:]], dim=-1) 
		x[:, 0] /= 100 # scale down pT
		x[:, 1] /= 10 # scale down mass
		# convert phi into unit-vector representation
		p = torch.stack([p[:, 0], p[:, 1].cos(), p[:, 1].sin()], dim=-1) # (:, 3)
		x_source, p_source = self.invariant_encoder(x, edge_index, p=p, edge_attr=None) # p_source is not updated
		# print(f'encoded source: {x_source.abs().mean()}\n', x_source[:, -5:]) # __debug__

		# global attn pooling to initialize lstm hidden state
		h = self.pool(x_source, batch) # event summary: single vector
		# print(f'pool:{h.abs().mean()}\n', h[:, -5:]) # __debug__
		
		# logits for number of tops prediction
		count_logits = self.count_logits_nn(h)
		
		# lstm readouts
		x_key = self.readout_key_nn(x_source)
		x_value = self.readout_value_nn(x_source)
		x_out = self.readout(x_key, x_value, h, batch)
		x_out = x_out.reshape(batch_size * self.max_num_output, self.hidden_dim) # (N * 4, H)
		# print(f'lstm x_out: {x_out.abs().mean()}\n', x_out[:, -5:]) # __debug__

		# compute number of perdictions
		if not inference or force_correct_num_pred:
			num_pred = input['num_target'].to(self.device)
			assert (num_pred <= self.max_num_output).all(), 'Too many targets'
		else:
			num_pred = torch.argmax(count_logits, axis=1) + 1

		############ decoder ############
		# construct edges between inputs, outputs, and within outputs
		self_attn_edges, cross_attn_edges = self.get_decoder_edges(graph, num_pred, inference)

		# cache edges between inputs and outputs for attention interpretation
		self._cross_attn_edges = cross_attn_edges

		# compute batch_out with invalid output nodes removed
		batch_out = torch.arange(0, batch_size).unsqueeze(-1).repeat(1, self.max_num_output).view(-1).to(x_source.device)
		num_pred = num_pred.view(batch_size, 1)
		out_index = torch.arange(0, self.max_num_output).view(-1, self.max_num_output).repeat(batch_size, 1).to(x_source.device)
		valid_out = (out_index < num_pred).reshape(-1)

		# pre-decoder cross attntion
		x_out, _ = self.pre_decoder(x_source, x_out, self_attn_edges, None, cross_attn_edges, None) # update x_out, p_out == None because pre_decoder is non-geometric
		# retrieve invariant attention
		alpha_init = self.pre_decoder.get_alpha() # (|E|, L=1, H)
		alpha_init = alpha_init.mean(-1).mean(-1) # average over layer and heads

		# initialize p_out
		senders = cross_attn_edges[0, :]
		receivers = cross_attn_edges[1, :]
		# weight p_source[i, j] by a_[i, j]
		p_out = p_source[senders] * alpha_init.unsqueeze(-1) # (|E|, d_space) * (|E|, 1) -> (|E|, d_space)
		# sum over source index
		p_out = scatter_add(p_out, receivers, dim=0, dim_size=batch_size*self.max_num_output) # covariant
		self.y_intermediates = []
		y_init = utils.format_prediction(x_out[:, :2], p_out)
		self.y_intermediates.append(y_init)

		# normalize the phi-vector
		p_out = self.normalize_phi_vec(p_out)
		# perform decoder self & cross attention
		if self.num_convs[1] > 0: # avoid unintended compuation when decoder is supoosed to not exist
			x_out, p_out = self.covariant_decoder(x_source, x_out, self_attn_edges, None, cross_attn_edges, None, p_source=p_source, p_out=p_out)
			self.y_intermediates.extend(self.covariant_decoder.y_intermediates)
		# print(f'decoded p: {p_out.abs().mean()}\n', p_out) # __debug__
		# print(f'decoded x: {x_out.abs().mean()}\n', x_out.abs().mean(-1)) # __debug__
		# break exact covariance with another decoder supplied with absolute eta info
		if self.break_eta_covariance:
			x_out = self.augment_linear_out(torch.cat([x_out, p_out[:, 0].unsqueeze(-1)], dim=-1))
			x_source = self.augment_linear_source(torch.cat([x_source, p_source[:, 0].unsqueeze(-1)], dim=-1))
			x_out, p_out = self.post_decoder(x_source, x_out, self_attn_edges, None, cross_attn_edges, None, p_source=p_source, p_out=p_out)
			self.y_intermediates.extend(self.post_decoder.y_intermediates)
		# scale outputs to speed up initial optimization
		pt_m_pred = torch.FloatTensor([100, 5]).to(self.device) * self.out_linear(x_out)
		eta_phi_pred = p_out
		# target is in [pT, eta, phi_vec, m]
		y_pred = torch.cat([pt_m_pred[..., 0].unsqueeze(-1), eta_phi_pred, pt_m_pred[..., 1].unsqueeze(-1)], dim=-1)
		y_pred = y_pred.reshape(-1, self.max_num_output, 5)
		y_pred = torch.cat([y_pred[..., :-1], y_pred[..., -1].unsqueeze(-1) + self.mass], dim=-1) # add 173 to mass
		if y_pred.isnan().sum() > 0:
			print('y_pred has NAN')
			raise SystemExit
		return y_pred, count_logits
	
	def normalize_phi_vec(self, p, eps=1e-4):
		phi_vec = p[:, 1:]
		phi_vec = phi_vec / (phi_vec.norm(dim=-1, keepdim=True) + eps)
		return torch.cat([p[:, 0].unsqueeze(-1), phi_vec], dim=-1)
	
	def test_covariance(self, input, sigma=1):
		with torch.no_grad():
			batch = input['graph'].batch.to(self.device)
			batch_size = batch.max() + 1
			num_target = input['num_target'].to(self.device)
			assert num_target.max() == num_target.min(), 'num_target is not constant, which is not supported for this test'
			y_pred_1, count_logits_1 = self.forward(input)
			y_pred_1, count_logits_1 = self.vec_to_angle_pred(y_pred_1).detach().cpu(), count_logits_1.detach().cpu()
			# boost and rotate final state objects
			dp = sigma * torch.rand(batch_size, 2) # (B, 2)
			input['graph'].x[:, 1:3] += dp[batch]
			input['graph'].x[:, 2] = self.restrict_range(input['graph'].x[:, 2])
			y_pred_2, count_logits_2 = self.forward(input)
			y_pred_2, count_logits_2 = self.vec_to_angle_pred(y_pred_2).detach().cpu(), count_logits_2.detach().cpu()
			dp = dp.view(-1, 1, 2).repeat(1, self.max_num_output, 1).detach().cpu()
			
			# logits
			plt.figure()
			std = count_logits_1.std()
			mean = count_logits_1.mean()
			bins = torch.linspace(mean - 5 * std, mean + 5 * std, 100)
			plt.hist(count_logits_2.view(-1) - count_logits_1.view(-1), label='$\Delta$ logits', bins=bins, alpha=0.5)
			plt.hist(count_logits_1.view(-1), label='logits', bins=bins, alpha=0.5)
			plt.legend()
			# pT
			plt.figure()
			std = y_pred_1[:, :, 0].std()
			mean = y_pred_1[:, :, 0].mean()
			bins = torch.linspace(mean - 5 * std, mean + 5 * std, 100)
			plt.hist(y_pred_2[:, :, 0].view(-1) - y_pred_1[:, :, 0].view(-1), label='$\Delta$ pT', bins=bins, alpha=0.5)
			plt.hist(y_pred_1[:, :, 0].view(-1), label='pT', bins=bins, alpha=0.5)
			plt.legend()
			# mass
			plt.figure()
			std = y_pred_1[:, :, 3].std()
			mean = y_pred_1[:, :, 3].mean()
			bins = torch.linspace(mean - 5 * std, mean + 5 * std, 100)
			plt.hist(y_pred_2[:, :, 3].view(-1) - y_pred_1[:, :, 3].view(-1), label='$\Delta$ m', bins=bins, alpha=0.5)
			plt.hist(y_pred_1[:, :, 3].view(-1), label='m', bins=bins, alpha=0.5)
			plt.legend()
			# eta
			plt.figure()
			std = 2
			mean = 0
			bins = torch.linspace(mean - 5 * std, mean + 5 * std, 100)
			plt.hist((y_pred_1[:, :, 1] + dp[:, :, 0]).view(-1), label='$\eta_1 + \delta \eta$', bins=bins, alpha=0.5)
			plt.hist(y_pred_2[:, :, 1].view(-1), label='$\eta_2$', bins=bins, alpha=0.5)
			plt.legend()
			# phi
			plt.figure()
			std = 2
			mean = 0
			bins = torch.linspace(mean - 5 * std, mean + 5 * std, 100)
			plt.hist(self.restrict_range(y_pred_1[:, :, 2] + dp[:, :, 1]).view(-1), label='$\phi_1 + \delta \phi$', bins=bins, alpha=0.5)
			plt.hist(y_pred_2[:, :, 2].view(-1), label='$\phi_2$', bins=bins, alpha=0.5)
			plt.legend()

			assert ((count_logits_1 - count_logits_2).abs() < 1e-3 * count_logits_2.abs()).float().mean() > 0.95, "count logits aren't invariant"
			pred_index = torch.arange(self.max_num_output).view(1, -1).repeat(batch_size, 1).to(self.device) # (B, N_max)
			assert ((y_pred_2[:, :, 0] - y_pred_1[:, :, 0]).abs() < 1e-3 * y_pred_1[:, :, 0].abs()).float().mean() > 0.95, "pTs aren't invariant"
			assert ((y_pred_2[:, :, 3] - y_pred_1[:, :, 3]).abs() < 1e-3 * y_pred_1[:, :, 3].abs()).float().mean() > 0.95, "masses aren't invariant"
			assert (((y_pred_2[:, :, 1] - y_pred_1[:, :, 1] - dp[:, :, 0]).abs() < 1e-3 * dp[:, :, 0].abs()) + (pred_index > num_target.unsqueeze(-1)).detach().cpu()).float().mean() > 0.95, "etas aren't covariant"
			assert ((self.restrict_range(y_pred_2[:, :, 2] - y_pred_1[:, :, 2] - dp[:, :, 1])).abs() < 1e-3 * self.restrict_range(dp[:, :, 1]).abs() + (pred_index > num_target.unsqueeze(-1)).detach().cpu()).float().mean() > 0.95, "phis aren't covariant"
			print('All tests passed!')
	
	def restrict_range(self, phi):
		# return phi in [-pi, pi]
		return (phi + self.pi).remainder(2*self.pi) - self.pi 
	
	def reparameterize_target(self, y):
		pT = y[..., 0]
		eta = y[..., 1]
		cos_phi = y[..., 2]
		sin_phi = y[..., 3]
		m = y[..., 4]
		return torch.stack([pT*cos_phi, pT*sin_phi, eta, m], dim=-1)

	def reparameterize_pred(self, y):
		pT = y[..., 0]
		eta = y[..., 1]
		cos_phi = y[..., 2]
		sin_phi = y[..., 3]
		m = y[..., 4]
		return torch.stack([pT*cos_phi, pT*sin_phi, eta, m], dim=-1)

	def loss(self, y_pred, y_target):
		y_pred, count_logits = y_pred
		y_target, num_target = y_target['momenta'].to(self.device), y_target['num_target'].to(self.device)
		# Mask placeholder predictions/targets so that they are properly dealt with in EMD calculation
		y_pred = self.mask_invalid(y_pred, num_target, ghost_value=self.ghost_value)
		y_target = self.mask_invalid(y_target, num_target, ghost_value=self.ghost_value)
		# Counting loss
		count_loss = torch.mean(self.cross_entropy_loss(count_logits, num_target - 1)) # -1 so that 1 top corresponds to class 0
		y_pred_matched, _ = self.match(y_pred, y_target, num_target) # match predictions to target
		diff = self.reparameterize_pred(y_pred_matched) - self.reparameterize_target(y_target) # convert (pT, eta, cos(phi), sin(phi), m) to (pT*cos(phi), pT*sin(phi), eta, m)
		loss_scale_factor = self.loss_scale_factor
		kinematics_loss = torch.norm(loss_scale_factor * diff, p=self.p_norm, dim=-1).sum(-1) / diff.shape[-1] / num_target
		kinematics_loss = kinematics_loss.mean()
		# compute intermediate losses
		y_intermediates = [self.mask_invalid(y.reshape(-1, self.max_num_output, 5), num_target, ghost_value=self.ghost_value) for y in self.y_intermediates[:-1]]
		intermediate_losses = []
		for y_pred in y_intermediates:
			y_pred_matched, _ = self.match(y_pred, y_target, num_target) # match predictions to target
			diff = self.reparameterize_pred(y_pred_matched) - self.reparameterize_target(y_target) # convert (pT, eta, cos(phi), sin(phi), m) to (pT*cos(phi), pT*sin(phi), eta, m)
			this_loss = torch.norm(loss_scale_factor * diff, p=self.p_norm, dim=-1).sum(-1) / diff.shape[-1] / num_target # per-component "normalized" L2 distance
			this_loss = this_loss.mean()
			intermediate_losses.append(this_loss)
		intermediate_loss = torch.stack(intermediate_losses).mean() if intermediate_losses else 0 * kinematics_loss
		loss = (1 - self.beta) * count_loss + self.beta * (0.5 * kinematics_loss + 0.5 * intermediate_loss)
		num_pred = torch.argmax(count_logits, axis=1) + 1
		acc = (num_pred == num_target).float().mean()

		# norm penalty for phi-messages
		if self.geometric:
			if self.num_convs[1] > 0:
				phi_message_norm = self.covariant_decoder.get_phi_message_norm()
				# no loss from the pre-decoder because there's nothing you can do with attention there
				unit_norm_loss = 0.01 * (phi_message_norm - 1).abs().mean()
			else:
				unit_norm_loss = 0 * kinematics_loss
			loss = loss + unit_norm_loss
		loss_info = {
			'loss': loss.item(),
			'count_loss': count_loss.item(),
			'kinematics_loss': kinematics_loss.item(),
			'intermediate_loss': intermediate_loss.item(),
			'unit_norm_loss': unit_norm_loss.item() if self.geometric else 0,
			'-count_acc': -acc.item(),
			# Return non-placeholder predictions/targets for logging metrics
			'y_pred': y_pred_matched[y_pred_matched != self.ghost_value].reshape(-1, y_pred.shape[-1]).detach().cpu(),
			'y_target': y_target[y_target != self.ghost_value].reshape(-1, y_target.shape[-1]).detach().cpu(),
		}
		loss_info.update({f'intermediate_loss_{i}': intermediate_losses[i].item() for i in range(len(intermediate_losses))})
		return loss, loss_info

	def match(self, y_pred, y_target, num_target):
		_, y_matched_pred, opt_perm = self.masked_earth_mover_distance(y_pred, y_target, num_target, self.p_norm, scale=self.match_scale_factor)
		return y_matched_pred, opt_perm

	def masked_earth_mover_distance(self, y_pred, y_target, num_target, p_norm, scale):
		""" y_pred & y_target: (B, N, D),
			B = batch size, 
			N = number of objects in each event, 
			D = dimension of the vector representing each object
		"""
		B = y_pred.shape[0]
		N = y_pred.shape[1]
		D = y_pred.shape[2]
		perms = list(itertools.permutations(range(N)))
		P = len(perms)
		perms = torch.LongTensor(list(perms)).view(-1, 1, N, 1).repeat(1, B, 1, D).to(y_target.device) # (P, B, N, D)
		Y_pred = y_pred.unsqueeze(0).repeat(P, 1, 1, 1) # (P, B, N, D)
		Y_pred = Y_pred.gather(2, perms) # (P, B, N, D) Y_pred[p, b, n, d] = y_pred[p, b, perms[p, b, n, d]==perms[p, _, n, _], d]
		Y_target = y_target.unsqueeze(0).repeat(P, 1, 1, 1) # (P, B, N, D)
		# convert vec to angle to compute dR
		diff = (self.vec_to_angle_pred(Y_pred) - self.vec_to_angle_target(Y_target))
		diff = torch.stack([diff[..., 0], diff[..., 1], self.restrict_range(diff[..., 2]), diff[..., 3]], dim=-1)
		# sum over 4-vector components and average over objects in each event -> (P, B)
		# dY[P, B]: the error in the B-th event if permuting according to the P-th permuation.
		dY = torch.norm(scale * diff, p=p_norm, dim=-1).sum(-1) / num_target 
		# opt_perm: the index for the optimal permutation in each event
		dY_emd, opt_perm = dY.min(0)
		opt_perm_ = opt_perm.view(1, B, 1, 1).repeat(1, 1, N, D)
		y_matched_pred = Y_pred.gather(0, opt_perm_)
		return dY_emd, y_matched_pred, opt_perm # dY_emd: (B,)

	def permute(self, y, perm, inverse=False):
		""" y: (B, N, D)
			B = batch size, 
			N = number of objects in each event, 
			D = dimension of the vector representing each object
			perm = the index for the permutation to apply
		"""
		B = y.shape[0]
		N = y.shape[1]
		D = y.shape[2]
		perms = list(itertools.permutations(range(N)))
		if inverse:
			perms = np.argsort(np.array(perms), axis=-1).tolist()
		P = len(perms)
		perms = torch.LongTensor(list(perms)).view(-1, 1, N, 1).repeat(1, B, 1, D).to(y.device) # (P, B, N, D)
		Y = y.unsqueeze(0).repeat(P, 1, 1, 1) # (P, B, N, D)
		Y = Y.gather(2, perms) # (P, B, N, D) Y[p, b, n, d] = y[p, b, perms[p, b, n, d]==perms[p, _, n, _], d]
		perm = perm.view(1, B, 1, 1).repeat(1, 1, N, D)
		y_matched = Y.gather(0, perm)
		return y_matched

	def mask_invalid(self, y, num_valid, ghost_value):
		B = y.shape[0]
		N_max = y.shape[1]
		D = y.shape[2]
		num_valid = num_valid.unsqueeze(-1).repeat(1, N_max)
		ghost_value = ghost_value * torch.ones_like(y)
		index = torch.arange(N_max).repeat(y.shape[0], 1).to(y.device)
		mask = (index < num_valid).long()
		mask = mask.unsqueeze(-1)
		y_masked = mask * y + (1 - mask) * ghost_value
		return y_masked

	def get_decoder_edges(self, graph, num_pred, inference):	
		if not inference: # During training, edges were precomputed in the dataset and batched as rectangular tensors to speed up data loading
			# some preprocessing to get the shapes right
			cross_attn_edges, self_attn_edges, n_x, batch_size = graph.cross_attn_edges, graph.self_attn_edges, graph.n_x, graph.batch.max() + 1
			cross_attn_edges, self_attn_edges, n_x = cross_attn_edges.to(self.device), self_attn_edges.to(self.device), n_x.to(self.device)
			cross_attn_edges = cross_attn_edges.reshape([-1, 2, graph.max_cross_attn_edges[0]])
			self_attn_edges = self_attn_edges.reshape([-1, 2, graph.max_self_attn_edges[0]])
			source_index = torch.arange(batch_size).to(self.device)
			source_offset_ = torch.cumsum(n_x, dim=0).to(self.device)
			source_offset = torch.cat([torch.zeros(1).to(self.device), source_offset_[:-1]]).long()
			output_offset = torch.arange(0, self.max_num_output * batch_size, self.max_num_output).long().to(self.device)
			# apply source offset
			cross_attn_edges[:, 0, :] = cross_attn_edges[:, 0, :] + source_offset.unsqueeze(-1)
			# apply output offset
			cross_attn_edges[:, 1, :] = cross_attn_edges[:, 1, :] + output_offset.unsqueeze(-1)
			# apply output offset
			self_attn_edges = self_attn_edges + output_offset.view(-1, 1, 1)
			# remove padding
			self_attn_edges = self.get_valid_edges_indices(self_attn_edges)
			cross_attn_edges = self.get_valid_edges_indices(cross_attn_edges)
		if inference: # edges are constructed on the fly because num_pred varies depending on the model
			graph = utils.get_xy_graph(graph, num_pred, self.max_num_output)
			cross_attn_edges, self_attn_edges = graph.edge_index_cross, graph.edge_index_out
		return self_attn_edges, cross_attn_edges

	def get_valid_edges_indices(self, stacked_edge_index):
		edge_index = stacked_edge_index.swapaxes(-1, -2).reshape(-1, 2)
		keep = edge_index[:, 0] >= 0
		return edge_index[keep].t()

	def run_inference(self, test_loader, max_num_batch=float('inf'), version='nominal', force_correct_num_pred=False):
		self.eval()
		self.float()
		count_correct_cum = 0
		kinematics_diff_norm_l1_cum = 0 # sum over batch of [avg over matched pairs of (the norm of kinematics difference)]
		kinematics_diff_norm_l2_cum = 0
		total = 0
		truth_matched = []
		identified = []
		info = []
		W_decay_pid = []
		Y_target = []
		Y_pred = []
		Y_reco = []
		Y_gnn_reco = []
		N_bj = []
		N_target = []
		N_pred = []
		probs = []
		top_attention_indices = []
		triplet_attn = []
		KL = []
		N_object = []
		pred_errors = []
		reco_triplet_indices = []
		attention_matched = []
		gnn_reco_matched = []
		count = 0
		with torch.no_grad():
			for data in tqdm(test_loader):
				if count >= max_num_batch:
					break
				count += 1
				####### Get targets #######
				y_target = data['y']
				y_target, num_target = y_target['momenta'].to(self.device), y_target['num_target'].to(self.device)
				y_target = self.angle_to_vec(y_target)
				####### Get inputs #######
				input = data['x']
				input = {k: v.to(self.device) for k, v in input.items()}
				input_features = input['graph'].x
				is_jet = (input_features[:, 4] + input_features[:, 5]) > 0 # bjet or non-bjet
				####### Prediction #######
				# Unless force_correct_num_pred == True, the model predicts number of outputs in inference mode
				y_pred, count_logits = self(input, inference=True, force_correct_num_pred=force_correct_num_pred)
				prob = F.softmax(count_logits, dim=1).unsqueeze(0)
				num_pred = torch.argmax(count_logits, axis=1) + 1 if not force_correct_num_pred else num_target

				####### Mask and match #######
				# Mask placeholder predictions/targets so that they are properly dealt with in EMD calculation
				y_pred = self.mask_invalid(y_pred, num_pred, ghost_value=self.ghost_value) # Note we use num_pred as opposed to num_target as done in loss()
				y_target = self.mask_invalid(y_target, num_target, ghost_value=self.ghost_value)

				count_correct = num_pred == num_target
				# Only correctly counted events are matched
				if count_correct.sum() > 0:
					# Match by assuming there are num_target predictions and targets
					y_pred[count_correct], opt_perm = self.match(y_pred[count_correct], y_target[count_correct], num_target[count_correct])

					# permute reco_triplet_index so that triplet attention is computed correctly for each predicetd top
					reco_triplet_index = data['reco_triplet_indices'].view(-1, self.max_num_output, 3).to(self.device) # (B, N, 3)
					reco_triplet_index[count_correct] = self.permute(reco_triplet_index[count_correct], opt_perm, inverse=True)
					# Get the top-3 attention index for each target in this batch, placeholder targets and targets with less than 3 input objects return [-2, -2, -2].
					if hasattr(self, '_cross_attn_edges'):
						top_attention_index, triplet_attention, kl, n_object = self.get_attention_info(reco_triplet_index, is_jet) # (*, 3), (*), (*)
						top_attention_index = top_attention_index.view(-1, self.max_num_output, 3) # (B, N, 3)
					else:
						top_attention_index = (reco_triplet_index * 0).to(self.device)
					# match prediction to truth instead of the other way around
					reco_triplet_index = data['reco_triplet_indices'].view(-1, self.max_num_output, 3).to(self.device)
					top_attention_index[count_correct] = self.permute(top_attention_index[count_correct], opt_perm)
					triplet_attention = triplet_attention.view(-1, self.max_num_output, 1).to(self.device)
					kl = kl.view(-1, self.max_num_output, 1).to(self.device)
					n_object = n_object.view(-1, self.max_num_output, 1).to(self.device)
					kl[count_correct] = self.permute(kl[count_correct], opt_perm)
					n_object[count_correct] = self.permute(n_object.view(-1, self.max_num_output, 1)[count_correct], opt_perm)
					triplet_attention[count_correct] = self.permute(triplet_attention.view(-1, self.max_num_output, 1)[count_correct], opt_perm)
					kl = kl.view(-1)
					n_object = n_object.view(-1)
					triplet_attention = triplet_attention.view(-1)

				# Compute prediction errors on matched events
				pred_error = torch.norm(self.loss_scale_factor * (self.reparameterize_pred(y_pred) - self.reparameterize_target(y_target)), p=2, dim=-1) / self.loss_scale_factor.shape[-1]
				pred_error = pred_error.view(-1)
				if count_correct.sum() > 0:
					kinematics_diff_norm_l1 = torch.norm(self.loss_scale_factor * (self.reparameterize_pred(y_pred[count_correct]) - self.reparameterize_target(y_target[count_correct])), p=1, dim=-1) / self.loss_scale_factor.shape[-1]
					kinematics_diff_norm_l1_cum += (kinematics_diff_norm_l1.sum(dim=-1) / num_pred[count_correct]).sum()
					kinematics_diff_norm_l2 = torch.norm(self.loss_scale_factor * (self.reparameterize_pred(y_pred[count_correct]) - self.reparameterize_target(y_target[count_correct])), p=2, dim=-1) / self.loss_scale_factor.shape[-1]
					kinematics_diff_norm_l2_cum += (kinematics_diff_norm_l2.sum(dim=-1) / num_pred[count_correct]).sum()
					count_correct_cum += count_correct.sum()
				
				####### Get additional info #######
				num_graphs = input['graph'].num_graphs
				y_reco = data['reco_top'].float().to(self.device)
				num_target = data['num_target'].long().to(self.device)
				total += num_graphs
				# count realistic bjets
				n_bj = scatter_add(input['graph'].x[:, 5], input['graph'].batch, dim=0)
				####### Get GNN reco info #######
				y_gnn_reco = data['gnn_reco_top'].float().to(self.device)
				gnn_predicted = data['gnn_predicted'].long().to(self.device)
				# repeat gnn reco pred and we'll only take the truth-matched one in the end
				not_predicted = gnn_predicted[:, 1].logical_not()
				y_gnn_reco[not_predicted, 1] = y_gnn_reco[not_predicted, 0] #(1 - gnn_predicted[:, 1]).unsqueeze(-1) * y_gnn_reco[:, 0, :] + gnn_predicted[:, 1].unsqueeze(-1)  * y_gnn_reco[:, 1, :]
				# do dR matching between gnn reco and truth
				_, gnn_reco_perm = self.match(self.angle_to_vec(y_gnn_reco), y_target, num_target)
				y_gnn_reco = self.permute(y_gnn_reco, gnn_reco_perm).squeeze(0)

				####### Convert vectorized-phi to angle #######
				y_target = self.vec_to_angle_target(y_target)
				y_pred = self.vec_to_angle_pred(y_pred)

				####### Fill arrays #######
				Y_target.extend(y_target)
				Y_reco.extend(y_reco)
				Y_pred.extend(y_pred)
				Y_gnn_reco.extend(y_gnn_reco)
				N_bj.extend(n_bj)
				N_target.extend(num_target)
				N_pred.extend(num_pred)
				probs.extend(prob)
				truth_matched.extend(data['truth_matched'])
				triplet_attn.extend(triplet_attention)
				KL.extend(kl.tolist())
				N_object.extend(n_object.tolist())
				pred_errors.extend(pred_error.tolist())
				identified.extend(data['identified'])
				W_decay_pid.extend(data['W_decay_pid'])
				info.extend(data['info'])
				top_attention_indices.extend(top_attention_index)
				reco_triplet_indices.extend(reco_triplet_index)
				attention_matched.extend([set(a.tolist()) == set(b.tolist()) for a, b in zip(top_attention_index.view(-1, 3), reco_triplet_index.view(-1, 3))])
		
		####### Averaging #######
		mean_kinematics_diff_norm_l1 = kinematics_diff_norm_l1_cum / count_correct_cum
		mean_kinematics_diff_norm_l2 = kinematics_diff_norm_l2_cum / count_correct_cum
		count_acc = count_correct_cum / total
		
		####### Combined batches #######
		probs = torch.cat(probs, dim=0)
		N_pred = torch.LongTensor(N_pred)
		N_target = torch.LongTensor(N_target)
		N_bj = torch.stack(N_bj)
		Y_target = torch.cat(Y_target, dim=0)
		Y_reco = torch.cat(Y_reco, dim=0)
		Y_pred = torch.cat(Y_pred, dim=0)
		Y_gnn_reco = torch.cat(Y_gnn_reco, dim=0)
		gnn_reco_matched = torch.all(Y_reco == Y_gnn_reco, dim=-1)
		y_dim = Y_target.shape[-1]
		Y_target = Y_target.reshape(-1, y_dim)
		Y_pred = Y_pred.reshape(-1, y_dim)
		Y_reco = Y_reco.reshape(-1, y_dim)
		truth_matched = torch.cat(truth_matched, dim=0).reshape(-1)
		identified = torch.cat(identified, dim=0).reshape(-1)
		W_decay_pid = torch.cat(W_decay_pid, dim=0).reshape(-1)
		info = torch.stack(info, dim=0)
		top_attention_indices = torch.stack(top_attention_indices, dim=0)
		triplet_attn = torch.FloatTensor(triplet_attn)
		KL = torch.FloatTensor(KL)
		N_object = torch.FloatTensor(N_object)
		pred_errors = torch.FloatTensor(pred_errors)
		reco_triplet_indices = torch.stack(reco_triplet_indices, dim=0)
		attention_matched = torch.BoolTensor(attention_matched)
		test_result_dict = {
							'probs': probs.detach().cpu().numpy(),
							'N_bj': N_bj.detach().cpu().numpy(),
							'num_pred': N_pred.detach().cpu().numpy(),
							'num_target': N_target.detach().cpu().numpy(),
							'y_pred': Y_pred.detach().cpu().numpy(),
							'y_target': Y_target.detach().cpu().numpy(),
							'y_reco': Y_reco.detach().cpu().numpy(),
							'y_gnn_reco': Y_gnn_reco.detach().cpu().numpy(),
							'mean_kinematics_diff_norm_l1': mean_kinematics_diff_norm_l1.detach().cpu().numpy(),
							'mean_kinematics_diff_norm_l2': mean_kinematics_diff_norm_l2.detach().cpu().numpy(),
							'count_acc': count_acc.detach().cpu().numpy(),
							'truth_matched': truth_matched.detach().cpu().numpy(),
							'identified': identified.detach().cpu().numpy(),
							'W_decay_pid': W_decay_pid.detach().cpu().numpy(),
							'info': info.detach().cpu().numpy(),
							'top_attention_indices': top_attention_indices.detach().cpu().numpy(),
							'triplet_attn': triplet_attn.detach().cpu().numpy(),
							'KL': KL.detach().cpu().numpy(),
							'N_object': N_object.detach().cpu().numpy(),
							'pred_errors': pred_errors.detach().cpu().numpy(),
							'reco_triplet_indices': reco_triplet_indices.detach().cpu().numpy(),
							'attention_matched': attention_matched.detach().cpu().numpy(),
							'gnn_reco_matched': gnn_reco_matched.detach().cpu().numpy(),
						}
		torch.save(test_result_dict, os.path.join(self.output_dir, f'test_result_{version}.pt'))
		self.logger.info(f"Saved test result at {os.path.join(self.output_dir, f'test_result_{version}.pt')}")
		return test_result_dict

	def get_attention_info(self, reco_triplet_index, mask=None):
		# get the index of input objects with top-3 attention scores for each output
		# mask: if not None, 1 for valid, 0 for invalid nodes
		reco_triplet_index = reco_triplet_index.view(-1, 3)
		if self.num_convs[1] == 0: # avoid unintended compuation when decoder is supoosed to not exist
			return -2 * torch.ones(self._cross_attn_edges[1].max() + 1, 3).to(self.device)
		cross_attn_edges = self._cross_attn_edges
		if mask is not None:
			# remove edges (i, j) where mask[i] == 0
			edge_mask = mask[cross_attn_edges[0]] == 1 # (|E|,)
		else:
			edge_mask = torch.ones(cross_attn_edges[0].shape).to(self.device) == 1
		a = self.get_alpha() # (|E|, L, H)
		a = a.mean(-1).mean(-1) # avg over decoder layers and heads
		index_top = []
		triplet_a = []
		kl = []
		n_object = []
		for j in range(cross_attn_edges[1].max() + 1):
			goes_to_j = (cross_attn_edges[1] == j) & edge_mask
			n_obj = (goes_to_j).sum()
			n_object.append(n_obj)
			a_j = a[goes_to_j]
			kl.append(n_obj.log() + (a_j * a_j.log()).sum())
			if 3 > n_obj:
				index_top_ = -2 * torch.ones(3).to(a.device)
				triplet_a.append(-1)
			else:
				a_top_, index_top_ = a_j.topk(3)
				reco_triplet_index_j = reco_triplet_index[j]
				if (reco_triplet_index_j >= 0).all(): # if matched to a triplet
					triplet_a.append(a_j[reco_triplet_index_j].sum().item()) # get total attention over the triplet
				else:
					triplet_a.append(-1)
				# a_top.append(a_top_)
			index_top.append(index_top_)
		return torch.stack(index_top, dim=0), torch.FloatTensor(triplet_a), torch.FloatTensor(kl), torch.LongTensor(n_object)

	
	def get_alpha(self):
		# get cached attention scores
		# return torch.cat([self.pre_decoder.get_alpha(), self.covariant_decoder.get_alpha()], dim=1) # (|E|, 1 + L, H)
		return self.covariant_decoder.get_alpha() # (|E|, L, H)

class Set2Set(torch.nn.Module):
	def __init__(self, in_channels, num_outputs, num_layers=1):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = 2 * in_channels
		self.num_outputs = num_outputs
		self.num_layers = num_layers
		self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels, num_layers)
		self.reset_parameters()

	def reset_parameters(self):
		self.lstm.reset_parameters()

	def forward(self, k, v, h, batch):
		""""""
		batch_size = batch.max().item() + 1
		h = (h.view((self.num_layers, batch_size, self.in_channels)),
			 h.view((self.num_layers, batch_size, self.in_channels)))
		q_star = v.new_zeros(batch_size, self.out_channels)
		q = v.new_zeros(batch_size, self.in_channels)
		readouts = []

		for i in range(self.num_outputs):
			q, h = self.lstm(q_star.unsqueeze(0), h) # updates querie and hidden state
			q = q.view(batch_size, self.in_channels)
			e = (k * q[batch]).sum(dim=-1, keepdim=True)
			a = softmax(e, batch, num_nodes=batch_size) # attend over input
			r = scatter_add(a * v, batch, dim=0, dim_size=batch_size) # readout
			readouts.append(r)
			q_star = torch.cat([q, r], dim=-1)
		readout = torch.cat(readouts, dim=-1)
		return readout

	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class NonCovariantTopFormer(CovariantTopFormer):
	def __init__(self, in_dim, hidden_dim, out_dim, max_num_output, output_dir, use_gpu=True, lr=1e-4, schedule_lr=False, num_convs=(3, 3), heads=8, dist_scale=1, beta=0.5, dropout=0., match_scale_factor=1, p_norm=2, mass=172.76):
		super().__init__(in_dim, hidden_dim, out_dim, max_num_output, output_dir, use_gpu, lr, schedule_lr, num_convs, heads, dist_scale, beta, dropout, match_scale_factor, p_norm, mass, geometric=True)

	def define_modules(self):	
		self.invariant_encoder = TransformerEncoder(in_dim=self.in_dim + 1, hidden_dim=self.hidden_dim, num_convs=self.num_convs[0], heads=self.heads, dropout=self.dropout, geometric=self.geometric, update_p=False, d_space=self.d_space)
		self.pre_decoder = TransformerDecoder(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, num_convs=1, heads=self.heads, dropout=self.dropout, geometric=False)
		self.covariant_decoder = TransformerDecoder(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, num_convs=self.num_convs[1], heads=self.heads, dropout=self.dropout, geometric=self.geometric, update_p=True, d_space=self.d_space)
		self.out_linear = Linear(self.hidden_dim, 2) # pT and mass
		self.pool_attn_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, 1))
		self.pool_value_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim))
		self.pool = GlobalAttention(self.pool_attn_nn, self.pool_value_nn)
		self.readout_key_nn = GraphMLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout, normalize_input=True)
		self.readout_value_nn = GraphMLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout, normalize_input=True)
		self.readout = Set2Set(in_channels=self.hidden_dim, num_outputs=self.max_num_output, num_layers=1)
		self.count_logits_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, self.max_num_output))

	def forward(self, input, inference=False, force_correct_num_pred=False):
		graph = input['graph']
		x, edge_index, batch = graph.x.to(self.device), graph.edge_index.to(self.device), graph.batch.to(self.device)
		batch_size = batch.max() + 1
		############ encoder ############
		# process input graph
		# assume x = [pT, eta, phi, m, ...one-hot...]
		p, x = x[:, 1:3], torch.cat([x[:, 0].unsqueeze(-1), x[:, 3:]], dim=-1) 
		x[:, 0] /= 100 # scale down pT
		x[:, 1] /= 10 # scale down mass
		# convert phi into unit-vector representation
		p = torch.stack([p[:, 0], p[:, 1].cos(), p[:, 1].sin()], dim=-1) # (:, 3)
		# treat coordinates as invariant features, breaking covariance
		x = torch.cat([x, p], dim=-1)
		x_source, p_source = self.invariant_encoder(x, edge_index, p=p, edge_attr=None) # p_source is not updated
		# print(f'encoded source: {x_source.abs().mean()}\n', x_source[:, -5:]) # __debug__

		# global attn pooling to initialize lstm hidden state
		h = self.pool(x_source, batch)
		# print(f'pool:{h.abs().mean()}\n', h[:, -5:]) # __debug__
		
		# logits for number of tops prediction
		count_logits = self.count_logits_nn(h)
		
		# lstm readouts
		x_key = self.readout_key_nn(x_source)
		x_value = self.readout_value_nn(x_source)
		x_out = self.readout(x_key, x_value, h, batch)
		x_out = x_out.reshape(batch_size * self.max_num_output, self.hidden_dim) # (N * 4, H)
		# print(f'lstm x_out: {x_out.abs().mean()}\n', x_out[:, -5:]) # __debug__

		# compute number of perdictions
		if not inference or force_correct_num_pred:
			num_pred = input['num_target'].to(self.device)
			assert (num_pred <= self.max_num_output).all(), 'Too many targets'
		else:
			num_pred = torch.argmax(count_logits, axis=1) + 1

		############ decoder ############
		# construct edges between inputs, outputs, and within outputs
		self_attn_edges, cross_attn_edges = self.get_decoder_edges(graph, num_pred, inference)

		# cache edges between inputs and outputs for attention interpretation
		self._cross_attn_edges = cross_attn_edges

		# compute batch_out with invalid output nodes removed
		batch_out = torch.arange(0, batch_size).unsqueeze(-1).repeat(1, self.max_num_output).view(-1).to(x_source.device)
		num_pred = num_pred.view(batch_size, 1)
		out_index = torch.arange(0, self.max_num_output).view(-1, self.max_num_output).repeat(batch_size, 1).to(x_source.device)
		valid_out = (out_index < num_pred).reshape(-1)

		# pre-decoder cross attntion
		x_out, _ = self.pre_decoder(x_source, x_out, self_attn_edges, None, cross_attn_edges, None) # update x_out, p_out == None because pre_decoder is non-geometric
		# retrieve invariant attention
		alpha_init = self.pre_decoder.get_alpha() # (|E|, L=1, H)
		alpha_init = alpha_init.mean(-1).mean(-1) # average over layer and heads

		# initialize p_out
		senders = cross_attn_edges[0, :]
		receivers = cross_attn_edges[1, :]
		# weight p_source[i, j] by a_[i, j]
		p_out = p_source[senders] * alpha_init.unsqueeze(-1) # (|E|, d_space) * (|E|, 1) -> (|E|, d_space)
		# sum over source index
		p_out = scatter_add(p_out, receivers, dim=0) # covariant
		self.y_intermediates = []
		y_init = utils.format_prediction(x_out[:, :2], p_out)
		self.y_intermediates.append(y_init)

		# normalize the phi-vector
		p_out = self.normalize_phi_vec(p_out)
		# perform decoder self & cross attention
		x_out, p_out = self.covariant_decoder(x_source, x_out, self_attn_edges, None, cross_attn_edges, None, p_source=p_source, p_out=p_out)
		self.y_intermediates.extend(self.covariant_decoder.y_intermediates)
		# print(f'decoded p: {p_out.abs().mean()}\n', p_out) # __debug__
		# print(f'decoded x: {x_out.abs().mean()}\n', x_out.abs().mean(-1)) # __debug__
		# break exact covariance with another decoder supplied with absolute eta info
		# scale outputs to speed up initial optimization
		pt_m_pred = torch.FloatTensor([100, 5]).to(self.device) * self.out_linear(x_out)
		eta_phi_pred = p_out
		# target is in [pT, eta, phi_vec, m]
		y_pred = torch.cat([pt_m_pred[..., 0].unsqueeze(-1), eta_phi_pred, pt_m_pred[..., 1].unsqueeze(-1)], dim=-1)
		y_pred = y_pred.reshape(-1, self.max_num_output, 5)
		y_pred = torch.cat([y_pred[..., :-1], y_pred[..., -1].unsqueeze(-1) + self.mass], dim=-1) # add 173 to mass
		if y_pred.isnan().sum() > 0:
			print('y_pred has NAN')
			raise SystemExit
		return y_pred, count_logits


class Classifier(BaseModel):
	def __init__(self, in_dim, hidden_dim, out_dim, num_classes, output_dir, use_gpu=True, lr=1e-4, schedule_lr=False, num_convs=3, d_space=3, heads=4, mass=172.76, dropout=0, boost_invariant=True):
		self.hidden_dim = hidden_dim
		self.num_convs = num_convs
		self.dropout = dropout
		self.heads = heads
		self.geometric = True
		self.d_space = d_space
		self.num_classes = num_classes
		self.uniform_attention = False
		self.boost_invariant = boost_invariant
		super().__init__(in_dim, out_dim, output_dir, use_gpu, lr, schedule_lr)
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.mass = mass
		self.detector = True

	def define_modules(self):	
		self.invariant_encoder = TransformerEncoder(in_dim=self.in_dim - 2 + int(not self.boost_invariant), hidden_dim=self.hidden_dim, num_convs=self.num_convs, heads=self.heads, dropout=self.dropout, geometric=self.geometric, update_p=False, d_space=self.d_space, uniform_attention=self.uniform_attention)
		self.pool_attn_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, 1))
		self.pool_value_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim))
		self.pool = GlobalAttention(self.pool_attn_nn, self.pool_value_nn)
		self.logits_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, self.num_classes))

	def forward_and_return_loss(self, data, return_y=False):
		y_target = data.y.to(self.device)
		y_pred = self(data)
		loss, loss_info = self.loss(y_pred, y_target)
		if return_y:
			return loss, loss_info, y_target.detach().cpu(), y_pred.detach().cpu()
		return loss, loss_info
	
	def forward(self, data):
		x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)
		# assume x = [pT, eta, phi, m, ...one-hot...]
		p, x = x[:, 1:3], torch.cat([x[:, 0].unsqueeze(-1), x[:, 3:]], dim=-1) 
		x[:, 0] /= 100 # scale down pT
		# convert phi into unit-vector representation
		p = torch.stack([p[:, 0], p[:, 1].cos(), p[:, 1].sin()], dim=-1) # (:, 3)
		# treat eta as invariant feature, breaking boost invariance
		if not self.boost_invariant:
			x = torch.cat([x, p[:, 0].unsqueeze(-1)], dim=-1)
		x_source, p_source = self.invariant_encoder(x, edge_index, p=p, edge_attr=None) # p_source is not updated
		# global attn pooling
		h = self.pool(x_source, batch)
		# logits for class identity
		logits = self.logits_nn(h)
		return logits

	def loss(self, y_pred, y_target):
		class_target = y_target
		logits = y_pred
		loss = torch.mean(self.cross_entropy_loss(logits, class_target))
		class_pred = torch.argmax(logits, axis=1)
		acc = (class_pred == class_target).float().mean()
		loss_info = {
			'loss': loss.item(),
			'acc': acc.item(),
		}
		return loss, loss_info