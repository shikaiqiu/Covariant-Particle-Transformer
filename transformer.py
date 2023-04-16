import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU, Dropout, LayerNorm, Identity, Sigmoid
from torch_geometric.nn import GCNConv, global_add_pool, Set2Set, GlobalAttention
from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import zeros, ones
import utils

def zero_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0)

def gate_linear_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(1)

class ResidualGraphConv(torch.nn.Module):
	def __init__(self, in_dim, heads):
		super().__init__()
		self.norm_conv = LayerNorm(in_dim)
		self.conv = GCNConv(in_dim, in_dim)

	def forward(self, x, edge_index):
		x = F.relu(x + self.conv(self.norm_conv(x), edge_index)) # not using edge attr
		return x

class ResidualSelfAttention(torch.nn.Module):
	def __init__(self, hidden_dim, heads, edge_dim, dropout=0., geometric=False, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.norm_conv = LayerNorm(hidden_dim)
		self.edge_dim = edge_dim
		self.geometric = geometric
		self.d_space = d_space
		if edge_dim != None:
			self.norm_edge = LayerNorm(edge_dim)
		self.uniform_attention = uniform_attention
		self.conv = CovariantGraphAttention(hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=self.uniform_attention)

	def forward(self, x, p, edge_index, edge_attr):
		x_normed = self.norm_conv(x)
		if self.geometric:
			in_feat = torch.cat([p, x_normed], dim=-1)
		else:
			in_feat = x_normed
		if self.edge_dim != None:
			m_x, p = self.conv(in_feat, edge_index, self.norm_edge(edge_attr))
			x = x + m_x # p = p + m_p, already
		else:
			m_x, p = self.conv(in_feat, edge_index)
			x = x + m_x # p = p + m_p, already
		# if self.geometric:
		# 	print('self_attn:\n', p) # __debug__
		return x, p

class ResidualCrossAttentionBlock(torch.nn.Module):
	def __init__(self, hidden_dim, heads, edge_dim, dropout=0., geometric=False, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.norm_conv_source = LayerNorm(hidden_dim)
		self.norm_conv_out = LayerNorm(hidden_dim)
		self.geometric = geometric
		self.edge_dim = edge_dim
		if edge_dim != None:
			self.norm_edge = LayerNorm(edge_dim)
		self.uniform_attention = uniform_attention
		self.conv = CovariantGraphAttention(hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=self.uniform_attention)
	
	def get_alpha(self):
		return self.conv._alpha

	def get_phi_message_norm(self):
		return self.conv._phi_message_norm
	
	def forward(self, x_source, p_source, x_out, p_out, cross_edge_index, cross_edge_attr):
		x_normed_source = self.norm_conv_source(x_source)
		x_normed_out = self.norm_conv_out(x_out)
		if self.geometric:
			in_feat = (torch.cat([p_source, x_normed_source], dim=-1), torch.cat([p_out, x_normed_out], dim=-1))
		else:
			in_feat = (x_normed_source, x_normed_out)
		if self.edge_dim != None:
			m_x_out, p_out = self.conv(in_feat, cross_edge_index, self.norm_edge(cross_edge_attr)) # not using edge attr
			x_out = x_out + m_x_out
		else:
			m_x_out, p_out = self.conv(in_feat, cross_edge_index) # not using edge attr
			x_out = x_out + m_x_out
		# if self.geometric:
		# 	print('cross_attn:\n', p_out) # __debug__
		return x_out, p_out

class ResidualFeedForward(torch.nn.Module):
	def __init__(self, hidden_dim, dropout=0.):
		super().__init__()
		self.norm_fc = LayerNorm(hidden_dim)
		self.fc = Sequential(Dropout(dropout), Linear(hidden_dim, hidden_dim), LeakyReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim))

	def forward(self, x):
		return x + self.fc(self.norm_fc(x))

class TransformerEncoderBlock(torch.nn.Module):
	def __init__(self, hidden_dim, heads, edge_dim, dropout=0., geometric=False, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.self_attn = ResidualSelfAttention(hidden_dim, heads, edge_dim, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention)
		self.feedforward = ResidualFeedForward(hidden_dim, dropout=dropout)
		self.edge_dim = edge_dim
		self.geometric = geometric
		self.d_space = d_space
		if self.edge_dim != None:
			self.edge_nn = Sequential(Dropout(dropout), LayerNorm(edge_dim + 2 * hidden_dim), Linear(edge_dim + 2 * hidden_dim, edge_dim), LeakyReLU(), Dropout(dropout), Linear(edge_dim, edge_dim))
		
	def forward(self, x, p, edge_index, edge_attr):
		x, p = self.self_attn(x, p, edge_index, edge_attr)
		x = self.feedforward(x)
		if self.edge_dim != None:
			x_i, x_j = x[edge_index[0]], x[edge_index[1]]
			edge_attr = edge_attr + self.edge_nn(torch.cat([edge_attr, x_i, x_j], dim=-1))
		return x, p, edge_attr

class TransformerDecoderBlock(torch.nn.Module):
	def __init__(self, hidden_dim, edge_dim, heads, dropout=0., geometric=False, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.geometric = geometric
		self.self_attn = ResidualSelfAttention(hidden_dim, heads, edge_dim=None, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention)
		self.cross_attn = ResidualCrossAttentionBlock(hidden_dim, heads, edge_dim, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention)
		self.feedforward = ResidualFeedForward(hidden_dim, dropout=dropout)
		self.edge_dim = edge_dim
		if self.edge_dim != None:
			# self.self_edge_nn = Sequential(Dropout(dropout), LayerNorm(edge_dim + 2 * hidden_dim), Linear(edge_dim + 2 * hidden_dim, edge_dim), LeakyReLU(), Dropout(dropout), Linear(edge_dim, edge_dim))
			self.cross_edge_nn = Sequential(Dropout(dropout), LayerNorm(edge_dim + hidden_dim + hidden_dim), Linear(edge_dim + hidden_dim + hidden_dim, edge_dim), LeakyReLU(), Dropout(dropout), Linear(edge_dim, edge_dim))
			self.cross_edge_gate = Sequential(Dropout(dropout), Linear(edge_dim + hidden_dim + hidden_dim, hidden_dim), Sigmoid())
			self.cross_edge_gate.apply(gate_linear_init_weights) # w = 0, b = 1
		self.get_alpha = self.cross_attn.get_alpha
		self.get_phi_message_norm = self.cross_attn.get_phi_message_norm

	def forward(self, x_source, p_source, x_out, p_out, self_edge_index, self_edge_attr, cross_edge_index, cross_edge_attr):
		x_out, p_out = self.self_attn(x_out, p_out, self_edge_index, self_edge_attr)
		x_out, p_out = self.cross_attn(x_source, p_source, x_out, p_out, cross_edge_index, cross_edge_attr)
		x_out = self.feedforward(x_out)
		if self.edge_dim != None:
			x_i, x_j = x_source[cross_edge_index[0]], x_out[cross_edge_index[1]]
			pair_and_edge = torch.cat([cross_edge_attr, x_i, x_j], dim=-1)
			g = self.cross_edge_gate(pair_and_edge)
			cross_edge_attr = cross_edge_attr + g * self.cross_edge_nn(pair_and_edge)
			# x_i, x_j = x_out[self_edge_index[0]], x_out[self_edge_index[1]]
			# self_edge_attr = self_edge_attr + self.self_edge_nn(torch.cat([self_edge_attr, x_i, x_j], dim=-1))
		return x_out, p_out, cross_edge_attr

class GraphMLP(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, dropout, normalize_input):
		super().__init__()
		if normalize_input:
			self.norm_input = LayerNorm(in_dim)
		else:
			self.norm_input = torch.nn.Identity(in_dim)
		self.linear1 = Linear(in_dim, hidden_dim)
		self.act1 = LeakyReLU()
		self.norm2 = LayerNorm(hidden_dim)
		self.dropout2 = Dropout(dropout)
		self.linear2 = Linear(hidden_dim, out_dim)
	def forward(self, x):
		x = self.linear1(x)
		x = self.act1(x)
		x = self.norm2(x)
		x = self.dropout2(x)
		x = self.linear2(x)
		return x

class TransformerEncoder(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, edge_dim=None, num_convs=6, heads=8, dropout=0., recycle=0, geometric=False, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.num_convs = num_convs
		self.heads = heads
		self.recycle = recycle
		self.geometric = geometric
		self.uniform_attention = uniform_attention
		self.input_embedding = GraphMLP(in_dim, hidden_dim, hidden_dim, dropout, normalize_input=False)
		self.edge_dim = edge_dim
		self.d_space = d_space
		if edge_dim != None:
			self.edge_embedding = Sequential(Dropout(dropout), Linear(edge_dim, hidden_dim), LeakyReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim))
			edge_dim = hidden_dim
		self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(hidden_dim=hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention) for i in range(self.num_convs)])

	def forward(self, x, edge_index, edge_attr, p=None):
		if self.geometric:
			assert p != None and p.shape[-1] == self.d_space, 'p should be a (-1, d_space) tensor'
		x = self.input_embedding(x)
		if self.edge_dim != None:
			edge_attr = self.edge_embedding(edge_attr)
		for _ in range(self.recycle + 1):
			for encoder_block in self.encoder_blocks:
				x, p, edge_attr = encoder_block(x, p, edge_index, edge_attr)
		if self.edge_dim:
			return x, p, edge_attr
		return x, p

class TransformerDecoder(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, edge_dim=None, num_convs=6, heads=8, dropout=0., geometric=False, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.num_convs = num_convs
		self.output_embedding = GraphMLP(in_dim, hidden_dim, hidden_dim, dropout, normalize_input=True)
		self.edge_dim = edge_dim
		self.geometric = geometric
		self.update_p = update_p
		self.d_space = d_space
		self.uniform_attention = uniform_attention
		if edge_dim != None:
			self.edge_embedding = Sequential(Dropout(dropout), Linear(edge_dim, hidden_dim), LeakyReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim))
			edge_dim = hidden_dim
		self.decoder_blocks = nn.ModuleList([TransformerDecoderBlock(hidden_dim=hidden_dim, edge_dim=edge_dim, heads=heads, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention) for i in range(self.num_convs)])
	
	def get_alpha(self):
		alphas = [block.get_alpha() for block in self.decoder_blocks]
		alpha = torch.stack(alphas, dim=1)
		return alpha # (|E|, L, H)

	def get_phi_message_norm(self):
		phi_message_norms = [block.get_phi_message_norm() for block in self.decoder_blocks]
		return torch.cat(phi_message_norms)
	
	def forward(self, x_source, x_out, self_edge_index, self_edge_attr, cross_edge_index, cross_edge_attr, p_source=None, p_out=None):
		self.y_intermediates = []
		if self.geometric:
			assert p_source != None and p_source.shape[-1] == self.d_space, 'p_source should be a (-1, d_space) tensor'
			assert p_out != None and p_out.shape[-1] == self.d_space, 'p_out should be a (-1, d_space) tensor'
		x_out = self.output_embedding(x_out)
		if self.edge_dim != None:
			cross_edge_attr = self.edge_embedding(cross_edge_attr)
		for decoder_block in self.decoder_blocks:
			# if self.geometric:
			# 	p_out = p_out.detach() # stop grad
			x_out, p_out, cross_edge_attr = decoder_block(x_source, p_source, x_out, p_out, self_edge_index, self_edge_attr, cross_edge_index, cross_edge_attr)
			# record intermediate predictions
			if self.geometric:
				y_pred = utils.format_prediction(x_out[:, :2], p_out)
				self.y_intermediates.append(y_pred) #(B*N, 5)
		if self.edge_dim:
			return x_out, p_out, cross_edge_attr
		return x_out, p_out

#################### Equivariant Graph Attention ####################

import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class CovariantGraphAttention(MessagePassing):
	def __init__(self, hidden_dim: int,
				 heads: int = 1, gate: bool = True,
				 dropout: float = 0., edge_dim: Optional[int] = None,
				 bias: bool = True, geometric=False, d_space=3, update_p=False, uniform_attention=False, **kwargs):
		kwargs.setdefault('aggr', 'add')
		super().__init__(node_dim=0, **kwargs)

		self.hidden_dim = hidden_dim
		self.heads = heads
		self.gate = gate
		self.dropout = dropout
		self.edge_dim = edge_dim
		self.geometric = geometric
		self.d_space = d_space
		self.update_p = update_p
		self._alpha = None
		self._phi_message_norm = None
		self.pi = 3.1415926
		self.uniform_attention = uniform_attention

		if self.geometric:
			self.geometric_encoder = Sequential(Linear(d_space, hidden_dim), LeakyReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim))
			self.geometric_decoder = Sequential(Linear(hidden_dim, hidden_dim), LeakyReLU(), Dropout(dropout), Linear(hidden_dim, d_space)) # consider tanh to keep norm of m_p[:, 1:] close to 1?
		
		self.lin_query = Linear(hidden_dim, hidden_dim, bias=False)
		self.lin_key = Linear(hidden_dim, hidden_dim, bias=False)
		self.lin_value = Linear(hidden_dim, hidden_dim, bias=False)
		
		if edge_dim is not None:
			self.lin_edge = Sequential(LayerNorm(edge_dim), Linear(edge_dim, heads, bias=False))
		else:
			self.lin_edge = self.register_parameter('lin_edge', None)
		if self.gate:
			self.gate_nn = Sequential(Dropout(dropout), Linear(2 * hidden_dim, hidden_dim), Sigmoid())
			if self.geometric:
				self.gate_nn_space = Sequential(Dropout(dropout), Linear(2 * hidden_dim, d_space), Sigmoid())

		self.reset_parameters()

	def reset_parameters(self):
		self.lin_key.reset_parameters()
		self.lin_query.reset_parameters()
		self.lin_value.reset_parameters()
		if self.edge_dim:
			self.lin_edge.reset_parameters()
		if self.geometric:
			self.geometric_decoder[-1].apply(zero_init_weights) # updates to 4-vectors are initialized to identity
		if self.gate:
			self.gate_nn.apply(gate_linear_init_weights) # w = 0, b = 1
			if self.geometric:
				self.gate_nn_space.apply(gate_linear_init_weights) # w = 0, b = 1

	def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
				edge_attr: OptTensor = None):
	
		if isinstance(x, Tensor):
			in_feat: PairTensor = (x, x)
		else:
			in_feat = x
		if self.geometric:
			assert in_feat[1].shape[1] == in_feat[0].shape[1] == self.hidden_dim + self.d_space, 'Geometric mode requires x to be (N, d_space + hidden_dim, D)'

		# propagate_type: (x: PairTensor, edge_attr: OptTensor)
		m = self.propagate(edge_index, x=in_feat, edge_attr=edge_attr, size=None) # aggreagated messgaes 
		m = m.contiguous().view(-1, self.hidden_dim) # (|E|, D)
		if self.geometric:
			x = in_feat[1][:, self.d_space:] # initial target feat
			p = in_feat[1][:, :self.d_space] # initial target feat
			m_x = m
			m_p = self.geometric_decoder(m)
		else:
			x = in_feat[1] # initial target feat
			p = None
			m_x = m
			m_p = None

		if self.gate:
			m_x = self.gate_nn(torch.cat([x, m_x], dim=-1)) * m_x
			if self.geometric:
				m_p = self.gate_nn_space(torch.cat([x, m_x], dim=-1)) * m_p

		if self.geometric and self.update_p:
			# bias the update towards identity
			m_p = self.normalize_phi_vec(m_p + torch.FloatTensor([0, 1, 0]).view(-1, 3).to(m_p.device), store_norm=True) # m_p = [deta, vec(dphi)]
			eta = p[:, 0] + m_p[:, 0]
			phi = self.rotate(p[:, 1:], m_p[:, 1:], inverse=False)
			p = torch.cat([eta.view(-1, 1), phi], dim=-1)
			# normalize again to prevent accumulation of numerical error
			p = self.normalize_phi_vec(p)
		return m_x, p

	def normalize_phi_vec(self, p, eps=1e-8, store_norm=False):
		phi_vec = p[:, 1:]
		phi_message_norm = phi_vec.norm(dim=-1, keepdim=True)
		if store_norm:
			self._phi_message_norm = phi_message_norm.view(-1)
		phi_vec = phi_vec / (phi_message_norm + eps)
		return torch.cat([p[:, 0].view(-1, 1), phi_vec], dim=-1)


	def to_angle(self, phi):
		# return phi in [-pi, pi]
		return (phi + self.pi).remainder(2*self.pi) - self.pi 

	def rotation_from_vec(self, v):
		""" 
		v = [c, s] # (B, 2)
		R = [[c, -s]
			 [s, c]]
		"""
		c = v[..., 0] # (B, )
		s = v[..., 1] # (B, )
		r = torch.stack([c, -s, s, c], dim=-1) # (B, 4)
		r = r.reshape(-1, 2, 2) # (B, 2, 2)
		return r

	def rotate(self, phi2, phi1, inverse=False):
		# phi : (B, 2)
		r = self.rotation_from_vec(phi1) # (B, 2, 2)
		if inverse:
			r = torch.transpose(r, -1, -2) # transpose to get the inverse (B, 2, 2)
		phi21 = torch.einsum('...ij, ...j', r, phi2) # rotate phi2 by -phi1
		return phi21

	def boost(self, p2, p1):
		# boost p2 into p1's frame
		# p = [eta, cos(phi), sin(phi)], (cos(phi), sin(phi)) is a unit vector representing the phi-angle
		eta21 = p2[..., 0] - p1[..., 0]
		phi1, phi2 = p1[..., 1:], p2[..., 1:] 
		phi21 = self.rotate(phi2, phi1, inverse=True)
		p21 = torch.cat([eta21.view(-1, 1), phi21], dim=-1)
		return p21

	def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor,
				index: Tensor, ptr: OptTensor,
				size_i: Optional[int]) -> Tensor:

		if self.geometric:
			p_i = x_i[:, :self.d_space] # (|E|, d_space)
			p_j = x_j[:, :self.d_space] # (|E|, d_space)
			x_i = x_i[:, self.d_space:] # (|E|, d_node)
			x_j = x_j[:, self.d_space:] # (|E|, d_node)
		query = self.lin_query(x_i).view(-1, self.heads, self.hidden_dim // self.heads)
		if self.geometric:
			# boost into i's frame
			relative_p = self.boost(p_j, p_i)
			relative_p = self.geometric_encoder(relative_p)
			key = self.lin_key(x_j + relative_p).view(-1, self.heads, self.hidden_dim // self.heads)
			value = self.lin_value(x_j + relative_p).view(-1, self.heads, self.hidden_dim // self.heads)
		else:
			key = self.lin_key(x_j).view(-1, self.heads, self.hidden_dim // self.heads)
			value = self.lin_value(x_j).view(-1, self.heads, self.hidden_dim // self.heads)

		edge_bias = 0
		if self.lin_edge is not None:
			assert edge_attr is not None
			edge_bias = self.lin_edge(edge_attr).view(-1, self.heads) # (|E|, H)
			# key = key + edge_attr
			# value = value + edge_attr
		if not self.uniform_attention:
			alpha = (query * key).sum(dim=-1) / math.sqrt(self.hidden_dim // self.heads) + edge_bias
		else:
			alpha = torch.zeros(query.shape[0], query.shape[1]).to(query.device)
		alpha = softmax(alpha, index, ptr, size_i)
		self._alpha = alpha

		m = value * alpha.view(-1, self.heads, 1) # invariant messages (|E|, H, D // H)
		return m

	def __repr__(self):
		return '{}({}, {}, heads={})'.format(self.__class__.__name__,
											 self.hidden_dim,
											 self.hidden_dim, self.heads)
