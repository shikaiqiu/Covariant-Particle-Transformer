from setformer import *
from torch_geometric.nn import GCNConv, global_add_pool


def build_mlp(hidden_dim, out_dim, num_hidden_layers=1, normalize_input=True):
	net = []
	if normalize_input:
		net.append(LayerNorm(hidden_dim))
	for _ in range(num_hidden_layers):
		net.append(Linear(hidden_dim, hidden_dim))
		net.append(LeakyReLU())
	net.append(Linear(hidden_dim, out_dim))
	return Sequential(*net)

class GCN(CovariantTopFormer):
	def __init__(self, in_dim, hidden_dim, out_dim, max_num_output, output_dir, use_gpu=True, lr=1e-4, schedule_lr=False, num_convs=(3, 0), heads=8, dist_scale=1, beta=0.5, dropout=0., match_scale_factor=1, p_norm=2, mass=172.76, num_mlp_hidden_layer=2, vectorize_phi=False):
		assert num_convs[1] == 0, 'GCN has no decoder, set num_convs to (*, 0).'
		self.num_mlp_hidden_layer = num_mlp_hidden_layer
		self.vectorize_phi = vectorize_phi
		self.out_dim = 5 if vectorize_phi else 4 # (pT, eta, cos(phi), sin(phi), m) or (px, py, eta, m)
		super().__init__(in_dim, hidden_dim, out_dim, max_num_output, output_dir, use_gpu, lr, schedule_lr, num_convs, heads, dist_scale, beta, dropout, match_scale_factor, p_norm, mass, geometric=True)

	def define_modules(self):	
		self.ebm = Linear(self.in_dim + 1, self.hidden_dim) # +1 due to vector rep of phi
		self.convs = nn.ModuleList([GCNConv(self.hidden_dim, self.hidden_dim) for i in range(self.num_convs[0])])
		self.norm_convs = nn.ModuleList([LayerNorm(self.hidden_dim) for i in range(self.num_convs[0])])
		self.pool = global_add_pool
		# output: pT, eta, cos(phi), sin(phi), m
		self.out_mlp = build_mlp(self.hidden_dim, self.max_num_output * self.out_dim, self.num_mlp_hidden_layer)
		# self.count_logits_nn = Sequential(LayerNorm(self.hidden_dim), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Linear(self.hidden_dim, self.max_num_output))

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
		x = self.ebm(x)
		for conv, norm in zip(self.convs, self.norm_convs):
			x = x + F.relu(conv(norm(x), edge_index))
		y = self.pool(x, batch)
		############ decoder ############
		# output: pT, eta, cos(phi), sin(phi), m
		y = self.out_mlp(y) # (B, max_num_output * 5)
		y = y.reshape(-1, self.max_num_output, self.out_dim) # (B, max_num_output, 5)
		if self.vectorize_phi:
			# normalize
			phi_normalized = y[..., 2:4] / (y[..., 2:4].norm(dim=-1).view(-1, self.max_num_output, 1) + 1e-5)
			# should return in [pT, eta, phi_vec, m]
			y = torch.cat([y[..., :2], phi_normalized, y[..., -1].view(-1, self.max_num_output, 1)], dim=-1)
			# scale back pT and mass 
			y = torch.FloatTensor([100, 1, 1, 1, 5]).to(self.device) * y + torch.FloatTensor([0, 0, 0, 0, self.mass]).to(self.device) 
		else:
			y = torch.FloatTensor([100, 100, 1, 5]).to(self.device) * y + torch.FloatTensor([0, 0, 0, self.mass]).to(self.device) 
		# dummy logits, no impact
		logits = torch.ones(batch_size, self.max_num_output).to(self.device)
		# no intermediate pred
		self.y_intermediates = []
		return y, logits

	def vec_to_angle_pred(self, p):
		# no need for differentiability, just for dR-match and pull plot
		if self.vectorize_phi:
			return super().vec_to_angle(p)
		# (px, py, eta, m) -> (pT, eta, phi, m)
		px, py, eta, m = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
		pT = (px ** 2 + py ** 2).sqrt()
		new_p = torch.stack([pT, eta, torch.atan2(py, px), m], dim=-1) # (..., 4)
		return new_p

	def reparameterize_pred(self, p):
		# return in (px, py, eta, m) for loss calculation
		if self.vectorize_phi:
			return super().reparameterize_pred(p)
		return p


class DeepSets(CovariantTopFormer):
	def __init__(self, in_dim, hidden_dim, out_dim, max_num_output, output_dir, use_gpu=True, lr=1e-4, schedule_lr=False, num_convs=(3, 0), heads=8, dist_scale=1, beta=0.5, dropout=0., match_scale_factor=1, p_norm=2, mass=172.76, num_mlp_hidden_layer=2, vectorize_phi=False):
		assert num_convs[1] == 0, 'DeepSets has no decoder, set num_convs to (*, 0).'
		self.num_mlp_hidden_layer = num_mlp_hidden_layer
		self.vectorize_phi = vectorize_phi
		self.out_dim = 5 if vectorize_phi else 4 # (pT, eta, cos(phi), sin(phi), m) or (px, py, eta, m)
		super().__init__(in_dim, hidden_dim, out_dim, max_num_output, output_dir, use_gpu, lr, schedule_lr, num_convs, heads, dist_scale, beta, dropout, match_scale_factor, p_norm, mass, geometric=True)

	def define_modules(self):	
		self.ebm = Linear(self.in_dim + 1, self.hidden_dim) # +1 due to vector rep of phi
		self.encoder = build_mlp(self.hidden_dim, self.hidden_dim, self.num_convs[0], normalize_input=False)
		# self.norm_convs = nn.ModuleList([LayerNorm(hidden_dim) for i in range(self.num_convs)])
		self.pool = global_add_pool
		# output: pT, eta, cos(phi), sin(phi), m
		self.out_mlp = build_mlp(self.hidden_dim, self.max_num_output * self.out_dim, self.num_mlp_hidden_layer)
		# self.count_logits_nn = Sequential(LayerNorm(self.hidden_dim), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Linear(self.hidden_dim, self.max_num_output))

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
		x = self.ebm(x)
		x = self.encoder(x)
		y = self.pool(x, batch)
		############ decoder ############
		# output: pT, eta, cos(phi), sin(phi), m
		y = self.out_mlp(y) # (B, max_num_output * 5)
		y = y.reshape(-1, self.max_num_output, self.out_dim) # (B, max_num_output, 5)
		if self.vectorize_phi:
			# normalize
			phi_normalized = y[..., 2:4] / (y[..., 2:4].norm(dim=-1).view(-1, self.max_num_output, 1) + 1e-5)
			# should return in [pT, eta, phi_vec, m]
			y = torch.cat([y[..., :2], phi_normalized, y[..., -1].view(-1, self.max_num_output, 1)], dim=-1)
			# scale back pT and mass 
			y = torch.FloatTensor([100, 1, 1, 1, 5]).to(self.device) * y + torch.FloatTensor([0, 0, 0, 0, self.mass]).to(self.device) 
		else:
			y = torch.FloatTensor([100, 100, 1, 5]).to(self.device) * y + torch.FloatTensor([0, 0, 0, self.mass]).to(self.device) 
		# dummy logits, no impact
		logits = torch.ones(batch_size, self.max_num_output).to(self.device)
		# no intermediate pred
		self.y_intermediates = []
		return y, logits

	def vec_to_angle_pred(self, p):
		# no need for differentiability, just for dR-match and pull plot
		if self.vectorize_phi:
			return super().vec_to_angle(p)
		# (px, py, eta, m) -> (pT, eta, phi, m)
		px, py, eta, m = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
		pT = (px ** 2 + py ** 2).sqrt()
		new_p = torch.stack([pT, eta, torch.atan2(py, px), m], dim=-1) # (..., 4)
		return new_p

	def reparameterize_pred(self, p):
		# return in (px, py, eta, m) for loss calculation
		if self.vectorize_phi:
			return super().reparameterize_pred(p)
		return p