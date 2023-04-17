import torch
import numpy as np
import torch
from torch_geometric.data import Data, Batch

dataset_id = {
	't_schan': 0,
	't_tchan': 1,
	'ttbar': 2,
	'ttH': 3,
	'ttH_applied': 3,
	'ttyy_had': 4,
	'ttyy_lep': 5,
	'ttt': 6,
	'tttt': 7,
	'ttW': 8,
	'tHjb': 9,
	'ttH_odd': 10,
}

def to_torch(data, pad_y_up_to, drop_one_hot=None, test=False, train_with_xy_graph=True, detector_x=False, detector_y=False):
	# if training with xy graph, during training, graph will be an xy graph with output padding but correct number of target nodes
	data = make_graph(data, pad_y_up_to, drop_one_hot, test, train_with_xy_graph)
	y = torch.FloatTensor(data['y'])
	num_target = y.shape[0]
	x = {'num_target': num_target, 'graph': data['graph']}
	truth_matched = torch.LongTensor(data['truth_matched'])
	W_decay_pid = torch.LongTensor(data['W_decay_pid'])
	if W_decay_pid.shape[0] != y.shape[0]:
		W_decay_pid = torch.zeros(y.shape[0])
	reco_top = torch.FloatTensor(data['reco_top'])
	if 'gnn_reco_top' in data:
		gnn_reco_top = torch.FloatTensor(data['gnn_reco_top'])
	else:
		gnn_reco_top = torch.zeros_like(reco_top)
	reco_triplet_indices = torch.LongTensor(data['reco_triplet_indices'])
	if detector_y:
		y = xyze_to_detector(y)
		reco_top = xyze_to_detector(reco_top, warn_nan=False)
		gnn_reco_top = xyze_to_detector(gnn_reco_top, warn_nan=False)
	else:
		y = xyze_to_xyzm(y)
		reco_top = xyze_to_xyzm(reco_top, warn_nan=False)
		gnn_reco_top = xyze_to_xyzm(gnn_reco_top, warn_nan=False)
	if detector_x:
		x['graph'].x = torch.cat([xyze_to_detector(x['graph'].x[:, :4]), x['graph'].x[:, 4:]], dim=-1)
	if 'identified' in data:
		identified = torch.LongTensor(data['identified'])
	else:
		identified = torch.zeros_like(truth_matched)
	if 'gnn_predicted' in data:
		gnn_predicted = torch.LongTensor(data['gnn_predicted'])
	else:
		gnn_predicted = torch.zeros_like(truth_matched)
	if pad_y_up_to >= 0:
		y_pad = torch.zeros([pad_y_up_to - y.shape[0], y.shape[1]])
		indices_pad = -torch.ones([pad_y_up_to - y.shape[0], reco_triplet_indices.shape[1]]).long()
		truth_matched_pad = torch.zeros([pad_y_up_to - y.shape[0]])
		y = torch.cat([y, y_pad])
		reco_triplet_indices = torch.cat([reco_triplet_indices, indices_pad])
		reco_top = torch.cat([reco_top, y_pad])
		gnn_reco_top = torch.cat([gnn_reco_top, y_pad])
		gnn_predicted = torch.cat([gnn_predicted, truth_matched_pad])
		truth_matched = torch.cat([truth_matched, truth_matched_pad])
		identified = torch.cat([identified, truth_matched_pad])
		W_decay_pid = torch.cat([W_decay_pid, truth_matched_pad])
	y = {'momenta': y, 'num_target': num_target}
	if test:
		dsid = dataset_id[data['dataset'].split('/')[-1]]
		return {
				'x': x, # input: graph, num_target
				'y': y, # label: (padded) truth top kinematics
				'num_target': num_target,
				'truth_matched': truth_matched,
				'identified': identified, 
				'reco_top': reco_top, 
				'gnn_predicted': gnn_predicted,
				'gnn_reco_top': gnn_reco_top,
				# 'min_dR_candidate_top': torch.FloatTensor(data['min_dR_candidate_top']), 
				# 'is_valid_candidate': torch.LongTensor(data['is_valid_candidate']), 
				'info': torch.LongTensor(data['info'] + [dsid]),
				'W_decay_pid': W_decay_pid,
				'reco_triplet_indices': reco_triplet_indices,
		}
	else:
		return {
				'x': x,
				'y': y,
				'num_target': num_target,
				# 'truth_matched': truth_matched,
		}

def get_to_torch(pad_y_up_to, detector_x=False, detector_y=False, test=False):
	print(f'Padding up to {pad_y_up_to} outputs')
	if detector_x:
		print('X: (pT, y, phi, m)')
	if detector_y:
		print('Y: (pT, y, phi, m)')
	return lambda data: to_torch(data, pad_y_up_to=pad_y_up_to, detector_x=detector_x, detector_y=detector_y, test=test)

def make_graph(data, max_num_output, drop_one_hot, test, train_with_xy_graph, device=None):
	node_features, edge_tuples, edge_features = data['x']
	node_features = torch.FloatTensor(node_features)
	edge_tuples = torch.LongTensor(edge_tuples)
	if drop_one_hot != None:
		dropped = torch.arange(node_features.shape[0])[node_features[:, drop_one_hot] == 1]
		# node_features = node_features[node_features[:, drop_one_hot] != 1]
		# instead of dropping the nodes, we just disconnect them from the graph
		for index in dropped:
			sender_is_dropped = edge_tuples[0] == index
			receiver_is_dropped = edge_tuples[1] == index
			edge_is_kept = (sender_is_dropped + sender_is_dropped) == 0
			edge_tuples = edge_tuples.t()[edge_is_kept].t()
	# edge_features = torch.FloatTensor(edge_features).view(-1, 1)
	if test or not train_with_xy_graph:
		graph = Data(node_features, edge_tuples) #, edge_features)
		data['graph'] = graph
	else:
		max_source_nodes = 40
		max_cross_attn_edges = max_num_output * max_source_nodes
		max_self_attn_edges = max_num_output * max_num_output
		cross_attn_edges = torch.LongTensor(data['output_edges'][0])
		self_attn_edges = torch.LongTensor(data['output_edges'][1])
		if max_cross_attn_edges - cross_attn_edges.shape[1] >= 0:
			cross_attn_edges_pad = -1e6 * torch.ones([2, max_cross_attn_edges - cross_attn_edges.shape[1]])
			cross_attn_edges = torch.cat([cross_attn_edges, cross_attn_edges_pad], dim=1).long()
		else:
			cross_attn_edges = cross_attn_edges[:, :max_cross_attn_edges]
		self_attn_edges_pad = -1e6 * torch.ones([2, max_self_attn_edges - self_attn_edges.shape[1]])
		self_attn_edges = torch.cat([self_attn_edges, self_attn_edges_pad], dim=1).long()
		graph = Data(node_features, edge_tuples) #, edge_features)
		graph.cross_attn_edges = cross_attn_edges
		graph.self_attn_edges = self_attn_edges
		graph.n_x = node_features.shape[0]
		graph.max_cross_attn_edges = max_cross_attn_edges
		graph.max_self_attn_edges = max_self_attn_edges
		data['graph'] = graph
	return data

def split_dataset_rand(D):
	n_train = int(len(D) * 0.8)
	n_val = int(len(D) * 0.1)
	n_test = len(D) - n_train - n_val
	return torch.utils.data.random_split(D, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

def split_dataset(D, name, max_train_event=None, max_val_event=None, max_test_event=None):
	number = torch.load(f'{name}/number.pt')
	number = np.array(number)
	idx = np.arange(number.shape[0])
	train_idx = idx[number % 4 < 3][:max_train_event]
	val_idx = idx[number % 8 == 3][:max_val_event]
	test_idx = idx[number % 8 == 7][:max_test_event]
	return torch.utils.data.Subset(D, train_idx), torch.utils.data.Subset(D, val_idx), torch.utils.data.Subset(D, test_idx)

def get_xy_graph(input_graph, num_target, max_num_output, device=None):
	if not isinstance(input_graph, Batch):
		node_features, batch = input_graph.x, torch.zeros(input_graph.x.shape[0])
	else:
		node_features, batch = input_graph.x, input_graph.batch
	num_batches = int((torch.max(batch) + 1).item())
	# create cross attention edges between output nodes and input nodes
	source_nodes = torch.arange(batch.size(0), device=batch.device)
	cross_attn_edge_tensors = []
	for offset in range(max_num_output):
		output_nodes = max_num_output * batch + offset
		cross_attn_edge_tensors.append(torch.stack([source_nodes, output_nodes]))
	cross_attn_edges = torch.cat(cross_attn_edge_tensors, dim=1).long()
	# create self attention edges among output nodes
	self_attn_edge_tensors = [
		max_num_output * b +
		torch.stack([
				i * torch.ones(num_target[b] - 1, device=batch.device), 
				torch.cat([torch.arange(0, i, device=batch.device), torch.arange(i + 1, num_target[b], device=batch.device)])
			], dim=0)
		for b in range(num_batches) for i in range(num_target[b])
	]
	self_attn_edges = torch.cat(self_attn_edge_tensors, dim=1).long()
	if device == None:
		device = node_features.device
	return XYData(node_features, torch.zeros([num_batches * max_num_output, 4]).to(device), input_graph.edge_index, self_attn_edges.to(device), cross_attn_edges.to(device))

class XYData(Data):
	def __init__(self, x, y, x_edge_index, y_edge_index, xy_edge_index):
		super().__init__()
		self.x_in = x
		self.x_out = y
		self.edge_index_in = x_edge_index
		self.edge_index_out = y_edge_index
		self.edge_index_cross = xy_edge_index
	def __inc__(self, key, value):
		if 'edge_index_cross' == key:
			return torch.tensor([[self.x_in.size(0)], [self.x_out.size(0)]])
		elif 'edge_index_in' == key:
			return self.x_in.size(0)
		elif 'edge_index_out' == key:
			return self.x_out.size(0)
		else:
			return super().__inc__(key, value)

def get_plot_configs(detector):
	if detector:
		hist_config = {
			"alpha": 0.8,
			"lw": 2,
			'histtype': 'step',
		}
		range_config = [
			dict([("bins",80), ("range",(0, 800))]),
			dict([("bins",80), ("range",(-3, 3))]),
			dict([("bins",80), ("range",(-3.15, 3.15))]),
			dict([("bins",80), ("range",(100, 300))]),
		]
		dim_labels=['$p_T$', '$y$', '$\phi$', '$m$']    
	else:
		hist_config = {
			"alpha": 0.8,
			"lw": 2,
			'histtype': 'step',
		}
		range_config = [
			dict([("bins",80), ("range",(-800, 800))]),
			dict([("bins",80), ("range",(-800, 800))]),
			dict([("bins",80), ("range",(-2000, 2000))]),
			dict([("bins",80), ("range",(100, 300))]),
		]
		dim_labels = ['$p_x$', '$p_y$', '$p_z$', '$m$']
	return dim_labels, hist_config, range_config

def get_reference_std(detector):
	if detector:
		return torch.FloatTensor([100, 1, 1, 5]) # a rough estimate of std
	else:
		return torch.FloatTensor([150, 150, 450, 5]) # a rough estimate of std

def get_ghost_value():
	return -1e6


def xyzm_to_detector(y):
	px, py, pz, m = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
	E = (y ** 2).sum(-1).sqrt()
	pT2 = px**2+py**2
	p2 = pT2 + pz ** 2
	pT = torch.sqrt(pT2)
	r = (E+pz)/(E-pz)
	rapdity = 1/2 * torch.log(r)
	isNan = torch.isnan(rapdity)
	if isNan.sum():
		print(f'Found {isNan.sum()} nans in energy')
	if isNan.sum():
		print(f'Found {isNan.sum()} nans in rapdity')
	isNan = torch.isnan(E)
	phi = torch.atan2(py,px)
	y = torch.stack([pT, rapdity, phi, m], axis=-1)
	return y

def xyze_to_xyzm(y, warn_nan=True):
	px, py, pz, E = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
	m2 = E**2 - (px ** 2 + py ** 2 + pz ** 2)
	m = m2.sqrt()
	isNan = torch.isnan(m)
	if isNan.sum() and warn_nan:
		print(f'Found {isNan.sum()} nans in mass')
	y = torch.stack([px, py, pz, m], axis=-1)
	return y

def xyze_to_detector(y, warn_nan=True):
	px, py, pz, E = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
	pT2 = px**2+py**2
	p2 = pT2 + pz ** 2
	pT = torch.sqrt(pT2)
	r = (E+pz)/(E-pz)
	rapdity = 1/2 * torch.log(r)
	isNan = torch.isnan(rapdity)
	if isNan.sum() and warn_nan:
		print(f'Found {isNan.sum()} nans in rapdity')
	phi = torch.atan2(py,px)
	m2 = E**2-p2
	m = torch.sqrt(m2.clip(0))
	y = torch.stack([pT, rapdity, phi, m], axis=-1)
	return y

def xyze_to_detector_np(y):
	px, py, pz, E = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
	E = np.maximum(E, 0)
	pz = np.clip(pz, -E, E)
	pT2 = px**2+py**2
	p2 = pT2 + pz ** 2
	pT = np.sqrt(pT2)
	r = (E+pz)/(E-pz)
	rapdity = 1/2 * np.log(r)
	isNan = np.isnan(rapdity)
	if isNan.sum():
		print(f'Found {isNan.sum()} nans in rapdity')
	phi = np.arctan2(py,px)
	m2 = E**2-p2
	m = np.sqrt(m2)
	return np.stack([pT, rapdity, phi, m], axis=-1)

def format_prediction(scaled_pt_m_pred, eta_phi_pred, mass=173):
	pt_m_pred = torch.FloatTensor([100, 5]).to(scaled_pt_m_pred.device) * scaled_pt_m_pred
	# target is in [pT, eta, phi_vec, m]
	y_pred = torch.cat([pt_m_pred[..., 0].unsqueeze(-1), eta_phi_pred, pt_m_pred[..., 1].unsqueeze(-1)], dim=-1)
	y_pred = torch.cat([y_pred[..., :-1], y_pred[..., -1].unsqueeze(-1) + mass], dim=-1) # add 173 to mass
	return y_pred