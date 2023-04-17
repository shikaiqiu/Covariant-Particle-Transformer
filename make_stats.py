#@title generate test statistics
import os
import os.path as osp
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'medium',
		  'figure.figsize': (8, 8),
		 'axes.labelsize': 'medium',
		 'axes.titlesize':'medium',
		 'xtick.labelsize':'medium',
		 'ytick.labelsize':'medium',
		  'figure.dpi': 100}
pylab.rcParams.update(params)

from matplotlib import cm
from seaborn import heatmap
import seaborn as sns
sns.set_style("white")
sns.axes_style("white")
import shutil
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)
import utils


def run(test_result, output_dir, max_num_output, bins, entries_per_bin):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	test_result['truth_matched'] = test_result['truth_matched'] == 1
	test_result['attention_matched'] = test_result['attention_matched'] == 1
	truth_matched = test_result['truth_matched']
	attention_matched = test_result['attention_matched']
	info = np.repeat(test_result['info'], repeats=max_num_output, axis=0).astype(int)
	N_genjet = info[:, 0]
	N_genlep = info[:, 1]
	# Number = info[:, 2]
	N_gen_triplet = info[:, 3]
	N_top = info[:, 4]
	N_hadtop = info[:, 5]
	dsid = info[:, 6]

	W_decay_pid = test_result['W_decay_pid'].astype(int)
	is_hadronic = ((W_decay_pid != 0) * (np.abs(W_decay_pid) < 10)).reshape(-1)
	is_leptonic = ((W_decay_pid != 0) * (np.abs(W_decay_pid) >= 10)).reshape(-1)

	y_target = to_detector(test_result['y_target'], never_mind=True)
	y_reco = to_detector(test_result['y_reco'], never_mind=True)
	y_gnn_reco = to_detector(test_result['y_gnn_reco'], never_mind=True)
	y_pred = to_detector(test_result['y_pred'], never_mind=True)
	valid_entry_mask = get_valid_entry_mask(test_result, max_num_output)
	valid_entry_mask_keep_incorrect_num = get_valid_entry_mask(test_result, max_num_output, keep_incorrect_num_top=True)
	
	N_bj = np.repeat(test_result['N_bj'], repeats=max_num_output, axis=0).astype(int)
	N_j = info[:, 0]
	N_l = info[:, 1]
	trigger_cut = (((N_j >= 3) * (N_bj > 0) * (N_l == 0)) + ((N_bj > 0) * (N_l > 0))) > 0
	# print(trigger_cut.mean())

	cuts = {
		'Inclusive': True,
		'Truth-matched': truth_matched,
		'Not truth-matched': (1 - truth_matched) == 1, 
		#'Alt': ...
		'Attention-matched': attention_matched,
		'Hadronic': is_hadronic == 1,
		'Leptonic': is_leptonic == 1,
		# 'pT > 60': y_target[:, 0] > 60,
		# 'pT < 60': y_target[:, 0] < 60,
		# '<=2 jet': N_genjet <= 2,
		# '<=1 jet': N_genjet <= 1,
		# 'tm >=6 jet': (N_genjet >= 6) * truth_matched == 1,
		# '>=9 jet': N_genjet >= 9,
		# '>= 1 lepton': N_genlep >= 1,
		# '>= 2 lepton': N_genlep >= 2,
	}

	dataset_id = {
		't_schan': 0,
		't_tchan': 1,
		'tt': 2,
		'ttH': 3,
		# 'ttH_applied': 3,
		'ttyy_had': 4,
		'ttyy_lep': 5,
		'ttt': 6,
		'tttt': 7,
		'ttW': 8,
		'tHjb': 9,
		'ttH_odd': 10,
	}

	n_top = {
		't_s': 1,
		't_t': 1,
		'tt': 2,
		'ttH': 2,
		# 'ttH_applied': 2,
		'ttyy_had': 2,
		'ttyy_lep': 2,
		'ttt': 3,
		'tttt': 4,
		'ttW': 2,
		'tHjb': 1,
		'ttH_odd': 2,
	}

	for dataset in dataset_id:
		dataset_cut = dsid == dataset_id[dataset]
		dataset_cut = (dataset_cut * trigger_cut) == 1
		print('Applied trigger cut')
		if dataset_cut.sum() == 0:
			continue
		print(f'dataset: {dataset}')
		dataset_dir = osp.join(output_dir, dataset)
		if not osp.exists(dataset_dir):
			os.makedirs(dataset_dir)
		csv_path = osp.join(output_dir, dataset, 'results.csv')
		with open(csv_path, 'w') as csvfile:
			csvwriter = csv.writer(csvfile)    
			fields = [
				'Cut', 'Frac. tops', 'P(Attention-matched|Cut, TM)', 'N(bins)',
				'Accuracy(# top)',
				'Chi2(Reduced)', 'Chi2(Pt)', 'Chi2(Y)', 'Chi2(Phi)', 'Chi2(M)',
				'MedPull(Pt)', 'MedPull(Y)', 'MedPull(Phi)', 'MedPull(M)',
				'Resolution(Pt)', 'Resolution(Y)', 'Resolution(Phi)', 'Resolution(M)',
			]
			csvwriter.writerow(fields)
			for cut_name in cuts:
				cut = cuts[cut_name]
				print(f'cut: {cut_name}')
				if cut_name == 'Inclusive':
					pass_cut = dataset_cut * cut * valid_entry_mask_keep_incorrect_num # only to look at classification
					cut_dir = osp.join(dataset_dir, cut_name + '_n_top_keep_incorrect')
					if not osp.exists(cut_dir):
						os.makedirs(cut_dir)
					# making n_top prediction plots
					_ = get_global_metric( 
						y_target, 
						y_pred, 
						test_result['num_target'],
						test_result['num_pred'],
						test_result['probs'],
						pass_cut,
						cut_dir,
						max_num_output,
						name='prediction',
						detector=True,
					)

				pass_cut = dataset_cut * cut * valid_entry_mask # Always take valid entries only (non-placeholder + correct number) only. Note this only ensures y_pred and y_target is valid but not y_reco which still can be 0.
				cut_dir = osp.join(dataset_dir, cut_name)
				if not osp.exists(cut_dir):
					os.makedirs(cut_dir)
				if not osp.exists(cut_dir + '_vs_gnn'):
					os.makedirs(cut_dir + '_vs_gnn')
				# making pull plots, error plots, ...
				med_pull_pred, resolution_pred, acc_pred = get_global_metric( 
					y_target, 
					y_pred, 
					test_result['num_target'],
					test_result['num_pred'],
					test_result['probs'],
					pass_cut,
					cut_dir,
					max_num_output,
					name='prediction',
					detector=True,
				)
				# plot resolution and chi2, produce chi2 csvs
				num_tops, frac_attention_matched, num_bins, chi2 = get_binned_metric(
					y_target, y_pred, y_reco, test_result['num_pred'], test_result['num_target'], truth_matched, test_result['attention_matched'],
					pass_cut,
					cut_dir,
					max_num_output,
					bins,
					entries_per_bin if (cut_name == 'Truth-matched') else 1e10
				)
				row = [cut_name, num_tops/(dataset_cut.sum() * n_top[dataset] / max_num_output), frac_attention_matched, num_bins] \
					+ [acc_pred] \
					+ chi2 \
					+ med_pull_pred.tolist() \
					+ resolution_pred.tolist()
				csvwriter.writerow(row)

				if cut_name == 'Truth-matched' or 'jet' in cut_name:
					med_pull_reco, resolution_reco, _ = get_global_metric(
						y_target, 
						y_reco, 
						test_result['num_target'],
						test_result['num_pred'],
						test_result['probs'],
						pass_cut,
						cut_dir,
						max_num_output,
						name='reco',
						detector=True
					)
					row = [cut_name + '-triplet-reco', np.nan, np.nan, np.nan] \
						+ [np.nan] \
						+ [np.nan] * 5 \
						+ med_pull_reco.tolist() \
						+ resolution_reco.tolist()
					csvwriter.writerow(row)

					# make plots for pred vs gnn reco in a separate dir
					_, frac_gnn_reco_matched, _, _ = get_binned_metric(
						y_target, y_pred, y_gnn_reco, test_result['num_pred'], test_result['num_target'], truth_matched, test_result['gnn_reco_matched'],
						pass_cut,
						cut_dir + '_vs_gnn',
						max_num_output,
						bins,
						entries_per_bin if (cut_name == 'Truth-matched') else 1e10
					)
					med_pull_reco, resolution_reco, _ = get_global_metric(
						y_target, 
						y_gnn_reco, 
						test_result['num_target'],
						test_result['num_pred'],
						test_result['probs'],
						pass_cut,
						cut_dir,
						max_num_output,
						name='gnn_reco',
						detector=True
					)
					row = [cut_name + '-gnn-triplet', np.nan, frac_gnn_reco_matched, np.nan] \
						+ [np.nan] \
						+ [np.nan] * 5 \
						+ med_pull_reco.tolist() \
						+ resolution_reco.tolist()
					csvwriter.writerow(row)

			# collect results
			summary_dir = osp.join(dataset_dir, 'summary')
			if not osp.exists(summary_dir):
				os.makedirs(summary_dir)
			# shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'reco.png'), osp.join(summary_dir, 'tm_pull_reco.png'))
			# Absolute:
			shutil.copy(osp.join(dataset_dir, 'Inclusive', 'number.png'), osp.join(summary_dir, 'n_top.png'))
			shutil.copy(osp.join(dataset_dir, 'Inclusive', 'N_top_prob.png'), osp.join(summary_dir, 'n_top_prob.png'))
			shutil.copy(osp.join(dataset_dir, 'Inclusive', 'prediction.png'), osp.join(summary_dir, 'all_pull_pred.png'))
			shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'prediction.png'), osp.join(summary_dir, 'tm_pull_pred.png'))
			shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'reco.png'), osp.join(summary_dir, 'tm_pull_reco.png'))
			shutil.copy(osp.join(dataset_dir, 'Not truth-matched', 'prediction.png'), osp.join(summary_dir, 'ntm_pull_pred.png'))
			# shutil.copy(osp.join(dataset_dir, 'Hadronic', 'prediction.png'), osp.join(summary_dir, 'had_pull_pred.png'))
			# shutil.copy(osp.join(dataset_dir, 'Leptonic', 'prediction.png'), osp.join(summary_dir, 'lep_pull_pred.png'))
			# shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'chi2_distribution.png'), osp.join(summary_dir, 'tm_chi2_distribution.png'))
			# shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'chi2_reduced.png'), osp.join(summary_dir, 'tm_chi2_reduced.png'))
			# shutil.copy(osp.join(dataset_dir, 'tm >=6 jet', 'gnn_reco.png'), osp.join(summary_dir, 'tm6j_pull_gnn_reco.png'))
			# Relative
			shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'chi2_distribution.png'), osp.join(summary_dir, 'tm_chi2_distribution.png'))
			shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'chi2_reduced.png'), osp.join(summary_dir, 'tm_chi2_reduced.png'))
			shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'frac_rejected_pred_alt.png'), osp.join(summary_dir, 'tm_frac_rejected_pred_alt.png'))
			shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'frac_rejected_pred.png'), osp.join(summary_dir, 'tm_frac_rejected_pred.png'))
			shutil.copy(osp.join(dataset_dir, 'Truth-matched', 'frac_attention_matched.png'), osp.join(summary_dir, 'tm_frac_attention_matched.png'))

		# csv
		shutil.copy(csv_path, osp.join(summary_dir, 'results.csv'))


def get_binned_metric(y_target, y_pred, y_pred_alt, num_pred, num_target, truth_matched, attention_matched, precut, output_dir, max_num_output, bins, entries_per_bin):
	dim_labels = ['p_T', 'y', '\phi', 'm']
	fmts = ['.1f', '.2g', '.2g', '.1f']

	x, y, num_bins = compute_bins(y_target, precut, bins=bins, entries_per_bin=entries_per_bin)
	xlabels = [f'{x:.2f}' for x in x][:-1] # + ['$\infty$']
	ylabels = [f'{y:.2f}' for y in y][:-1] # + ['$\infty$']
	dx = np.diff(x)
	dy = np.diff(y)
	dx[-1] = 1e20
	dy[-1] = 1e20
	y = y[::-1]
	dy = dy[::-1]
	ylabels = ylabels[::-1]
	X, Y = np.meshgrid(x, y)
	# stats_array = [compute_stats(test_result, precut, p, dp, eta, deta, max_num_output, pull=False) for eta, deta in zip(y[1:], dy) for p, dp in zip(x[:-1], dx)]
	stats_array = compute_binned_stats(y_target, y_pred, y_pred_alt, num_pred, num_target, truth_matched, attention_matched, precut, x, y, dx, dy, max_num_output, pull=False)

	def plot_item(item, title, index=0, fmt='f'):
		fig = plt.figure(dpi=200)
		ax = fig.add_subplot(1, 1, 1)
		Z = np.array([s[item][index] if isinstance(s[item], np.ndarray) else s[item] for s in stats_array]).reshape([X.shape[0] - 1, X.shape[1] - 1])
		heatmap(Z, square=True, robust=True, ax=ax, xticklabels=xlabels, yticklabels=ylabels, annot=True, fmt=fmt)
		ax.set_xlabel('$p_T$')
		ax.set_ylabel('$|y|$')
		ax.set_title(title)

	def plot_resolution(item):
		fig = plt.figure(dpi=200)
		axs = []
		for i in range(4):
			axs.append(fig.add_subplot(2, 2, i + 1))
		for i in range(4):
			ax = axs[i]
			Z = np.array([s[item][i] for s in stats_array]).reshape([X.shape[0] - 1, X.shape[1] - 1])
			heatmap(Z, square=True, robust=True, ax=ax, xticklabels=xlabels, yticklabels=ylabels, annot=True, fmt=fmts[i])
			ax.set_xlabel('$p_T$')
			ax.set_ylabel('$|y|$')
			ax.set_title('$\sqrt{<\Delta ' + dim_labels[i] + '^2>}$' if 'N' not in dim_labels[i] else dim_labels[i])
		# ax.view_init(30, -60)

	def plot_chi2():
		fig = plt.figure(dpi=200)
		axs = []
		for i in range(4):
			axs.append(fig.add_subplot(2, 2, i + 1))
		for i in range(4):
			ax = axs[i]
			pred_error = np.array([s['rmse_pred'][i] for s in stats_array]).reshape([X.shape[0] - 1, X.shape[1] - 1])
			pred_alt_error = np.array([s['rmse_pred_alt'][i] for s in stats_array]).reshape([X.shape[0] - 1, X.shape[1] - 1])
			Z = (pred_error / pred_alt_error) ** 2
			heatmap(Z, square=True, cmap='seismic', robust=True, center=1, ax=ax, xticklabels=xlabels, yticklabels=ylabels, annot=True, fmt='.2g')
			ax.set_xlabel('$p_T$')
			ax.set_ylabel('$|y|$')
			# ax.set_title('${<\Delta ' + dim_labels[i] + '_{pred}^2>} / {<\Delta ' + dim_labels[i] + '_{pred_alt}^2>}$')
			ax.set_title('$\chi^2_{' + f'{dim_labels[i]}' + '}$')

	def plot_reduced_chi2():
		fig = plt.figure(dpi=200)
		ax = fig.add_subplot(1, 1, 1)
		pred_error = np.array([s['rmse_pred'] for s in stats_array])
		pred_alt_error = np.array([s['rmse_pred_alt'] for s in stats_array])
		Z = ((pred_error / pred_alt_error) ** 2).mean(-1)
		Z = Z.reshape([X.shape[0] - 1, X.shape[1] - 1])
		heatmap(Z, square=True, cmap='seismic', robust=True, center=1, ax=ax, xticklabels=xlabels, yticklabels=ylabels, annot=True, fmt='.2g')
		ax.set_xlabel('$p_T$')
		ax.set_ylabel('$|y|$')
		ax.set_title('$\chi^2_{reduced}$')

	def plot_chi2_distributions():
		hist_config = {
				"alpha": 0.8,
				"lw": 1,
				'histtype': 'step',
		}
		chi2s_pred = []
		chi2s_pred_alt = []
		for s in stats_array:
			pred_alt_error = np.array(s['rmse_pred_alt'])
			dy_pred = np.array(s['dy_pred'])
			dy_pred_alt = np.array(s['dy_pred_alt'])
			chi2s_pred.append((dy_pred / pred_alt_error) ** 2)
			chi2s_pred_alt.append((dy_pred_alt / pred_alt_error) ** 2)
		chi2s_pred = np.concatenate(chi2s_pred)
		chi2s_pred_alt = np.concatenate(chi2s_pred_alt)
		fig = plt.figure(dpi=200)
		axs = []
		for i in range(4):
			axs.append(fig.add_subplot(2, 2, i + 1))
		for i in range(4):
			ax = axs[i]
			pred_mean = np.mean(chi2s_pred[:, i])
			pred_alt_mean = np.mean(chi2s_pred_alt[:, i])
			pred_med = np.median(chi2s_pred[:, i])
			pred_alt_med = np.median(chi2s_pred_alt[:, i])
			ax.hist(chi2s_pred[:, i], density=True, label=r'$\mathcal{M}$: ' + f'mean = {pred_mean:.3g}, med = {pred_med:.3g}', bins=np.linspace(0, 2.5, 25), **hist_config)
			ax.hist(chi2s_pred_alt[:, i], density=True, label=r"$\mathcal{M}':$ " + f'mean = {pred_alt_mean:.3g}, med = {pred_alt_med:.3g}', bins=np.linspace(0, 2.5, 25), **hist_config)
			ax.set_xlabel('$\chi^2_{' + f'{dim_labels[i]}' + '}$')
			# ax.set_yscale('log')
			ax.legend()
		plt.savefig(f'{output_dir}/chi2_distribution.png')
		plt.close()
		fig = plt.figure(dpi=200)
		pred_mean = np.mean(chi2s_pred.mean(-1),)
		pred_alt_mean = np.mean(chi2s_pred_alt.mean(-1),)
		pred_med = np.median(chi2s_pred.mean(-1))
		pred_alt_med = np.median(chi2s_pred_alt.mean(-1))
		plt.hist(chi2s_pred.mean(-1), density=True, label=r'$\mathcal{M}:$ ' + f'mean = {pred_mean:.3g}, med = {pred_med:.3g}', bins=np.linspace(0, 4, 40), **hist_config)
		plt.hist(chi2s_pred_alt.mean(-1), density=True, label=r"$\mathcal{M}':$ " + f'mean = {pred_alt_mean:.3g}, med = {pred_alt_med:.3g}', bins=np.linspace(0, 4, 40), **hist_config)
		plt.xlabel('$\chi^2_{reduced}$')
		# plt.yscale('log')
		plt.legend()
		plt.savefig(f'{output_dir}/chi2_distribution_reduced.png')
		plt.close()
		chi2, chi2_reduced = np.mean(chi2s_pred, axis=0), np.mean(chi2s_pred.mean(-1))
		return [chi2_reduced] + chi2.tolist()

	sns.set(font_scale=0.3)
	plot_item('num_tops', '$N(p_T, y)$', fmt='d')
	plt.savefig(f'{output_dir}/num_tops.png')
	plt.close()

	plot_item('frac_rejected_pred_alt', '$P(\mathrm{rejected}|p_T, y)$', fmt='.2g')
	plt.savefig(f'{output_dir}/frac_rejected_pred_alt.png')
	plt.close()

	plot_item('frac_rejected_pred', '$P(\mathrm{rejected}|p_T, y)$', fmt='.2g')
	plt.savefig(f'{output_dir}/frac_rejected_pred.png')
	plt.close()

	plot_item('frac_attention_matched', '$P(\mathrm{attention-matched}|p_T, y)$', fmt='.2g')
	plt.savefig(f'{output_dir}/frac_attention_matched.png')
	plt.close()

	sns.set(font_scale=0.3)
	plot_resolution('rmse_pred_alt')
	plt.savefig(f'{output_dir}/pred_alt_resolution.png')
	plt.close()

	sns.set(font_scale=0.3)
	plot_resolution('rmse_pred')
	plt.savefig(f'{output_dir}/pred_resolution.png')
	plt.close()

	sns.set(font_scale=0.3)
	plot_chi2()
	plt.savefig(f'{output_dir}/chi2.png')
	plt.close()

	sns.set(font_scale=0.5)
	plot_reduced_chi2()
	plt.savefig(f'{output_dir}/chi2_reduced.png')
	plt.close()

	sns.set(font_scale=0.5)
	chi2 = plot_chi2_distributions() # plots saved inside function call
	num_tops = sum([s['num_tops'] for s in stats_array])
	frac_attention_matched = (sum([s['attention_matched'] for s in stats_array]) / sum([s['truth_matched'] for s in stats_array])) if sum([s['truth_matched'] for s in stats_array]) else 0
	return num_tops, frac_attention_matched, num_bins, chi2


def compute_bins(y_target, precut, bins=10, entries_per_bin=None):
	if entries_per_bin != None:
		total = precut.sum()
		bins = int((total / entries_per_bin) ** 0.5)
	bins = max(bins, 2)
	pT = y_target[:, 0][precut]
	eta = y_target[:, 1][precut]
	pT_bins = np.quantile(pT, np.linspace(0, 1, bins))
	eta_bins = np.quantile(np.abs(eta), np.linspace(0, 1, bins))
	return pT_bins, eta_bins, (bins - 1) ** 2

def to_detector(y, never_mind=False):
	if never_mind:
		return np.copy(y)
	px, py, pz, E = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
	# if (E < 0).sum():
	#   print(f'Found {(E < 0).sum()} negative energy events')
	E = np.maximum(E, 0)
	# if (np.abs(pz) > E).sum():
	#   print(f'Found {(np.abs(pz) > E).sum()} |pz| > max(E, 0) events')
	pz = np.clip(pz, -E, E)
	pT2 = px**2+py**2
	p2 = pT2 + pz ** 2
	pT = np.sqrt(pT2)
	p = np.sqrt(p2)
	# eta = np.arctanh(pz/p)
	r = (E+pz)/(np.clip(E-pz, a_min=1e-5, a_max=None)) 
	eta = 1/2 * np.log(np.clip(r, a_min=1e-5, a_max=None)) # actually it's rapidity(y) not eta
	isNan = np.isnan(eta)
	if isNan.sum():
		print(f'Found {isNan.sum()} nans in eta')
	# plt.hist(pT, bins=np.linspace(0, 600, 40))
	phi = np.arctan2(py,px)
	m2 = E**2-p2
	valid_m2 = m2 >= 0
	m = np.sqrt(m2 * valid_m2)
	return np.concatenate([pT.reshape(-1 , 1), eta.reshape(-1 , 1), phi.reshape(-1 , 1), m.reshape(-1 , 1)], axis=-1)

def compute_binned_stats(y_target, y_pred, y_pred_alt, num_pred, num_target, truth_matched, attention_matched, precut, x, y, dx, dy, max_num_output, pull=False):
	eta_target = y_target[:, 1]
	p_target = y_target[:, 0]

	stats_array = []
	for eta, deta in tqdm(zip(y[1:], dy), total=dy.shape[0]):
		for p, dp in zip(x[:-1], dx):
			cut = precut * (p <= p_target) * (p_target < p + dp)
			cut = cut * (eta <= np.abs(eta_target)) * (np.abs(eta_target) < eta + deta) 
			# if cut.sum() == 0:
			#     print(f'No event in p={p}, y={eta} bin')
			if pull:
				dy_pred_alt = (np.abs(y_pred_alt - y_target) / (np.abs(y_target) + 1e-3))[cut == 1]
				dy_pred = (np.abs(y_pred - y_target) / (np.abs(y_target) + 1e-3))[cut == 1]
			else:
				dy_pred_alt = (y_pred_alt - y_target)[cut == 1]
				dy_pred = (y_pred - y_target)[cut == 1]
			is_attention_matched = attention_matched[cut == 1]
			is_truth_matched = truth_matched[cut == 1]

			dy_pred_alt[:, 2] = np.arccos(np.cos(dy_pred_alt[:, 2]))
			dy_pred[:, 2] = np.arccos(np.cos(dy_pred[:, 2]))

			dy_pred_alt, frac_rejected_pred_alt = reject_outliers(np.abs(dy_pred_alt), max_quantile=0.975)
			dy_pred, frac_rejected_pred  = reject_outliers(np.abs(dy_pred), max_quantile=0.975) if dy_pred.shape[0] > 0 else (np.zeros([0, 4]), np.nan)

			stats_array.append({
				'num_tops': cut.sum(),
				'truth_matched_frac': is_truth_matched.mean(),
				'frac_attention_matched': (is_attention_matched.sum() / is_truth_matched.sum()) if is_truth_matched.sum() > 0 else 0,
				'attention_matched': is_attention_matched.sum(),
				'truth_matched': is_truth_matched.sum(),
				'dy_pred': dy_pred,
				'dy_pred_alt': dy_pred_alt,
				'frac_rejected_pred_alt': frac_rejected_pred_alt,
				'frac_rejected_pred': frac_rejected_pred,
				'rmse_pred_alt': np.mean(dy_pred_alt ** 2, axis=0) ** 0.5, 
				'rmse_pred': np.mean(dy_pred ** 2, axis=0) ** 0.5, 
			})
	return stats_array

def get_global_metric(y_target, y_pred, num_target, num_pred, probs, precut, output_dir, max_num_output, name, detector=False, valid_y=1, y_pred_alt=None):
	sns.set(font_scale=1)
	if not detector:
		hist_config = {
				"alpha": 0.8,
				"lw": 2,
				'histtype': 'step',
		}
		range_config = [
			dict([("bins",40), ("range",(-800, 800))]),
			dict([("bins",40), ("range",(-800, 800))]),
			dict([("bins",40), ("range",(-2000, 2000))]),
			dict([("bins",40), ("range",(0, 2200))]),
		]
		dim_labels = ['$p_x$', '$p_y$', '$p_z$', '$E$']
		y_dim = y_target.shape[-1]
		num_total_y = y_target.shape[0]
	else:
		hist_config = {
				"alpha": 0.8,
				"lw": 2,
				'histtype': 'step',
		}
		range_config = [
			dict([("bins",40), ("range",(0, 800))]),
			dict([("bins",40), ("range",(-3, 3))]),
			dict([("bins",40), ("range",(-3.15, 3.15))]),
			dict([("bins",40), ("range",(163, 183))]),
		]
		dim_labels=['$p_T$', '$y$', '$\phi$', '$m$']
		dim_labels_units = ['$p_T/\mathrm{GeV}$', '$y$', '$\phi$', '$m/\mathrm{GeV}$']
		y_dim = y_target.shape[-1]
		num_total_y = y_target.shape[0]

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	
	cut = precut
	num_pred_per_top = np.repeat(num_pred, max_num_output, axis=0)[cut]
	num_target_per_top = np.repeat(num_target, max_num_output, axis=0)[cut]
	probs_per_top = np.repeat(probs, max_num_output, axis=0)[cut]
	plt.figure(dpi=200)
	for n in range(1, max_num_output + 1):
		plt.hist(probs_per_top[:, n - 1], label=f'$n={n}$', density=True, bins=np.linspace(0, 1, 41), lw=2, histtype='stepfilled', alpha=0.5)
	plt.xlabel(r'$P_\mathcal{M}(N_{top} = n)$')
	plt.ylabel('Density')
	plt.legend()
	plt.savefig(f'{output_dir}/N_top_prob.png')
	plt.close()
	plt.figure()
	acc = (num_pred_per_top == num_target_per_top).sum() / cut.sum()
	plt.hist(num_pred_per_top, label=f'acc = {acc:.2g}')
	plt.legend()
	plt.savefig(f'{output_dir}/number.png')
	plt.close()
	# ghost_value = utils.get_ghost_value() * np.ones(4)
	# valid_Y = np.abs(y_pred - ghost_value).sum(-1) > 0
	plt.figure()
	sns.set_style("white")
	sns.axes_style("white")
	_, axs = plt.subplots(2, y_dim, figsize=(4*y_dim, 4*2), constrained_layout=True)
	plot_comp(axs[0], y_pred[cut], y_pred_alt[cut] if y_pred_alt else None, y_target[cut], dim_labels_units, hist_config, range_config)
	pulls_reg, med_pull, resolution = plot_error(axs[1], y_pred[cut], y_target[cut], dim_labels, hist_config, range_config, label=r'Prediction', detector=detector)
	if y_pred_alt != None:
		pulls_reg, med_pull, resolution = plot_error(axs[1], y_pred_alt[cut], y_target[cut], dim_labels, hist_config, range_config, label=r"$\mathcal{M}'$", detector=detector)
	# diffs_reg = plot_error(axs[2], y_pred[cut], y_target[cut], num_total_y, dim_labels, hist_config, range_config, title=f'{N_top} top', label=f'{N_top} top', normalize_by_truth=False)
	plt.savefig(f'{output_dir}/{name}.png')
	plt.close()
	return med_pull, resolution, acc

def plot_error(axs, y_pred, y_truth, dim_labels, hist_config, range_config, label=None, cut_off=5, normalize_by_truth=True, detector=False):
	y_dim = y_truth.shape[-1]
	diffs = (y_pred - y_truth)
	if detector:
		diffs[:, 2] = np.arctan2(np.sin(diffs[:, 2]), np.cos(diffs[:, 2])) # signed delta phi
	if normalize_by_truth:
		pulls = diffs / y_truth
	else:
		pulls = diffs
	# pulls = pulls[np.sum(np.abs(y_pred), axis=1) != 0]
	pT_ratio = y_pred[:, 0] / y_truth[:, 0]
	pT_resolution = iqr(pT_ratio) / np.median(pT_ratio)
	iqr_diff = iqr(diffs)
	resolution = np.array([pT_resolution] + iqr_diff[1:].tolist())
	# med_abs = np.median(np.abs(pulls), axis=0)
	med = np.median(pulls, axis=0)
	for i in range(y_dim):
		pull = pulls[:, i]
		# med = np.median(pull)
		# sigma = mad(pull)
		if normalize_by_truth:
			axs[i].hist(pull, bins=np.linspace(-2, 2, 40), label=label, density=True)
		else:
			axs[i].hist(pull, **range_config[i], label=label)
		if normalize_by_truth:
			axs[i].set_xlabel('$' + '\Delta ' + dim_labels[i].replace('$', '') + '/' + dim_labels[i].replace('$', '') + '$')
		else:
			axs[i].set_xlabel('$' + '\Delta ' + dim_labels[i].replace('$', '') + '$')
		axs[i].set_ylabel('Density')
		# axs[i].title.set_text(rf'Med = {med[i]:.2g} Med abs = {med_abs[i]:.2g}')
		# axs[i].legend(fontsize='small')
	return pulls, med, resolution

def plot_comp(axs, y_pred, y_pred_alt, y_truth, dim_labels, hist_config, range_config, title=None):
	y_dim = y_truth.shape[-1]
	for i in range(y_dim):
		if not y_pred_alt is None:
			axs[i].hist(y_pred_alt[:, i], label=r"$\mathcal{M}'$", **hist_config, **range_config[i], density=True)
		axs[i].hist(y_pred[:, i], label=r'Prediction', **hist_config, **range_config[i], density=True)
		axs[i].hist(y_truth[:, i], label='Truth', **hist_config, **range_config[i], density=True)
		axs[i].set_xlabel(dim_labels[i])
		axs[i].set_ylabel('Normalized to unity')
		axs[i].legend(fontsize='small')

def iqr(x):
	return (np.quantile(x, 0.84, axis=0) - np.quantile(x, 0.16, axis=0)) / 2

def mad(x):
	return np.median(np.abs(x - np.median(x)))

def reject_outliers(x, min_quantile=None, max_quantile=None):
	_min = np.quantile(x, min_quantile, axis=0) if min_quantile else -np.inf
	_max = np.quantile(x, max_quantile, axis=0) if max_quantile else np.inf
	reject = np.sum((x > _max) + (x < _min), axis=-1)
	keep = reject == 0
	return x[keep], 1 - keep.mean()
	# return x

def get_valid_entry_mask(test_result, max_num_output, keep_incorrect_num_top=False):
	num_pred_per_top = np.repeat(test_result['num_pred'], max_num_output, axis=0)
	num_target_per_top = np.repeat(test_result['num_target'], max_num_output, axis=0)
	ghost_value = utils.get_ghost_value() * np.ones(4)
	valid_entry_mask = (np.abs(test_result['y_pred'] - ghost_value).sum(-1) > 0) * (num_target_per_top == num_pred_per_top)
	if keep_incorrect_num_top:
		valid_entry_mask = (np.abs(test_result['y_pred'] - ghost_value).sum(-1) > 0)
	return valid_entry_mask