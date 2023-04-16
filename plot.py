import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (8, 6),
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large',
#           'figure.dpi': 100}
# pylab.rcParams.update(params)

def plot_reduced_pull(y_pred, y_truth, num_total_y, dim_labels, hist_config, range_config, title=None, label=None, cut_off=5, normalize_by_truth=True, standardize=False):
    plt.figure()
    y_dim = y_truth.shape[-1]
    pulls = (y_pred - y_truth)
    if normalize_by_truth:
        pulls = pulls / y_truth
        bins=np.linspace(0, 3, 80)
    elif not standardize:
        bins=np.linspace(0, 800, 80)
    else:
        std = np.std(y_truth, axis=0)
        pulls /= std
        bins=np.linspace(0, 3, 80)
    # pulls = pulls[np.sum(np.abs(y_pred), axis=1) != 0]    
    pull = np.mean(pulls ** 2, axis=1) ** 0.5
    med = np.median(pull)
    mean = np.mean(pull)
    std = np.std(pull)
    mad_ = mad(pull)
    plt.hist(pull, bins=bins, label=label, density=True)
    plt.xlabel('reduced pull')
    plt.title(rf'Med = {med:.2g} $\quad$ Mad = {mad_:.2g} $\quad$ Mean = {mean:.2g} $\quad$ Std = {std:.2g}')
    plt.legend(fontsize='small')


def plot_pull(axs, y_pred, y_truth, num_total_y, dim_labels, hist_config, range_config, title=None, label=None, cut_off=5, normalize_by_truth=True):
    print(f'{title}: {y_pred.shape[0]/num_total_y * 100:.2g}%', f'{y_pred.shape[0]} / {num_total_y}')
    y_dim = y_truth.shape[-1]
    pulls = (y_pred - y_truth)
    if normalize_by_truth:
    	pulls = pulls / y_truth
    # pulls = pulls[np.sum(np.abs(y_pred), axis=1) != 0]
    for i in range(y_dim):
        pull = pulls[:, i]
        med = np.median(pull)
        sigma = mad(pull)
        if normalize_by_truth:
            axs[i].hist(pull, bins=np.linspace(-2, 2, 80), label=label)
        else:
            axs[i].hist(pull, **range_config[i], label=label)
        axs[i].set_xlabel('pull ' + dim_labels[i])
        axs[i].title.set_text(rf'Med = {med:.2g} $\quad$ Mad = {sigma:.2g}')
        axs[i].legend(fontsize='small')
    return pulls

def plot_comp(axs, y_clf, y_reg, y_truth, dim_labels, hist_config, range_config, title=None):
    y_dim = y_truth.shape[-1]
    for i in range(y_dim):
        axs[i].hist(y_truth[:, i], label='truth', **hist_config, **range_config[i], density=True)
        if not y_clf is None:
            axs[i].hist(y_clf[:, i], label='top reco', **hist_config, **range_config[i], density=True)
        axs[i].hist(y_reg[:, i], label='pred', **hist_config, **range_config[i], density=True)
        axs[i].set_xlabel(dim_labels[i])
        axs[i].legend(fontsize='small')


def evaluate_performance(output_dir, y, regression, truth_matched, identified, reco_top, min_dR_candidate_top, is_valid_candidate, coordinates):
    hist_config = {
            "alpha": 0.8,
            "lw": 2,
            'histtype': 'step',
        }
    if coordinates == 'spherical':
        range_config = [
            dict([("bins",40), ("range",(0, 800))]),
            dict([("bins",40), ("range",(-5, 5))]),
            dict([("bins",40), ("range",(-3.15, 3.15))]),
            dict([("bins",80), ("range",(0, 400))]),
        ]
        dim_labels=['$p_T$', '$\eta$', '$\phi$', '$m$']
    elif coordinates == 'cartesian':
        range_config = [
        dict([("bins",40), ("range",(-800, 800))]),
        dict([("bins",40), ("range",(-800, 800))]),
        dict([("bins",40), ("range",(-2000, 2000))]),
        dict([("bins",40), ("range",(150, 200))]),
        ]
        dim_labels=['$p_x$', '$p_y$', '$p_z$', '$m$']
    elif coordinates == 'cartesian_E':
        range_config = [
        dict([("bins",40), ("range",(-800, 800))]),
        dict([("bins",40), ("range",(-800, 800))]),
        dict([("bins",40), ("range",(-2000, 2000))]),
        dict([("bins",40), ("range",(0, 2200))]),
        ]
        dim_labels=['$p_x$', '$p_y$', '$p_z$', '$E$']

    num_total_y = y.shape[0]
    y_dim = y.shape[-1]
    eval_output_dir = f'{output_dir}/{coordinates}/'
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
        # # Case 1 Truth top is truth matched to a reconstructed triplet, which is correctly identified by the GNN
        # # Reco_triplet has the indices of jets that are matched to a top quark
        # # Pull plots for reg. vs truth, truth matched vs truth
        # case = 1
        # cut = identified.astype(bool)
        # y_clf = reco_top[cut]
        # y_reg = regression[cut]
        # plt.figure()
        # _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
        # pulls_clf = plot_pull(axs[0], y_clf, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Identifiable and Correctly Identified', label='top reco')
        # pulls_reg = plot_pull(axs[1], y_reg, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Regression', label='reg')
        # write_statistics(pulls_clf, dim_labels, case, num_total_y, f'{eval_output_dir}/clf.csv')
        # write_statistics(pulls_reg, dim_labels, case, num_total_y, f'{eval_output_dir}/reg.csv')
        # plot_comp(axs[2], y_clf, y_reg, y[cut], dim_labels, hist_config, range_config)
        # plt.savefig(f'{eval_output_dir}/{case}.png')
        # plt.show()
        
        # # Case 2,3 Truth top has a truth matched reco_triplet, but not identified:
        #     # 2. Use the reconstructed triplet that has the minimum dR with the truth top.    
        #     # 3. Use the truth matched reco_triplet(Perfect classifier)
        # case = 2
        # cut = (truth_matched * (1 - identified)).astype(bool)
        # y_clf = reco_top[cut]
        # y_reg = regression[cut]
        # plt.figure()
        # _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
        # pulls_clf = plot_pull(axs[0], y_clf, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Identifiable and Not Identified (Oracle)', label='top reco')
        # pulls_reg = plot_pull(axs[1], y_reg, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Regression', label='reg')
        # write_statistics(pulls_clf, dim_labels, case, num_total_y, f'{eval_output_dir}/clf.csv')
        # write_statistics(pulls_reg, dim_labels, case, num_total_y, f'{eval_output_dir}/reg.csv')
        # plot_comp(axs[2], y_clf, y_reg, y[cut], dim_labels, hist_config, range_config)
        # plt.savefig(f'{eval_output_dir}/{case}.png')
        # plt.show()

        # case = 3
        # cut = (cut * is_valid_candidate).astype(bool)
        # y_reg = regression[cut]
        # y_clf = min_dR_candidate_top[cut]
        # plt.figure()
        # _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
        # pulls_clf = plot_pull(axs[0], y_clf, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Identifiable and Not Identified (Min-dR truth-macthed)', label='top reco')
        # pulls_reg = plot_pull(axs[1], y_reg, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Regression', label='reg')
        # write_statistics(pulls_clf, dim_labels, case, num_total_y, f'{eval_output_dir}/clf.csv')
        # write_statistics(pulls_reg, dim_labels, case, num_total_y, f'{eval_output_dir}/reg.csv')
        # plot_comp(axs[2], y_clf, y_reg, y[cut], dim_labels, hist_config, range_config)
        # plt.savefig(f'{eval_output_dir}/{case}.png')
        # plt.show()


        # # Case 4 Truth top does not have a truth matched reco_triplet:
        # #        Use the reconstructed triplet that has the minimum dR with the truth top. 
        # case = 4
        # cut = (1 - truth_matched).astype(bool)
        # cut = (cut * is_valid_candidate).astype(bool)
        # y_clf = min_dR_candidate_top[cut]
        # y_reg = regression[cut]
        # plt.figure()
        # _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
        # pulls_clf = plot_pull(axs[0], y_clf, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Not Identifiable and Not Identified (Min-dR truth-macthed)', label='top reco')
        # pulls_reg = plot_pull(axs[1], y_reg, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Regression', label='reg')
        # write_statistics(pulls_clf, dim_labels, case, num_total_y, f'{eval_output_dir}/clf.csv')
        # write_statistics(pulls_reg, dim_labels, case, num_total_y, f'{eval_output_dir}/reg.csv')
        # plot_comp(axs[2], y_clf, y_reg, y[cut], dim_labels, hist_config, range_config)
        # plt.savefig(f'{eval_output_dir}/{case}.png')
        # plt.show()

        # # Case 5 Truth-matched or min-dR candidate
        # case = 5
        # cut = (truth_matched + is_valid_candidate) > 0
        # y_clf = truth_matched.reshape(-1, 1) * reco_top + (1 - truth_matched).reshape(-1, 1) * min_dR_candidate_top
        # y_clf = y_clf[cut]
        # y_reg = regression[cut]
        
        
        # plt.figure()
        # _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
        # pulls_clf = plot_pull(axs[0], y_clf, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Truth-matched or Min-dR candidate', label='top reco')
        # pulls_reg = plot_pull(axs[1], y_reg, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Regression', label='reg')
        # write_statistics(pulls_clf, dim_labels, case, num_total_y, f'{eval_output_dir}/clf.csv')
        # write_statistics(pulls_reg, dim_labels, case, num_total_y, f'{eval_output_dir}/reg.csv')
        # plot_comp(axs[2], y_clf, y_reg, y[cut], dim_labels, hist_config, range_config)
        # plt.savefig(f'{eval_output_dir}/{case}.png')
        # plt.show()

        # # Case 6 Identified or min-dR candidate
        # case = 6
        # cut = (identified + is_valid_candidate) > 0
        # y_clf = identified.reshape(-1, 1) * reco_top + (1 - identified).reshape(-1, 1) * min_dR_candidate_top
        # y_clf = y_clf[cut]
        # y_reg = regression[cut]
        
        
        # plt.figure()
        # _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
        # pulls_clf = plot_pull(axs[0], y_clf, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Top Reconstruction Identified or Min-dR candidate', label='top reco')
        # pulls_reg = plot_pull(axs[1], y_reg, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Regression', label='reg')
        # write_statistics(pulls_clf, dim_labels, case, num_total_y, f'{eval_output_dir}/clf.csv')
        # write_statistics(pulls_reg, dim_labels, case, num_total_y, f'{eval_output_dir}/reg.csv')
        # plot_comp(axs[2], y_clf, y_reg, y[cut], dim_labels, hist_config, range_config)
        # plt.savefig(f'{eval_output_dir}/{case}.png')
        # plt.show()

    # Case 7 Truth-matched(Reg v.s Reco)
    case = 7
    cut = truth_matched.astype(bool)
    y_clf = reco_top[cut]
    y_reg = regression[cut]
    
    plt.figure()
    _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
    pulls_clf = plot_pull(axs[0], y_clf, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Truth-matched triplet', label='top reco')
    pulls_reg = plot_pull(axs[1], y_reg, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Regression', label='reg')
    write_statistics(pulls_clf, dim_labels, case, num_total_y, f'{eval_output_dir}/clf.csv')
    write_statistics(pulls_reg, dim_labels, case, num_total_y, f'{eval_output_dir}/reg.csv')
    plot_comp(axs[2], y_clf, y_reg, y[cut], dim_labels, hist_config, range_config)
    plt.savefig(f'{eval_output_dir}/{case}.png')
    plt.show()

    # Case 8 Not truth-matched(not reconstructable)
    case = 8
    cut = (1 - truth_matched).astype(bool)
    y_clf = reco_top[cut]
    y_reg = regression[cut]
    
    plt.figure()
    _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
    pulls_reg = plot_pull(axs[1], y_reg, y[cut], num_total_y, dim_labels, hist_config, range_config, title='Regression Not Truth-matched', label='reg')
    write_statistics(pulls_clf, dim_labels, case, num_total_y, f'{eval_output_dir}/clf.csv')
    write_statistics(pulls_reg, dim_labels, case, num_total_y, f'{eval_output_dir}/reg.csv')
    plot_comp(axs[2], None, y_reg, y[cut], dim_labels, hist_config, range_config)
    plt.savefig(f'{eval_output_dir}/{case}.png')
    plt.show()


def eval_performance(y_target, y_pred, num_target, num_pred, probs, pre_cut, output_dir, max_num_output):
    hist_config = {
            "alpha": 0.8,
            "lw": 2,
            'histtype': 'step',
    }
    range_config = [
        dict([("bins",80), ("range",(-800, 800))]),
        dict([("bins",80), ("range",(-800, 800))]),
        dict([("bins",80), ("range",(-2000, 2000))]),
        dict([("bins",80), ("range",(0, 2200))]),
    ]
    dim_labels=['$p_x$', '$p_y$', '$p_z$', '$E$']
    y_dim = y_target.shape[-1]
    num_total_y = y_target.shape[0]
    eval_output_dir = f'{output_dir}'
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    num_pred_per_top = np.repeat(num_pred, max_num_output, axis=0)
    num_target_per_top = np.repeat(num_target, max_num_output, axis=0)
    
    for N_top in range(1, max_num_output + 1):
        is_N_top_event = num_target == N_top
        if (is_N_top_event).sum() > 0:
            output_subdir = f'{eval_output_dir}/{N_top}_top'
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            plt.figure()
            for n in range(1, max_num_output + 1):
                plt.hist(probs[is_N_top_event, n - 1], label=f'N={n}', density=False, bins=40, alpha=0.5)
            plt.xlabel('P(#top = N)')
            plt.ylabel('density')
            plt.legend()
            plt.savefig(f'{output_subdir}/N_top_prob.png')
            plt.figure()
            plt.hist(num_pred[is_N_top_event], label=f'acc = {(num_pred[is_N_top_event] == N_top).sum() / (is_N_top_event).sum():.2g}')
            plt.legend()
            plt.savefig(f'{eval_output_dir}/number.png')
            ghost_value = -1e6 * np.ones(4)
            valid_Y = np.abs(y_pred - ghost_value).sum(-1) > 0
            cut = (num_target_per_top == N_top) * valid_Y * (num_pred_per_top == N_top) * (pre_cut == 1)
            plt.figure()
            _, axs = plt.subplots(3, y_dim, figsize=(4*y_dim, 4*3), constrained_layout=True)
            plot_comp(axs[0], None, y_pred[cut], y_target[cut], dim_labels, hist_config, range_config)
            pulls_reg = plot_pull(axs[1], y_pred[cut], y_target[cut], num_total_y, dim_labels, hist_config, range_config, title=f'{N_top} top', label=f'{N_top} top')
            diffs_reg = plot_pull(axs[2], y_pred[cut], y_target[cut], num_total_y, dim_labels, hist_config, range_config, title=f'{N_top} top', label=f'{N_top} top', normalize_by_truth=False)
            plt.savefig(f'{output_subdir}/pull.png')
            plot_reduced_pull(y_pred[cut], y_target[cut], num_total_y, dim_labels, hist_config, range_config, title=f'{N_top} top', label=f'{N_top} top')
            plt.savefig(f'{output_subdir}/reduced_pull.png')
            plot_reduced_pull(y_pred[cut], y_target[cut], num_total_y, dim_labels, hist_config, range_config, title=f'{N_top} top', label=f'{N_top} top', normalize_by_truth=False)
            plt.savefig(f'{output_subdir}/reduced_diff.png')
            plot_reduced_pull(y_pred[cut], y_target[cut], num_total_y, dim_labels, hist_config, range_config, title=f'{N_top} top', label=f'{N_top} top', normalize_by_truth=False, standardize=True)
            plt.savefig(f'{output_subdir}/reduced_diff_standardized.png')
            write_statistics(pulls_reg, dim_labels, num_total_y, f'{output_subdir}/stats.csv')

    




def write_statistics(pulls, dim_labels, num_total_y, csv_file, cut_off=5):
    dim_labels = [l.replace('$', '') for l in dim_labels]
    mode = 'w'
    with open(csv_file, mode) as csvfile:
        csvwriter = csv.writer(csvfile)    
        fields = ['Fraction of truth tops in this case'] + ['Med[Sum(|pull|)]'] + ['Med[QuadSum(|pull|)]'] + ['P(QuadSum(|pull|)] < 0.2)'] + ['P(QuadSum(|pull|)] < 0.4)'] + ['P(QuadSum(|pull|)] < 0.8)'] + ['P(QuadSum(|pull|)] < 1.6)'] + ['P(QuadSum(|pull|)] < 1.0)'] + [f'Med[pull {x}]' for x in dim_labels] + [f'Mad[pull {x}]' for x in dim_labels]
          # writing the fields  
        csvwriter.writerow(fields)
        quad_sum_pull = np.sqrt(np.mean(pulls**2, axis=1))
        row = [pulls.shape[0]/num_total_y] + [np.median(np.sum(np.abs(pulls), axis=1))] + [np.median(quad_sum_pull)] + [np.sum(quad_sum_pull < 0.2) / pulls.shape[0]] + [np.sum(quad_sum_pull < 0.4) / pulls.shape[0]] + [np.sum(quad_sum_pull < 0.8) / pulls.shape[0]] + [np.sum(quad_sum_pull < 1.6) / pulls.shape[0]] + [np.median(pulls[:, i]) for i in range(len(dim_labels))] + [mad(pulls[:, i]) for i in range(len(dim_labels))]
        csvwriter.writerow(row)

def mad(x):
    return np.median(np.abs(x - np.median(x)))