#!/usr/bin/env python
#
# Example usage:
#
# NOTE:
# optype=real: Gaussian parametrization uses a non-linear transformation of split times
#   variance should decrease as split_time increases:
#   variance at node j = variance_coef * (sigmoid(sigmoid_coef * t_j) - sigmoid(sigmoid_coef * t_{parent(j)}))
#   non-linear transformation should be a monotonically non-decreasing function
#   sigmoid has a saturation effect: children will be similar to parent as we go down the tree
#   split times t_j scales inversely with the number of dimensions
import pickle as pickle
import pprint as pp
import time
from warnings import warn

import numpy as np

from imondrian.mondrianforest import MondrianForest
from imondrian.mondrianforest import process_command_line
from imondrian.mondrianforest_utils import get_filename_mf
from imondrian.mondrianforest_utils import load_data
from imondrian.mondrianforest_utils import precompute_minimal
from imondrian.mondrianforest_utils import reset_random_seed

try:
    import matplotlib.pyplot as plt
    from matplotlib import rc
    rc('font', **{'family': 'serif'})
    rc('text', usetex=True)
    rc('legend', handlelength=4)
    rc('legend', **{'fontsize': 9})
except Exception:
    warn('matplotlib not loaded: plotting not possible; set draw_mondrian=0')


time_0 = time.clock()
settings = process_command_line()
print()
print(('%' * 120))
print('Beginning mondrianforest.py')
print('Current settings:')
pp.pprint(vars(settings))

# Resetting random seed
reset_random_seed(settings)

# Loading data
print('\nLoading data ...')
data = load_data(settings)
print('Loading data ... completed')
print(('Dataset name = %s' % settings.dataset))
print('Characteristics of the dataset:')
print((
    'n_train = %d, n_test = %d, n_dim = %d' %
    (data['n_train'], data['n_test'], data['n_dim'])
))
if settings.optype == 'class':
    print(('n_class = %d' % (data['n_class'])))

# precomputation
param, cache = precompute_minimal(data, settings)
time_init = time.clock() - time_0

print('\nCreating Mondrian forest')
# online training with minibatches
time_method_sans_init = 0.
time_prediction = 0.
mf = MondrianForest(settings, data)
if settings.store_every:
    log_prob_test_minibatch = -np.inf * np.ones(settings.n_minibatches)
    log_prob_train_minibatch = -np.inf * np.ones(settings.n_minibatches)
    metric_test_minibatch = -np.inf * np.ones(settings.n_minibatches)
    metric_train_minibatch = -np.inf * np.ones(settings.n_minibatches)
    time_method_minibatch = np.inf * np.ones(settings.n_minibatches)
    forest_numleaves_minibatch = np.zeros(settings.n_minibatches)
for idx_minibatch in range(settings.n_minibatches):
    time_method_init = time.clock()
    is_last_minibatch = (idx_minibatch == settings.n_minibatches - 1)
    print_results = is_last_minibatch or (
        settings.verbose >= 2
    ) or settings.debug
    if print_results:
        print(('*' * 120))
        print(('idx_minibatch = %5d' % idx_minibatch))
    train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
    if settings.debug:
        print((
            'bagging = %s, train_ids_current_minibatch = %s' %
            (settings.bagging, train_ids_current_minibatch)
        ))
    if idx_minibatch == 0:
        mf.fit(data, train_ids_current_minibatch, settings, param, cache)
    else:
        mf.partial_fit(
            data, train_ids_current_minibatch,
            settings, param, cache,
        )
    for i_t, tree in enumerate(mf.forest):
        if settings.debug or settings.verbose >= 2:
            print(('-'*100))
            tree.print_tree(settings)
            print(('.'*100))
        if settings.draw_mondrian:
            tree.draw_mondrian(data, settings, idx_minibatch, i_t)
            if settings.save == 1:
                filename_plot = get_filename_mf(settings)
                if settings.store_every:
                    plt.savefig(
                        filename_plot + '-mondrians_minibatch-' +
                        str(idx_minibatch) + '.pdf', format='pdf',
                    )
    time_method_sans_init += time.clock() - time_method_init
    time_method = time_method_sans_init + time_init

    # Evaluate
    if is_last_minibatch or settings.store_every:
        time_predictions_init = time.clock()
        weights_prediction = np.ones(
            settings.n_mondrians,
        ) * 1.0 / settings.n_mondrians
        if False:
            if print_results:
                print('Results on training data (log predictive prob is bogus)')
            train_ids_cumulative = data['train_ids_partition']['cumulative'][idx_minibatch]
            # NOTE: some of these data points are not used for "training" if bagging is used
            pred_forest_train, metrics_train = \
                mf.evaluate_predictions(
                    data, data['x_train'][train_ids_cumulative, :],
                    data['y_train'][train_ids_cumulative],
                    settings, param, weights_prediction, print_results,
                )
        else:
            # not computing metrics on training data
            metrics_train = {'log_prob': -np.inf, 'acc': 0, 'mse': np.inf}
            pred_forest_train = None
        if print_results:
            print('\nResults on test data')
        pred_forest_test, metrics_test = \
            mf.evaluate_predictions(
                data, data['x_test'], data['y_test'],
                settings, param, weights_prediction, print_results,
            )
        name_metric = settings.name_metric     # acc or mse
        log_prob_train = metrics_train['log_prob']
        log_prob_test = metrics_test['log_prob']
        metric_train = metrics_train[name_metric]
        metric_test = metrics_test[name_metric]
        if settings.store_every:
            log_prob_train_minibatch[idx_minibatch] = metrics_train['log_prob']
            log_prob_test_minibatch[idx_minibatch] = metrics_test['log_prob']
            metric_train_minibatch[idx_minibatch] = metrics_train[name_metric]
            metric_test_minibatch[idx_minibatch] = metrics_test[name_metric]
            time_method_minibatch[idx_minibatch] = time_method
            tree_numleaves = np.zeros(settings.n_mondrians)
            for i_t, tree in enumerate(mf.forest):
                tree_numleaves[i_t] = len(tree.leaf_nodes)
            forest_numleaves_minibatch[idx_minibatch] = np.mean(tree_numleaves)
        time_prediction += time.clock() - time_predictions_init

# printing test performance:
if settings.store_every:
    print('printing test performance for every minibatch:')
    print('idx_minibatch\tmetric_test\ttime_method\tnum_leaves')
    for idx_minibatch in range(settings.n_minibatches):
        print((
            '%10d\t%.3f\t\t%.3f\t\t%.1f' %
            (
                idx_minibatch,
                metric_test_minibatch[idx_minibatch],
                time_method_minibatch[idx_minibatch], forest_numleaves_minibatch[idx_minibatch],
            )
        ))
print('\nFinal forest stats:')
tree_stats = np.zeros((settings.n_mondrians, 2))
tree_average_depth = np.zeros(settings.n_mondrians)
for i_t, tree in enumerate(mf.forest):
    tree_stats[
        i_t, -
        2:
    ] = np.array([len(tree.leaf_nodes), len(tree.non_leaf_nodes)])
    tree_average_depth[i_t] = tree.get_average_depth(settings, data)[0]
print((
    'mean(num_leaves) = %.1f, mean(num_non_leaves) = %.1f, mean(tree_average_depth) = %.1f'
    % (np.mean(tree_stats[:, -2]), np.mean(tree_stats[:, -1]), np.mean(tree_average_depth))
))
print((
    'n_train = %d, log_2(n_train) = %.1f, mean(tree_average_depth) = %.1f +- %.1f'
    % (data['n_train'], np.log2(data['n_train']), np.mean(tree_average_depth), np.std(tree_average_depth))
))

if settings.draw_mondrian:
    if settings.save == 1:
        plt.savefig(filename_plot + '-mondrians-final.pdf', format='pdf')
    else:
        plt.show()

# Write results to disk (timing doesn't include saving)
time_total = time.clock() - time_0
# resetting
if settings.save == 1:
    filename = get_filename_mf(settings) + '.p'
    print(('filename = ' + filename))
    results = {
        'log_prob_test': log_prob_test, 'log_prob_train': log_prob_train,
        'metric_test': metric_test, 'metric_train': metric_train,
        'time_total': time_total, 'time_method': time_method,
        'time_init': time_init, 'time_method_sans_init': time_method_sans_init,
        'time_prediction': time_prediction,
    }
    if 'log_prob2' in metrics_test:
        results['log_prob2_test'] = metrics_test['log_prob2']
    store_data = settings.dataset[:3] == 'toy' or settings.dataset == 'sim-reg'
    if store_data:
        results['data'] = data
    if settings.store_every:
        results['log_prob_test_minibatch'] = log_prob_test_minibatch
        results['log_prob_train_minibatch'] = log_prob_train_minibatch
        results['metric_test_minibatch'] = metric_test_minibatch
        results['metric_train_minibatch'] = metric_train_minibatch
        results['time_method_minibatch'] = time_method_minibatch
        results['forest_numleaves_minibatch'] = forest_numleaves_minibatch
    results['settings'] = settings
    results['tree_stats'] = tree_stats
    results['tree_average_depth'] = tree_average_depth
    pickle.dump(
        results, open(filename, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    # storing final predictions as well; recreate new "results" dict
    results = {
        'pred_forest_train': pred_forest_train,
        'pred_forest_test': pred_forest_test,
    }
    filename2 = filename[:-2] + '.tree_predictions.p'
    pickle.dump(
        results, open(filename2, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

time_total = time.clock() - time_0
print()
print(('Time for initializing Mondrian forest (seconds) = %f' % (time_init)))
print((
    'Time for executing mondrianforest.py (seconds) = %f' % (
        time_method_sans_init
    )
))
print((
    'Total time for executing mondrianforest.py, including init (seconds) = %f' % (
        time_method
    )
))
print(('Time for prediction/evaluation (seconds) = %f' % (time_prediction)))
print(('Total time (Loading data/ initializing / running / predictions / saving) (seconds) = %f\n' % (time_total)))
