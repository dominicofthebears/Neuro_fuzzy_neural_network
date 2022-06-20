import os.path
import random
import sys
from sys import argv

import pandas as pd
import torch

from matplotlib import pyplot as plt

from anfis.load_model import metrics
from features_preprocessing import Feature_ranking
from features_preprocessing.Feature_ranking import mutual_info
from features_preprocessing.Preprocessing import chimerge, Discretization
from train_anfis import opt, train
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from time import perf_counter
import numpy as np
import collections

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)


# librerie da installare
# pip install torch
# pip install ipywidgets
# pip install hyperopt

def equal_frequency_binning(data, bins, column):
    col = dataframe[column].astype(float)
    split = np.array_split(np.sort(data), bins)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    val = cutoffs[0] / 2
    col = col.apply(lambda x: val if x <= cutoffs[0] else x)
    for i in range(0, bins - 1):
        if i == bins - 2:
            val = (cutoffs[i] + 1) / 2
            col = col.apply(lambda x: val if cutoffs[i] <= x else x)
        else:
            val = (cutoffs[i] + cutoffs[i + 1]) / 2
            col = col.apply(lambda x: val if cutoffs[i] <= x < cutoffs[i + 1] else x)
    return col


def equal_width_binning(column, m):
    w = (max(column) - min(column)) / m
    left = 0
    right = w
    for i in range(0, m):
        val = (left + right) / 2
        if right == 1:
            column = column.apply(lambda x: val if left <= x <= right else x)
        else:
            column = column.apply(lambda x: val if left <= x < right else x)
        left += w
        right += w
    return column


def discretize_feature(interval, column):
    col = dataframe[column].astype(float)
    for interval in interval:
        val = (interval[0] + interval[1]) / 2
        col = col.apply(lambda x: val if interval[0] <= x <= interval[1] else x)
    return col


def fit_and_score(params):
    print(params)
    start_time = perf_counter()

    k = 0
    cols = dataframe[columns_sel]

    if approach == 'triangular':
        for feature in columns_sel:
            print("Discretizing column: " + feature)

            if discretization == 'supervised':
                intervals = chimerge(dataframe, feature, dataframe.columns[-1], params['num_fuzzy_terms'][k])
                dataframe[feature] = discretize_feature(intervals, feature)
            if discretization == 'frequency': dataframe[feature] = equal_frequency_binning(dataframe[feature],
                                                                                           params['num_fuzzy_terms'][k],
                                                                                           feature)
            if discretization == 'width': dataframe[feature] = equal_width_binning(dataframe[feature],
                                                                                   params['num_fuzzy_terms'][k])
            k += 1

    model, scores, best_epoch = opt(dataframe, n_features, params['learning_rate'], params['batch_size'],
                                    params['num_fuzzy_terms'], columns_sel[:n_features], approach)

    score = min(scores)
    print("score->", score)

    end_time = perf_counter()

    global best_score, best_model, best_time, results_dict, best_terms
    print("best_score->", best_score)
    if best_score > score:
        best_score = score
        best_model = model
        best_time = end_time - start_time
        best_terms = params['num_fuzzy_terms']
        print("NEW BEST SCORE", best_score)

    results, conf_matrix = metrics(dataset, columns_sel, n_features, model, target)
    metrics_results = ""
    for key, value in results.items():
        metrics_results += (str(key) + ' : ' + str(value) + '\n')
    key = score
    val = "Learning rate: " + str(params['learning_rate']) + "  Batch size: " + str(params['batch_size']) + \
          "  Num fuzzy terms: " + str(params['num_fuzzy_terms']) + "  Val loss: " + str(score) + '\n' + \
          "--------------------------------;" + '\n' + metrics_results + "--------------------------------;" + '\n' + str(
        conf_matrix) + '\n\n\n'
    results_dict[key] = val

    keys = sorted(list(results_dict.keys()))
    f = open("../results/test/" + dataset + "_" + str(n_features) + "_" + approach + "_test_results.csv", "w")
    f.write("SELECTED COLUMNS;" + "\n")
    for element in columns_sel:
        f.write(element + "\n")
    f.write("--------------------------------;" + '\n')
    for item in keys:
        f.write(str(results_dict[item]))
    f.close()

    path = "../plots/" + dataset + "/" + str(n_features) + "/"
    epochs = [i for i in range(0, len(list(scores)))]
    plt.scatter(epochs, list(scores))
    if not (os.path.exists(path)):
        os.makedirs(path)
    plt.savefig(
        path + dataset + '_' + str(n_features) + '_loss_plot_' + str(params['learning_rate']) + '_' + str(
            params['batch_size']) + '.png')
    plt.close()
    dataframe[columns_sel] = cols
    return {'loss': score, 'status': STATUS_OK,
            'time': end_time - start_time,
            'best_epoch': best_epoch}


if len(argv) == 1:
    dataset = 'maldroid'
    target = 'binary'
    approach = 'gaussian'
    n_features = 2
    discretization = 'none'

else:
    try:
        filename, dataset, target, approach, n_features, discretization = sys.argv
        n_features = int(n_features)
        if dataset != 'maldroid' and dataset != 'CICIDS': raise IOError
        if target != 'binary' and target != 'multiclass': raise IOError
        if approach != 'gaussian' and approach != 'triangular': raise IOError
        if n_features <= 0: raise IOError
        if approach == 'gaussian' and discretization != 'none': raise IOError
        if approach == 'triangular' and discretization != 'width' and discretization != 'frequency' and discretization != 'supervised': raise IOError

    except IOError:
        print("Parametri errati, verificare la guida e riprovare")
        sys.exit(1)
    except (ValueError, TypeError):
        print("Il numero di features deve essere un valore intero positivo")
        sys.exit(1)

# model selection
print('Starting model selection...\n')
print(
    "Dataset: " + str(dataset) + "\nTarget: " + str(target) + "\nApproach: " + str(approach) + "\nNumber of features: "
    + str(n_features) + "\nDiscretization: " + str(discretization) + '\n')

best_score = np.inf
best_model = None
results_dict = collections.OrderedDict()
best_terms = []
values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

outfile = open('../log_files/' + dataset + '_' + str(n_features) + '_opt.log', 'w')
dataframe = pd.read_csv("../datasets/" + dataset + '/' + dataset + "_train_" + target + ".csv", header=0, sep=',')
num_classes = len(dataframe.iloc[:, -1].value_counts())

ordered_dict = mutual_info(dataframe)
columns_sel = []
columns_list = list(ordered_dict.keys())

idx = 0
l = 0
while l < n_features:
    col_name = columns_list[idx]
    var = list(dataframe[col_name].value_counts().items())[0]
    if (str(var[0]) != '0.0') or (str(var[0]) == '0.0' and var[1] < len(dataframe[col_name]) / 2):
        columns_sel.append(columns_list[idx])
        l += 1
    idx += 1

space = {
    'batch_size': hp.choice('batch_size', [64, 128, 256, 512]),
    'learning_rate': hp.uniform("learning_rate", (0.00001), (0.0001)),
    'num_fuzzy_terms': [
        hp.choice("num_fuzzy_terms_" + str(random.random()) + str(columns_sel[i] for i in range(0, n_features)), values)
        for x in range(0, n_features)],
    'n_classes': num_classes}

best_score = np.inf
best_model = None

best_time = 0
best_numparameters = 0
trials = Trials()
max_evals = 20
best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=max_evals, trials=trials,
            rstate=np.random.default_rng(seed))
best_params = hyperopt.space_eval(space, best)

outfile.write("\nHyperopt trials")
outfile.write("\ntid,loss,learning_rate,batch_size,time, num_epochs")
for trial in trials.trials:
    outfile.write("\n%d,%f,%8f,%d,%f, %d" % (trial['tid'],
                                             trial['result']['loss'],
                                             trial['misc']['vals']['learning_rate'][0],
                                             trial['misc']['vals']['batch_size'][0],
                                             trial['result']['time'],
                                             trial['result']['best_epoch']
                                             ))

outfile.write("\n\nBest parameters:")
print(best_params, file=outfile)
outfile.write('\nBest Time taken: %f' % best_time)

print(best_model)

if (approach == 'triangular'):
    l = 0
    for feature in columns_sel:
        if (discretization == 'supervised'):
            intervals = chimerge(dataframe, feature, dataframe.columns[-1], best_terms[l])
            dataframe[feature] = discretize_feature(intervals, feature)
        if (discretization == 'frequency'): dataframe[feature] = equal_frequency_binning(dataframe[feature],
                                                                                         best_terms[l], feature)
        if (discretization == 'width'): dataframe[feature] = equal_width_binning(dataframe[feature], best_terms[l])
        l += 1

train(dataframe, n_features, best_params['learning_rate'], best_params['batch_size'],
      best_params['num_fuzzy_terms'],
      columns_sel, approach, dataset)
