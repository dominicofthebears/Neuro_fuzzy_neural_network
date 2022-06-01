import re
import warnings
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from features_preprocessing.Preprocessing import values_dictionary, frequency_encoder


def boolean_vector_builder(dataframe):
    boolean_vector = np.zeros(len(dataframe.columns))
    for i in range(0, len(dataframe.columns)):
        if re.search("(discrete|boolean)", str(dataframe[dataframe.columns[i]])):
            boolean_vector[i] = True
        else:
            boolean_vector[i] = False
    return boolean_vector.astype(bool)


def mutual_info(dataframe):
    dicts_list = values_dictionary(dataframe)
    X = dataframe.iloc[:, :-1]
    Y = dataframe.iloc[:, -1]
    # X = frequency_encoder(X, dicts_list)
    # X.drop(-1, axis=1, inplace=True)
    discrete = boolean_vector_builder(X)
    warnings.filterwarnings("ignore") #inserito solo per ignorare un warning sollevato nella mutual info di CICIDS a causa del tipo (discreto/continuo) delle features
    fs = mutual_info_classif(X, Y, discrete_features=discrete)
    results = {}
    for i in range(0, len(fs)):
        results[X.columns[i]] = fs[i]
    results = (sorted(results.items(), key=lambda x: x[1], reverse=True))
    # print(*results, sep="\n")
    return dict(results)


def features_histogram(dataframe, dataset):
    n_features = 2
    ordered_dict = mutual_info(dataframe)
    columns_sel = []
    columns_list = list(ordered_dict.keys())

    idx = 0
    l = 0
    while l < n_features:
        col_name = columns_list[idx]
        var = list(dataframe[col_name].value_counts().items())[0]
        if (str(var[0]) != '0.0') or (str(var[0]) == '0.0' and var[1] < len(dataframe[col_name])/2):
            columns_sel.append(columns_list[idx])
            l += 1
        idx += 1
    """
    for column in dataframe.columns:
        
        plt.title(column)
        plt.xlim(min, max)
        plt.ylim(0, len(dataframe[column]))
        plt.hist(dataframe[column].value_counts().keys(), bins=4, histtype="bar")
        plt.show()
        plt.savefig("../histograms/" + dataset + '/' + column, format="png")
        
        vect = dataframe[dataframe[column] < 0.06]
        vect = vect[column]
        
                plt.bar(list(dataframe[column].value_counts().keys()), list(dataframe[column].value_counts().values),
                width=0.01, align='edge', edgecolor='black')
        plt.axis([0, 0.06, 0, len(dataframe[column])])
        plt.show()
        
        print(vect.value_counts())
    """



def main():
    dataset = "maldroid"
    dataframe = pd.read_csv("../datasets/" + dataset + '/' + dataset + "_train.csv")
    features_histogram(dataframe, dataset)


if __name__ == '__main__':
    main()
