import numpy as np
import pandas
import pandas as pd
from Orange import classification, data
from Orange.data import Domain, DiscreteVariable
from sklearn.metrics import classification_report, confusion_matrix
import torch

import anfis.load_model
from features_preprocessing import Feature_ranking


def cn2_metrics(dataset_name, columns_sel, model):
    df_test = pd.read_csv("../datasets/" + dataset_name + "/" + dataset_name + '_test.csv', header=0, sep=',')
    df_test = df_test[columns_sel]

    image_all = df_test.values

    y_test = pd.read_csv("../datasets/" + dataset_name + "/" + dataset_name + "_test.csv", header=0, sep=',')
    y_test = y_test[y_test.columns[-1]]

    conv_test = np.asarray(image_all)

    pred = model.predict(conv_test)
    pred2 = np.zeros(len(pred))
    for i in range(0, len(pred)):
        pred2[i] = list(pred[i]).index(pred[i].max())

    results = classification_report(y_test, pred2, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, pred2)

    return results, conf_matrix


def main():
    dataset = "CICIDS"
    train_csv = pd.read_csv("../datasets/" + dataset + '/' + dataset + "_train.csv", header=0, sep=',')
    target_values = train_csv.iloc[:, -1].unique()
    targets = train_csv.iloc[:, -1].to_numpy()

    n_features = 3
    features = list(Feature_ranking.mutual_info(train_csv).keys())
    istances = pd.DataFrame()
    for i in range(0, n_features):
        istances = pd.concat([istances, train_csv[features[i]]], axis=1)

    istances_domain = Domain.from_numpy(istances.to_numpy())
    final_domain = Domain([value for value in istances_domain],
                          DiscreteVariable.make("Class", values=(str(val) for val in target_values)))
    train_table = data.Table.from_numpy(final_domain, istances, targets)

    cn2_learner = classification.rules.CN2Learner()
    cn2_learner.rule_finder.search_strategy.constrain_continuous = True

    cn2_classifier = cn2_learner(train_table)
    print(len(cn2_classifier.rule_list))

    results, conf_matrix = (cn2_metrics(dataset, istances.columns, cn2_classifier))
    f = open("../cn2/cn2_results/" + dataset + '_' + str(n_features) + ".csv", "w")
    f.write("SELECTED COLUMNS;" + "\n")
    for element in istances.columns:
        f.write(element + "\n")
    f.write("--------------------------------;" + '\n')
    for key, value in results.items():
        f.write(str(key) + ":" + str(value) + '\n')
    f.write("--------------------------------;" + '\n')
    f.write(str(conf_matrix))
    f.close()


if __name__ == '__main__':
    main()
