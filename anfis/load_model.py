import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
import os
from torch.autograd import Variable
import plotly.graph_objects as go

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def metrics(dataset_name, columns_sel, n_feature, model, target):
    if model is None:
        model = torch.load('../models/model_' + dataset_name + '_' + str(n_feature) + '.h5')

    df_test = pd.read_csv("../datasets/" + dataset_name + "/" + dataset_name + "_test_" + target + ".csv", header=0, sep=',')
    df_test = df_test[columns_sel]

    image_all = df_test.values

    y_test = pd.read_csv("../datasets/" + dataset_name + "/" + dataset_name + "_test_" + target + ".csv", header=0, sep=',')
    y_test = y_test[y_test.columns[-1]]

    conv_test = torch.tensor(image_all, dtype=torch.float)

    pred = model(torch.Tensor(conv_test))
    pred2 = torch.argmax(pred, 1)

    pred2 = pred2.detach().numpy()

    results = classification_report(y_test, pred2, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, pred2)

    return results, conf_matrix
