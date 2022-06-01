from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from anfis import load_model
from membership import make_anfis, make_anfis_T
import experimental
# import load_model
from torch.utils.data import Dataset, DataLoader
import numpy as np

seed = 123
np.random.seed(seed)
import torch

torch.manual_seed(seed)


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def make_one_hot(data, num_categories, dtype=torch.float):
    num_entries = len(data)
    # Convert data to a torch tensor of indices, with extra dimension:
    cats = torch.Tensor(data).long().unsqueeze(1)
    # Now convert this to one-hot representation:
    y = torch.zeros((num_entries, num_categories), dtype=dtype) \
        .scatter(1, cats, 1)
    y.requires_grad = True
    return y


def get_data_one_hot(dataset, n_feature, batch_size, columns_sel, targets):
    # columns_sel.append('Classification')

    dataframe = dataset[columns_sel]  # prendere le colonne FS dal dataset intero.

    array = dataframe.values
    d_data = array[:, 0:len(dataframe.columns)]

    d_target = dataset.iloc[:, -1].to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(d_data, d_target, test_size=0.2, stratify=d_target,
                                                      random_state=69, shuffle=True)

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())

    x = torch.Tensor(X_train)
    y = make_one_hot(y_train, num_categories=targets)
    td = TensorDataset(x, y)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=False), DataLoader(val_dataset,batch_size=batch_size), DataLoader(
        td, batch_size=batch_size, shuffle=False)


def train(dataframe, n_feature, learning_rate, bs, num_items, columns_sel, approach, dataset):
    targets = len(dataframe.iloc[:, -1].value_counts())
    train_data, val_data, x = get_data_one_hot(dataframe, n_feature, bs, columns_sel, targets)
    x_train, y_train = x.dataset.tensors
    if approach == 'gaussian':
        model = make_anfis(x_train, num_mfs=num_items, num_out=targets, hybrid=False)
    else:
        model = make_anfis_T(x_train, num_mfs=num_items, num_out=targets, hybrid=False, dataframe=dataframe[columns_sel])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, score, best_epoch = experimental.train_anfis_cat(model, train_data, val_data, optimizer, 150, targets)
    torch.save(model, '../models/model_' + dataset + '_' + str(n_feature) + '_' + approach + '.h5')
    return model


def opt(dataframe, n_feature, learning_rate, bs, num_items, columns_sel, approach):
    targets = len(dataframe.iloc[:, -1].value_counts())
    train_data, val_data, x = get_data_one_hot(dataframe, n_feature, bs, columns_sel, targets)
    x_train, y_train = x.dataset.tensors
    if approach == 'gaussian':
        model = make_anfis(x_train, num_mfs=num_items, num_out=targets, hybrid=False)
    else:
        model = make_anfis_T(x_train, num_mfs=num_items, num_out=targets, hybrid=False, dataframe=dataframe[columns_sel])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, scores, epoch = experimental.train_anfis_cat(model, train_data, val_data, optimizer, 150, targets)

    return model, scores, epoch
