from collections import Counter
from re import sub, search
import numpy as np
import optbinning
import warnings

# costruisco il metodo in modo tale che gli si passi il dataframe per intero, per poi andare ad escludere la colonna relativa alla classe
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


def values_dictionary(dataframe):
    dicts_list = []
    for i in range(0,
                   len(dataframe.columns) - 1):  # togliere il -1 se si vuole che anche la classe sia sostituita dalla frequenza
        col = dataframe[dataframe.columns[i]]
        if search("[a-zA-Z]", str(col.iloc[1])):
            dicts_list.append(dict(col.value_counts()))
    return dicts_list


def frequency_encoder(dataframe, dicts_list):
    count = 0
    i = 0
    while count < len(dicts_list):
        if search("[a-zA-Z]", str(dataframe.iloc[1, i])):
            dictionary = dicts_list[count]
            count += 1
            for k, v in dictionary.items():
                dataframe[dataframe.columns[i]].replace(k, v, inplace=True)
            dataframe[dataframe.columns[i]] = dataframe[dataframe.columns[i]].apply(
                lambda x: sub("((.)*([a-zA-Z]+)(.)*)+", '0', str(x))).astype(int)
        i += 1
    # questa parte serve ad applicare un ordinal encoder alla colonna della classe, in quanto serve per anfis questo formato
    col = dataframe.iloc[:, -1]
    count = 0
    for j in range(0, len(dataframe)):
        if search("[a-zA-Z]", str(col[j])):
            col.replace(col[j], str(count), inplace=True)
            count += 1
    dataframe.loc[:, -1] = col
    return dataframe


def chimerge(data, attr, label, max_intervals):  # scala lentamente
    warnings.filterwarnings("ignore")
    distinct_vals = sorted(set(data[attr]))  # Sort the distinct values
    labels = sorted(set(data[label]))  # Get all possible labels
    empty_count = {l: 0 for l in labels}  # A helper function for padding the Counter()
    intervals = [[distinct_vals[i], distinct_vals[i]] for i in
                 range(len(distinct_vals))]  # Initialize the intervals for each attribute
    while len(intervals) > max_intervals:  # While loop
        chi = []
        for i in range(len(intervals) - 1):
            # Calculate the Chi2 value
            obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
            obs1 = data[data[attr].between(intervals[i + 1][0], intervals[i + 1][1])]
            total = len(obs0) + len(obs1)
            count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
            count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
            count_total = count_0 + count_1
            expected_0 = count_total * sum(count_0) / total
            expected_1 = count_total * sum(count_1) / total
            chi_ = (count_0 - expected_0) ** 2 / expected_0 + (count_1 - expected_1) ** 2 / expected_1
            chi_ = np.nan_to_num(chi_)  # Deal with the zero counts
            chi.append(sum(chi_))  # Finally do the summation for Chi2
        min_chi = min(chi)  # Find the minimal Chi2 for current iteration
        for i, v in enumerate(chi):
            if v == min_chi:
                min_chi_index = i  # Find the index of the interval to be merged
                break
        new_intervals = []  # Prepare for the merged new data array
        skip = False
        done = False
        for i in range(len(intervals)):
            if skip:
                skip = False
                continue
            if i == min_chi_index and not done:  # Merge the intervals
                t = intervals[i] + intervals[i + 1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[i])
        intervals = new_intervals
    return intervals


def tree_discretization(dataframe,
                        column):  # bisogna specificare la profondità, quindi non va bene per il nostro caso, e non restituisce direttamente gli intervalli
    tree_model = DecisionTreeClassifier(max_depth=2)
    tree_model.fit(dataframe[column].to_frame(), dataframe.iloc[:, -1])
    dataframe['predictions'] = tree_model.predict_proba(dataframe[column].to_frame())[:, 1]
    results = pd.concat([dataframe.groupby(['predictions'])[column].min(),
                         dataframe.groupby(['predictions'])[column].max()], axis=1)
    print(results)


def mdlp(dataframe, column):  # non funziona probabilmente perche vuole solo interi
    discretizer = optbinning.MDLP(max_candidates=32)
    discretizer.fit(dataframe[column], dataframe.iloc[:, -1])
    print(discretizer.splits)


def optimal_binning(dataframe, col):  # funziona correttamente su maldroid, su KDD tende a creare 2 intervalli
    discretizer = optbinning.MulticlassOptimalBinning()
    discretizer.fit(dataframe[col], dataframe.iloc[:, -1])
    dataframe[col] = pd.DataFrame(discretizer.transform(dataframe[col], metric="bins", show_digits=7))
    print(dataframe[col].unique())


class Discretization: #anche questo ha il numero di intervalli come criterio di stop
    ''' A process that transforms quantitative data into qualitative data '''

    def __init__(cls):
        print('Data discretization process started')

    def get_new_intervals(cls, intervals, chi, min_chi):
        ''' To merge the interval based on minimum chi square value '''

        min_chi_index = np.where(chi == min_chi)[0][0]
        new_intervals = []
        skip = False
        done = False
        for i in range(len(intervals)):
            if skip:
                skip = False
                continue
            if i == min_chi_index and not done:
                t = intervals[i] + intervals[i + 1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[i])
        return new_intervals

    def get_chimerge_intervals(self, data, colName, label, max_intervals):
        '''
            1. Compute the χ 2 value for each pair of adjacent intervals
            2. Merge the pair of adjacent intervals with the lowest χ 2 value
            3. Repeat œ and  until χ 2 values of all adjacent pairs exceeds a threshold
        '''

        warnings.filterwarnings("ignore")
        cross_tab = pd.crosstab(data[colName], data[label])
        cross_stat = cross_tab.values
        cls_num = cross_stat.shape[-1]
        threshold = chi2.isf(0.05, df=cls_num - 1)

        # Getting unique values of input column
        distinct_vals = np.unique(data[colName])

        # Getting unique values of output column
        labels = np.unique(data[label])

        # Initially set the value to zero for all unique output column values
        empty_count = {l: 0 for l in labels}
        intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))]
        while len(intervals) > max_intervals:
            chi = []
            for i in range(len(intervals) - 1):
                # Find chi square for Interval 1
                row1 = data[data[colName].between(intervals[i][0], intervals[i][1])]
                # Find chi square for Interval 2
                row2 = data[data[colName].between(intervals[i + 1][0], intervals[i + 1][1])]
                total = len(row1) + len(row2)

                # Generate Contigency
                count_0 = np.array([v for i, v in {**empty_count, **Counter(row1[label])}.items()])
                count_1 = np.array([v for i, v in {**empty_count, **Counter(row2[label])}.items()])
                count_total = count_0 + count_1

                # Find the expected value by the following formula
                # Expected Value → ( Row Sum * Column Sum ) / Total Sum
                expected_0 = count_total * sum(count_0) / total
                expected_1 = count_total * sum(count_1) / total
                chi_ = (count_0 - expected_0) ** 2 / expected_0 + (count_1 - expected_1) ** 2 / expected_1

                # Store the chi value to find minimum chi value
                chi_ = np.nan_to_num(chi_)
                chi.append(sum(chi_))
            min_chi = min(chi)
            if min_chi > threshold:
                return intervals
            intervals = self.get_new_intervals(intervals, chi, min_chi)
        return intervals


def main():
    dataframe = pd.read_csv("../datasets/maldroid/maldroid_train.csv")
    column = "pread64:continuous"

    optimal_binning(dataframe, column)


if __name__ == '__main__':
    main()
