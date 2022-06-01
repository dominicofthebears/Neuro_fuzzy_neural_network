from sys import argv

from pandas import read_csv


def feature_type(dataframe):
    for i in range(0, len(dataframe.columns)):
        col = dataframe.iloc[:, i]
        print(col.nunique())


def values_range(dataframe):
    values = []
    for i in range(0, len(dataframe.columns)):
        col = dataframe.iloc[:, i]
        if col.nunique() >= 15:
            print(dataframe.columns[i], ", ", col.min(), ", ", col.max())

def values_frequencies(dataframe):
    for i in range(0, len(dataframe.columns)):
        col = dataframe.iloc[:, i]
        if col.nunique() < 15:
            print(col.value_counts())


def main():
    dataframe = read_csv(argv[1])
    values_range(dataframe)


if __name__ == '__main__':
    main()
