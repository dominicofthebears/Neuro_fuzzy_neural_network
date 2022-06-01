import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def features_cleaning():
    dataframe = pd.read_csv("../datasets/maldroid/feature_vectors_syscallsbinders_frequency_5_Cat.csv")
    cleaned_dataframe = dataframe[
        ["getInstallerPackageName", "_newselect", "mprotect", "getPackageInfo", "CREATE_FOLDER_____", "getReceiverInfo",
         "mkdir", "pwrite64", "getServiceInfo",
         "pread64", "unlink", "munmap", "statfs64", "getActivityInfo", "fdatasync", "FS_PIPE_ACCESS___",
         "FS_ACCESS(READ)____", "FS_PIPE_ACCESS(WRITE)___",
         "FS_ACCESS____", "mmap2", "FS_ACCESS()____", "pipe", "fcntl64", "FS_ACCESS(CREATE__WRITE)__", "nanosleep",
         "getApplicationInfo", "FS_ACCESS(CREATE__READ__WRITE)",
         "rename", "access", "chmod", "gettid", "FS_ACCESS(CREATE)____", "brk", "DEVICE_ACCESS_____",
         "queryIntentServices", "NETWORK_ACCESS____", "stat64",
         "FS_ACCESS(WRITE)____", "open", "lstat64", "Class"]]

    cleaned_dataframe.columns = ["getInstallerPackageName:continuous", "_newselect:continuous", "mprotect:continuous", "getPackageInfo:continuous", "CREATE_FOLDER_____:continuous", "getReceiverInfo:continuous",
         "mkdir:continuous", "pwrite64:continuous", "getServiceInfo:continuous",
         "pread64:continuous", "unlink:continuous", "munmap:continuous", "statfs64:continuous", "getActivityInfo:continuous", "fdatasync:continuous", "FS_PIPE_ACCESS___:continuous",
         "FS_ACCESS(READ)____:continuous", "FS_PIPE_ACCESS(WRITE)___:continuous",
         "FS_ACCESS____:continuous", "mmap2:continuous", "FS_ACCESS()____:continuous", "pipe:continuous", "fcntl64:continuous", "FS_ACCESS(CREATE__WRITE)__:continuous", "nanosleep:continuous",
         "getApplicationInfo:continuous", "FS_ACCESS(CREATE__READ__WRITE):continuous",
         "rename:continuous", "access:continuous", "chmod:continuous", "gettid:continuous", "FS_ACCESS(CREATE)____:continuous", "brk:continuous", "DEVICE_ACCESS_____:continuous",
         "queryIntentServices:continuous", "NETWORK_ACCESS____:continuous", "stat64:continuous",
         "FS_ACCESS(WRITE)____:continuous", "open:continuous", "lstat64:continuous", "Class:discrete"]

    cleaned_dataframe.to_csv("../datasets/maldroid/feature_filtered_maldroid.csv", index=False)


def minmax_scaler(dataframe):
    labels = dataframe['Class:discrete']
    dataframe = dataframe.drop(['Class:discrete'], axis=1)
    scaler = MinMaxScaler()
    scaler.fit(dataframe, labels)
    return scaler


def transform(dataframe, scaler):
    columns = dataframe.columns
    labels = dataframe['Class:discrete']
    dataframe = dataframe.drop(['Class:discrete'], axis=1)
    labels.reset_index(drop=True, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    dataframe = pd.DataFrame(scaler.transform(dataframe))
    dataframe = pd.concat([dataframe, labels.astype(int)], axis=1)
    dataframe.columns = columns
    return dataframe



def main():
    features_cleaning()
    dataframe = pd.read_csv("../datasets/maldroid/feature_filtered_maldroid.csv")
    dataframe['Class:discrete'] = dataframe['Class:discrete'].apply(lambda x: 0 if x == 5 else 1)
    train, test = train_test_split(dataframe, test_size=0.3, stratify=dataframe["Class:discrete"], shuffle=True)
    # test.drop("Class:discrete", axis=1, inplace=True)
    scaler = minmax_scaler(train)
    train = transform(train, scaler)
    test = transform(test, scaler)
    train.to_csv("../datasets/maldroid/maldroid_train_binary.csv", index=False)
    test.to_csv("../datasets/maldroid/maldroid_test_binary.csv", index=False)


if __name__ == '__main__':
    main()
