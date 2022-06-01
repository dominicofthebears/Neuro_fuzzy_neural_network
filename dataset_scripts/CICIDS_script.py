import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import type_of_target
import numpy as np

def minmax_scaler(dataframe):
    labels = dataframe.iloc[:, -1]
    dataframe = dataframe.drop(dataframe.columns[-1], axis=1)
    scaler = MinMaxScaler()
    scaler.fit(dataframe, labels)
    return scaler


def transform(dataframe, scaler):
    columns = dataframe.columns
    dataframe[columns[-1]] = dataframe[columns[-1]].apply(lambda x : '0' if x == "BENIGN" else '1' if x == 'PortScan' else '2' if x == 'DoS Hulk' else '3' if x == 'DDoS'
     else '4' if x == 'DoS GoldenEye' else '5' if x == 'FTP-Patator' else '6' if x == 'DoS slowloris' else '7' if x == 'SSH-Patator' else '8' if x == 'DoS Slowhttptest' else '0')
    labels = dataframe.iloc[:, -1]
    dataframe = dataframe.drop(columns[-1], axis=1)
    labels.reset_index(drop=True, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    dataframe = pd.DataFrame(scaler.transform(dataframe))
    dataframe = pd.concat([dataframe, labels.astype(int)], axis=1)
    dataframe.columns = columns
    return dataframe


def main():
    training_set = pd.read_csv("../datasets/CICIDS/train_CICIDS2017Multiclass_R.csv")
    test_set = pd.read_csv("../datasets/CICIDS/0_test_CICIDS2017Multiclass_R.csv")
    training_set.drop(['Fwd Header Length', 'Total Fwd Packet', 'Fwd Seg Size Min', 'FWD Init Win Bytes', 'SYN Flag Count',
      'Bwd Packet Length Mean'], axis=1, inplace=True)
    test_set.drop(['Fwd Header Length', 'Total Fwd Packet', 'Fwd Seg Size Min', 'FWD Init Win Bytes', 'SYN Flag Count',
      'Bwd Packet Length Mean'], axis=1, inplace=True)

    training_set.columns = ["Dst Port:discrete","Protocol:discrete","Flow Duration:continuous","Total Bwd packets:continuous","Total Length of Fwd Packet:continuous","Total Length of Bwd Packet:continuous","Fwd Packet Length Max:continuous","Fwd Packet Length Min:continuous","Fwd Packet Length Mean:continuous","Fwd Packet Length Std:continuous","Bwd Packet Length Max:continuous","Bwd Packet Length Min:continuous","Bwd Packet Length Std:continuous","Flow_Bytes:continuous","Flow_Packets:continuous","Flow IAT Mean:continuous","Flow IAT Std:continuous","Flow IAT Max:continuous","Flow IAT Min:continuous","Fwd IAT Total:continuous","Fwd IAT Mean:continuous","Fwd IAT Std:continuous","Fwd IAT Max:continuous","Fwd IAT Min:continuous","Bwd IAT Total:continuous","Bwd IAT Mean:continuous","Bwd IAT Std:continuous","Bwd IAT Max:continuous","Bwd IAT Min:continuous","Fwd PSH Flags:discrete","Bwd PSH Flags:discrete","Fwd URG Flags:discrete","Bwd URG Flags:discrete","Bwd Header Length:continuous","Fwd_Packets:continuous","Bwd Packets/s:continuous","Packet Length Min:continuous","Packet Length Max:continuous","Packet Length Mean:continuous","Packet Length Std:continuous","Packet Length Variance:continuous","FIN Flag Count:continuous","RST Flag Count:continuous","PSH Flag Count:continuous","ACK Flag Count:continuous","URG Flag Count:discrete","CWR Flag Count:discrete","ECE Flag Count:discrete","Down/Up Ratio:continuous","Average Packet Size:continuous","Fwd Segment Size Avg:continuous","Bwd Segment Size Avg::continuous","Fwd Bytes/Bulk Avg:continuous","Fwd Packet/Bulk Avg:continuous","Fwd Bulk Rate Avg:continuous","Bwd Bytes/Bulk Avg:continuous","Bwd Packet/Bulk Avg:continuous","Bwd Bulk Rate Avg:continuous","Subflow Fwd Packets:discrete","Subflow Fwd Bytes:continuous","Subflow Bwd Packets:discrete","Subflow Bwd Bytes:continuous","Bwd Init Win Bytes:continuous","Fwd Act Data Pkts:continuous","Active Mean:continuous","Active Std:continuous","Active Max:continuous","Active Min:continuous","Idle Mean:continuous","Idle Std:continuous","Idle Max:continuous","Idle Min:continuous","Classification:discrete"]
    test_set.columns = training_set.columns
    scaler = minmax_scaler(training_set)
    training_set = transform(training_set, scaler)
    test_set = transform(test_set, scaler)

    training_set.to_csv("../datasets/CICIDS/CICIDS_train_multiclass.csv", index=False)
    test_set.to_csv("../datasets/CICIDS/CICIDS_test_multiclass.csv", index=False)



if __name__ == '__main__':
    main()
