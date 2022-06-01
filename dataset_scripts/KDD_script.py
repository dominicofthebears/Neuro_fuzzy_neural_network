import re
import pandas as pd
import sklearn
from sklearn.preprocessing import OrdinalEncoder

from features_preprocessing.Preprocessing import values_dictionary, frequency_encoder


def training_set(dataframe):
    dataframe.columns = ["duration:continuous", "protocol_type:discrete", "service:discrete", "flag:discrete",
                         "src_bytes:continuous", "dst_bytes:continuous", "land:boolean",
                         "wrong_fragment:continuous", "urgent:continuous", "hot:continuous",
                         "num_failed_logins:continuous", "logged_in:boolean", "num_compromised:continuous",
                         "root_shell:continuous", "su_attempted:continuous", "num_root:continuous",
                         "num_file_creations:continuous", "num_shells:continuous", "num_access_files:continuous",
                         "num_outbound_cmds:continuous", "is_host_login:boolean", "is_guest_login:boolean",
                         "count:continuous", "srv_count:continuous", "serror_rate:continuous",
                         "srv_serror_rate:continuous", "rerror_rate:continuous", "srv_rerror_rate:continuous",
                         "same_srv_rate:continuous", "diff_srv_rate:continuous",
                         "srv_diff_host_rate:continuous", "dst_host_count:continuous", "dst_host_srv_count:continuous",
                         "dst_host_same_srv_rate:continuous", "dst_host_diff_srv_rate:continuous",
                         "dst_host_same_src_port_rate:continuous", "dst_host_srv_diff_host_rate:continuous",
                         "dst_host_serror_rate:continuous", "dst_host_srv_serror_rate:continuous",
                         "dst_host_rerror_rate:continuous", "dst_host_srv_rerror_rate:continuous",
                         "attack_type:discrete", "label"]

    dataframe.drop("label", axis=1, inplace=True)

    dataframe.replace(
        ['smurf', 'teardrop', 'pod', 'back', 'land', 'apache2', 'udpstorm', 'mailbomb', 'processtable', 'neptune'],
        'Dos', inplace=True)

    dataframe.replace(['ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan'], 'Probe', inplace=True)

    dataframe.replace(['dictionary', 'ftp_write', 'guess_passwd', 'imap', 'named', 'sendmail', 'spy', 'xlock', 'xsnoop',
                       'snmpgetattack', 'httptunnerl', 'worm', 'snmpguess', 'multihop', 'phf', 'warezclient',
                       'warezmaster',
                       'httptunnel', 'snmpguess'], 'R2L', inplace=True)

    dataframe.replace(
        ['perl', 'ps', 'xterm', 'loadmodule', 'eject', 'buffer_overflow', 'sqlattack', 'rootkit', 'loadmodule'], 'U2R',
        inplace=True)

    return dataframe


def test_set(dataframe):
    dataframe.columns = ["duration:continuous", "protocol_type:discrete", "service:discrete", "flag:discrete",
                         "src_bytes:continuous", "dst_bytes:continuous", "land:boolean",
                         "wrong_fragment:continuous", "urgent:continuous", "hot:continuous",
                         "num_failed_logins:continuous", "logged_in:boolean", "num_compromised:continuous",
                         "root_shell:continuous", "su_attempted:continuous", "num_root:continuous",
                         "num_file_creations:continuous", "num_shells:continuous", "num_access_files:continuous",
                         "num_outbound_cmds:continuous", "is_host_login:boolean", "is_guest_login:boolean",
                         "count:continuous", "srv_count:continuous", "serror_rate:continuous",
                         "srv_serror_rate:continuous", "rerror_rate:continuous", "srv_rerror_rate:continuous",
                         "same_srv_rate:continuous", "diff_srv_rate:continuous",
                         "srv_diff_host_rate:continuous", "dst_host_count:continuous", "dst_host_srv_count:continuous",
                         "dst_host_same_srv_rate:continuous", "dst_host_diff_srv_rate:continuous",
                         "dst_host_same_src_port_rate:continuous", "dst_host_srv_diff_host_rate:continuous",
                         "dst_host_serror_rate:continuous", "dst_host_srv_serror_rate:continuous",
                         "dst_host_rerror_rate:continuous", "dst_host_srv_rerror_rate:continuous",
                         "attack_type:discrete", "label"]


    dataframe.replace(
        ['smurf', 'teardrop', 'pod', 'back', 'land', 'apache2', 'udpstorm', 'mailbomb', 'processtable', 'neptune'],
        'Dos', inplace=True)

    dataframe.replace(['ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan'], 'Probe', inplace=True)

    dataframe.replace(['dictionary', 'ftp_write', 'guess_passwd', 'imap', 'named', 'sendmail', 'spy', 'xlock', 'xsnoop',
                       'snmpgetattack', 'httptunnerl', 'worm', 'snmpguess', 'multihop', 'phf', 'warezclient',
                       'warezmaster',
                       'httptunnel', 'snmpguess'], 'R2L', inplace=True)

    dataframe.replace(
        ['perl', 'ps', 'xterm', 'loadmodule', 'eject', 'buffer_overflow', 'sqlattack', 'rootkit', 'loadmodule'], 'U2R',
        inplace=True)


    #dataframe.drop("attack_type", axis=1, inplace=True)
    dataframe.drop("label", axis=1, inplace=True)


    return dataframe


def main():
    train = training_set(pd.read_csv("../datasets/KDD/KDDTrain+_20Percent.txt", sep=',', header=None))
    test = test_set(pd.read_csv("../datasets/KDD/KDDTest+.txt", sep=',', header=None))
    dicts_list = values_dictionary(train)
    train = frequency_encoder(train, dicts_list)
    test = frequency_encoder(test, dicts_list)
    train.to_csv("../datasets/KDD/KDD_train.csv", index=False)
    test.to_csv("../datasets/KDD/KDD_test.csv", index=False)


if __name__ == '__main__':
    main()
