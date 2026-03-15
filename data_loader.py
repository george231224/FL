import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataLoader:
    def __init__(self, dataset_name, data_dir='./data'):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    # NSL-KDD
    def load_nsl_kdd(self):

        train_file = os.path.join(self.data_dir, 'NSL-KDD', 'KDDTrain+.txt')
        test_file = os.path.join(self.data_dir, 'NSL-KDD', 'KDDTest+.txt')

        train_df = pd.read_csv(train_file, header=None)
        test_df = pd.read_csv(test_file, header=None)

        # 原始41个特征名
        feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]

        train_df.columns = feature_names + ['label', 'difficulty']
        test_df.columns = feature_names + ['label', 'difficulty']

        X_train = train_df[feature_names].copy()
        X_test = test_df[feature_names].copy()

        y_train = train_df['label'].str.rstrip('.')
        y_test = test_df['label'].str.rstrip('.')

        # 二分类映射
        y_train = np.where(y_train == 'normal', 0, 1)
        y_test = np.where(y_test == 'normal', 0, 1)

        # One-Hot 编码
        categorical_cols = ['protocol_type', 'service', 'flag']

        X_train = pd.get_dummies(X_train, columns=categorical_cols)
        X_test = pd.get_dummies(X_test, columns=categorical_cols)

        # 对齐列
        X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

        #  只对数值列做标准化
        numeric_cols = [col for col in X_train.columns
                        if not any(cat in col for cat in categorical_cols)]

        self.scaler.fit(X_train[numeric_cols])
        X_train[numeric_cols] = self.scaler.transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        return X_train.values.astype(float), y_train, \
               X_test.values.astype(float), y_test

    # UNSW-NB15
    def load_unsw_nb15(self):

        train_path = os.path.join(self.data_dir, 'UNSW_NB15', 'UNSW_NB15_training-set.csv')
        test_path = os.path.join(self.data_dir, 'UNSW_NB15', 'UNSW_NB15_testing-set.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_df = train_df.drop(['id', 'attack_cat'], axis=1, errors='ignore')
        test_df = test_df.drop(['id', 'attack_cat'], axis=1, errors='ignore')

        y_train = train_df['label'].values
        y_test = test_df['label'].values

        X_train = train_df.drop('label', axis=1)
        X_test = test_df.drop('label', axis=1)

        categorical_cols = ['proto', 'service', 'state']
        categorical_cols = [col for col in categorical_cols if col in X_train.columns]

        # One-Hot
        X_train = pd.get_dummies(X_train, columns=categorical_cols)
        X_test = pd.get_dummies(X_test, columns=categorical_cols)

        # 对齐
        X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

        # 只缩放数值列
        numeric_cols = [col for col in X_train.columns
                        if not any(cat in col for cat in categorical_cols)]

        self.scaler.fit(X_train[numeric_cols])
        X_train[numeric_cols] = self.scaler.transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        return X_train.values.astype(float), y_train, \
               X_test.values.astype(float), y_test

    def load_data(self):
        if self.dataset_name == 'NSL-KDD':
            return self.load_nsl_kdd()
        elif self.dataset_name == 'UNSW-NB15':
            return self.load_unsw_nb15()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
