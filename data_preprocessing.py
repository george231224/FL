import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'  # 防止 joblib 在 Windows 上检测 CPU 核心数报错

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score


def mrmr_feature_selection(X_df, y, categorical_cols=None, n_bins=15, threshold=0.8, top_k=None):
    """MRMR 特征选择 — 标准贪心前向搜索 + 预计算MI矩阵 + 截断

    截断方式（二选一）:
      - top_k 不为 None: 硬截断，选前 20 个特征
      - top_k 为 None: 用 Relevance 归一化累加，达到 threshold 截断

    参数:
        X_df: pd.DataFrame, 原始特征（含类别列，未 One-Hot）
        y: np.ndarray, 标签
        categorical_cols: list[str], 类别特征列名
        n_bins: 连续特征等频分箱数（默认 15）
        threshold: 累计重要性阈值（top_k=None 时使用，默认 0.8）
        top_k: 硬截断特征数（优先于 threshold）

    返回:
        selected_cols: list[str], 选中的列名（按贪心选择顺序）
    """
    if categorical_cols is None:
        categorical_cols = []

    col_names = list(X_df.columns)
    n_all = len(col_names)

    print(f"\n[MRMR] 特征选择 (贪心前向 + 离散MI + 累计阈值={threshold})")
    print(f"  总特征: {n_all} (类别: {len(categorical_cols)}, 连续: {n_all - len(categorical_cols)})")
    print(f"  分箱数: {n_bins}")

    # ── Step 0: 全离散化 ──
    X_disc = np.empty((len(X_df), n_all), dtype=int)
    for j, col in enumerate(col_names):
        vals = X_df[col].values
        if col in categorical_cols:
            X_disc[:, j] = LabelEncoder().fit_transform(vals.astype(str))
        else:
            try:
                X_disc[:, j] = pd.qcut(vals, q=n_bins, labels=False, duplicates='drop')
            except Exception:
                X_disc[:, j] = 0
    print(f"  离散化完成: {X_disc.shape}")

    # ── Step 1: Relevance = MI(feature, target) ──
    y_int = y.astype(int)
    relevance = np.array([mutual_info_score(X_disc[:, j], y_int) for j in range(n_all)])

    print(f"  Relevance 统计: max={relevance.max():.4f}, "
          f"mean={relevance.mean():.4f}, min={relevance.min():.4f}")
    rank_by_rel = np.argsort(-relevance)
    for i in range(min(10, n_all)):
        idx = rank_by_rel[i]
        print(f"    {i+1:2d}. {col_names[idx]:20s}  Rel={relevance[idx]:.4f}")

    # ── Step 2: 预计算特征间 MI 矩阵 ──
    print(f"  预计算 {n_all}×{n_all} MI 矩阵...")
    mi_matrix = np.zeros((n_all, n_all))
    for i in range(n_all):
        for j in range(i + 1, n_all):
            mi_val = mutual_info_score(X_disc[:, i], X_disc[:, j])
            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val
    print(f"  MI 矩阵完成: max={mi_matrix.max():.4f}")

    # ── Step 3: 标准贪心前向选择（矩阵查表，瞬间完成）──
    selected = []
    remaining = set(range(n_all))
    redundancy_sum = np.zeros(n_all)  # 累计冗余（增量更新）

    # 第 1 个特征：纯 Relevance 最大
    first = int(np.argmax(relevance))
    selected.append(first)
    remaining.remove(first)

    # 贪心循环：遍历所有特征
    for step in range(1, n_all):
        if not remaining:
            break
        # 增量更新：加上最新选入特征与所有候选的 MI
        redundancy_sum += mi_matrix[selected[-1]]

        best_score, best_feat = -np.inf, -1
        for f in remaining:
            avg_red = redundancy_sum[f] / len(selected)
            score = relevance[f] - avg_red
            if score > best_score:
                best_score, best_feat = score, f

        if best_feat < 0:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)

    # 打印贪心排序结果
    print(f"\n  贪心 MRMR 排序 (共 {len(selected)} 个特征):")
    for i, idx in enumerate(selected):
        print(f"    {i+1:2d}. {col_names[idx]:20s}  Rel={relevance[idx]:.4f}")

    # ── Step 4: 截断 ──
    sel_relevance = np.array([relevance[i] for i in selected])
    rel_sum = sel_relevance.sum()
    if rel_sum > 0:
        norm_rel = sel_relevance / rel_sum
    else:
        norm_rel = np.ones(len(selected)) / len(selected)
    cumsum = np.cumsum(norm_rel)

    if top_k is not None:
        # 硬截断: 取前 top_k 个
        cutoff = min(top_k, len(selected))
        print(f"\n  Top-{top_k} 硬截断:")
    else:
        # 累计阈值截断
        cutoff = np.searchsorted(cumsum, threshold) + 1
        cutoff = max(cutoff, 3)
        cutoff = min(cutoff, len(selected))
        print(f"\n  累计 Relevance 截断 (阈值={threshold}):")

    for i in range(cutoff):
        idx = selected[i]
        print(f"    {i+1:2d}. {col_names[idx]:20s}  "
              f"Rel={relevance[idx]:.4f}  norm={norm_rel[i]:.4f}  "
              f"cumsum={cumsum[i]:.4f}"
              f"{'  ← 截断' if i == cutoff - 1 else ''}")
    if cutoff < len(selected):
        print(f"    ... 剩余 {len(selected) - cutoff} 个特征被截断")

    final_indices = selected[:cutoff]
    selected_cols = [col_names[i] for i in final_indices]
    print(f"\n[MRMR] 完成: {n_all} → {len(selected_cols)} 维 "
          f"(累计Rel={cumsum[cutoff-1]:.4f})")
    print(f"  类别特征: {[c for c in selected_cols if c in categorical_cols]}")
    print(f"  连续特征: {[c for c in selected_cols if c not in categorical_cols]}")
    return selected_cols


def _greedy_correlation_order(X_df, feature_names, anchor_order):
    """用贪心相关性把相似连续特征排到相邻位置。"""
    if len(feature_names) <= 2:
        return list(feature_names)

    corr = X_df[feature_names].corr(method='spearman').abs().fillna(0.0)
    rank = {name: idx for idx, name in enumerate(anchor_order)}
    ordered = [anchor_order[0]]
    remaining = set(feature_names)
    remaining.remove(anchor_order[0])

    while remaining:
        tail = ordered[-1]
        next_feat = max(
            remaining,
            key=lambda name: (float(corr.loc[tail, name]), -rank[name])
        )
        ordered.append(next_feat)
        remaining.remove(next_feat)
    return ordered


def _unsw_semantic_group(feature_name):
    """UNSW-NB15 连续特征的粗粒度语义分组。"""
    groups = [
        ({'dur', 'rate'}, 0),
        ({'sbytes', 'sload', 'smean', 'sloss', 'spkts', 'sttl', 'stcpb', 'sjit', 'swin', 'sinpkt'}, 1),
        ({'dbytes', 'dload', 'dmean', 'dloss', 'dpkts', 'dttl', 'dtcpb', 'djit', 'dwin', 'dinpkt'}, 2),
        ({'tcprtt', 'synack', 'ackdat'}, 3),
        ({'response_body_len', 'trans_depth', 'is_sm_ips_ports'}, 4),
    ]
    for features, group_id in groups:
        if feature_name in features:
            return group_id
    if feature_name.startswith('ct_'):
        return 5
    return 6


def reorder_unsw_feature_columns(X_train_df, selected_cols, categorical_cols, strategy='mrmr'):
    """重排 UNSW-NB15 特征列，保持类别特征始终位于末尾。"""
    selected_cat_cols = [c for c in categorical_cols if c in selected_cols]
    selected_num_cols = [c for c in selected_cols if c not in selected_cat_cols]

    if strategy == 'mrmr':
        ordered_num_cols = list(selected_num_cols)
    elif strategy == 'corr_greedy':
        ordered_num_cols = _greedy_correlation_order(X_train_df, selected_num_cols, selected_num_cols)
    elif strategy == 'semantic_group':
        mrmr_rank = {name: idx for idx, name in enumerate(selected_num_cols)}
        ordered_num_cols = sorted(
            selected_num_cols,
            key=lambda name: (_unsw_semantic_group(name), mrmr_rank[name])
        )
    else:
        raise ValueError(f"不支持的特征排序策略: {strategy}")

    return ordered_num_cols + selected_cat_cols, ordered_num_cols, selected_cat_cols


class NSLKDDPreprocessor:
    # 多分类：5类标签名（用于打印和评估）
    MULTI_CLASS_NAMES = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}

    def __init__(self, data_dir='./data', classification='multi', test_size=0.2, seed=42,
                 feature_order='mrmr'):
        """
        classification: 'binary' → Normal=0 / Attack=1
                        'multi'  → Normal=0 / DoS=1 / Probe=2 / R2L=3 / U2R=4
        test_size: 测试集比例 (default: 0.2)
        seed: 随机种子
        """
        self.data_dir = data_dir
        self.classification = classification
        self.test_size = test_size
        self.seed = seed
        self.scaler = RobustScaler(quantile_range=(10, 90))  # MinMaxScaler→RobustScaler: 对网络流量极端离群值更稳健
        self.label_encoder = LabelEncoder()
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self):
        return ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

    def _get_label_mapping(self):
        """返回原始标签 → 类别名称的映射"""
        if self.classification == 'binary':
            dos   = {'back','land','neptune','pod','smurf','teardrop',
                     'mailbomb','apache2','processtable','udpstorm'}
            probe = {'ipsweep','nmap','portsweep','satan','mscan','saint'}
            r2l   = {'ftp_write','guess_passwd','imap','multihop','phf','spy',
                     'warezclient','warezmaster','sendmail','named','snmpgetattack',
                     'snmpguess','xlock','xsnoop','worm'}
            u2r   = {'buffer_overflow','loadmodule','perl','rootkit','httptunnel',
                     'ps','sqlattack','xterm'}
            mapping = {'normal': 'normal'}
            for t in dos | probe | r2l | u2r:
                mapping[t] = 'attack'
            return mapping
        else:  # multi
            dos   = {'back','land','neptune','pod','smurf','teardrop',
                     'mailbomb','apache2','processtable','udpstorm'}
            probe = {'ipsweep','nmap','portsweep','satan','mscan','saint'}
            r2l   = {'ftp_write','guess_passwd','imap','multihop','phf','spy',
                     'warezclient','warezmaster','sendmail','named','snmpgetattack',
                     'snmpguess','xlock','xsnoop','worm'}
            u2r   = {'buffer_overflow','loadmodule','perl','rootkit','httptunnel',
                     'ps','sqlattack','xterm'}
            mapping = {'normal': 'normal'}
            for t in dos:   mapping[t] = 'dos'
            for t in probe: mapping[t] = 'probe'
            for t in r2l:   mapping[t] = 'r2l'
            for t in u2r:   mapping[t] = 'u2r'
            return mapping

    def load_and_preprocess(self):
        train_df = pd.read_csv(f'{self.data_dir}/NSL-KDD/KDDTrain+.txt', header=None)
        test_df  = pd.read_csv(f'{self.data_dir}/NSL-KDD/KDDTest+.txt',  header=None)

        train_df.columns = self.feature_names + ['label', 'difficulty']
        test_df.columns  = self.feature_names + ['label', 'difficulty']

        # 合并两个文件，统一重新划分（保证训练/测试同分布）
        df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"\n合并数据量: {len(df):,} 行 (KDDTrain+: {len(train_df):,}, KDDTest+: {len(test_df):,})")

        # One-Hot 编码（在合并数据上统一编码，自动覆盖所有 service/flag 取值）
        categorical_cols = ['protocol_type', 'service', 'flag']
        print(f"使用 One-Hot 编码处理分类特征: {categorical_cols}")
        print(f"原始特征维度: {len(self.feature_names)}")

        X_all = pd.get_dummies(df[self.feature_names].copy(), columns=categorical_cols)
        print(f"One-Hot 编码后特征维度: {len(X_all.columns)}")
        print(f"分类模式: {'二分类 (Normal/Attack)' if self.classification == 'binary' else '多分类 (Normal/DoS/Probe/R2L/U2R)'}")

        # 标签映射
        label_mapping = self._get_label_mapping()
        y_str = df['label'].str.rstrip('.').map(label_mapping).values

        # 显式整数映射，确保 Normal=0
        if self.classification == 'binary':
            int_map = {'normal': 0, 'attack': 1}
            self.label_encoder.classes_ = np.array(['normal', 'attack'])
            # 未覆盖的攻击类型（KDDTest+ 特有新型攻击）标记为 Attack
            y_all = pd.Series(y_str).map(int_map).fillna(1).values.astype(int)
        else:
            int_map = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
            self.label_encoder.classes_ = np.array(['normal', 'dos', 'probe', 'r2l', 'u2r'])
            y_all = pd.Series(y_str).map(int_map).fillna(0).values.astype(int)

        # 分层划分，保证训练集与测试集类别比例一致
        X_arr = X_all.values.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_all,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y_all
        )
        print(f"分层划分完成: 训练+验证 {len(y_train):,}  测试 {len(y_test):,}")

        # 归一化：仅在训练集上 fit，避免数据泄露
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        # 打印类别分布
        class_names = self.label_encoder.classes_
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n训练集类别分布:")
        for u, c in zip(unique, counts):
            print(f"  {class_names[u]:10s} ({u}): {c:6d} ({c/len(y_train)*100:.2f}%)")

        unique_t, counts_t = np.unique(y_test, return_counts=True)
        print(f"测试集类别分布:")
        for u, c in zip(unique_t, counts_t):
            print(f"  {class_names[u]:10s} ({u}): {c:6d} ({c/len(y_test)*100:.2f}%)")

        return X_train, y_train, X_test, y_test


class UNSWNB15Preprocessor:
    # 多分类：10类标签名（Normal=0 + 9种攻击，按字母序排列保持 normal 在首位）
    MULTI_CLASS_NAMES = {
        0: 'Normal', 1: 'Analysis', 2: 'Backdoor', 3: 'DoS',
        4: 'Exploits', 5: 'Fuzzers', 6: 'Generic',
        7: 'Reconnaissance', 8: 'Shellcode', 9: 'Worms'
    }

    def __init__(self, data_dir='./data', classification='multi', test_size=0.2, seed=42,
                 feature_order='mrmr'):
        """
        classification: 'binary' → Normal=0 / Attack=1  (使用 label 列)
                        'multi'  → Normal=0 + 9种攻击类型  (使用 attack_cat 列)
        test_size: 测试集比例 (default: 0.2)
        seed: 随机种子
        """
        self.data_dir = data_dir
        self.classification = classification
        self.test_size = test_size
        self.seed = seed
        self.feature_order = feature_order
        self.label_encoder = LabelEncoder()

    def load_and_preprocess(self):
        train_df = pd.read_csv(f'{self.data_dir}/UNSW_NB15/UNSW_NB15_training-set.csv')
        test_df  = pd.read_csv(f'{self.data_dir}/UNSW_NB15/UNSW_NB15_testing-set.csv')

        # 合并两个文件，统一重新划分（保证训练/测试同分布）
        df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"\n合并数据量: {len(df):,} 行 (training-set: {len(train_df):,}, testing-set: {len(test_df):,})")

        # 标签提取（在 drop 特征列之前）
        if self.classification == 'binary':
            y_all = df['label'].values.astype(int)
            self.label_encoder.classes_ = np.array(['normal', 'attack'])
        else:
            cat_all = df['attack_cat'].fillna('normal').str.strip().str.lower()
            int_map = {
                'normal':         0,
                'analysis':       1,
                'backdoor':       2,
                'dos':            3,
                'exploits':       4,
                'fuzzers':        5,
                'generic':        6,
                'reconnaissance': 7,
                'shellcode':      8,
                'worms':          9,
            }
            y_all = cat_all.map(int_map).fillna(0).values.astype(int)
            self.label_encoder.classes_ = np.array([
                'normal', 'analysis', 'backdoor', 'dos',
                'exploits', 'fuzzers', 'generic',
                'reconnaissance', 'shellcode', 'worms'
            ])

        # 删除 id、label、attack_cat 及低信息量列
        # ct_ftp_cmd 含空白非数字条目，is_ftp_login 几乎全为0，ct_flw_http_mthd 大量空值
        drop_cols = ['id', 'label', 'attack_cat',
                     'ct_ftp_cmd', 'is_ftp_login', 'ct_flw_http_mthd']
        X_all = df.drop(columns=drop_cols, errors='ignore')

        # 清洗 NaN / Inf（网络流量数据常见异常值）
        X_all = X_all.replace([np.inf, -np.inf], np.nan)
        X_all = X_all.fillna(0)

        print(f"原始特征维度: {len(X_all.columns)} (含类别列 proto/service/state)")
        print(f"分类模式: {'二分类 (Normal/Attack)' if self.classification == 'binary' else '多分类 (Normal + 9种攻击)'}")

        # 分层划分（在原始 DataFrame 上，保留列名用于 MRMR）
        categorical_cols = [c for c in ['proto', 'service', 'state'] if c in X_all.columns]
        train_idx, test_idx = train_test_split(
            np.arange(len(y_all)),
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y_all
        )
        X_train_df = X_all.iloc[train_idx].reset_index(drop=True)
        X_test_df  = X_all.iloc[test_idx].reset_index(drop=True)
        y_train = y_all[train_idx]
        y_test  = y_all[test_idx]
        print(f"分层划分完成: 训练+验证 {len(y_train):,}  测试 {len(y_test):,}")

        # MRMR 特征选择：不硬截断，用累计阈值 threshold=0.95 自动选择
        # v8: 从 top_k=20 改为 threshold 模式，保留更多有意义的特征
        # 20维丢掉近半信息(cumsum=0.57)，且包含4个零Relevance特征
        selected_cols = mrmr_feature_selection(
            X_train_df, y_train,
            categorical_cols=categorical_cols,
            n_bins=15,
            top_k=None,
            threshold=0.95
        )
        final_selected_cols, selected_num_cols, selected_cat_cols = reorder_unsw_feature_columns(
            X_train_df,
            selected_cols,
            categorical_cols=categorical_cols,
            strategy=self.feature_order
        )
        self.mrmr_selected_feature_names_ = selected_cols
        self.selected_feature_names_ = final_selected_cols
        self.selected_num_cols_ = selected_num_cols
        self.selected_cat_cols_ = selected_cat_cols
        self.feature_order_strategy_ = self.feature_order

        # 仅保留选中的列（.copy() 避免 SettingWithCopyWarning 和视图问题）
        X_train_df = X_train_df[final_selected_cols].copy()
        X_test_df  = X_test_df[final_selected_cols].copy()

        # LabelEncoding: 分类特征转为整数编码
        self._label_encoders = {}
        for col in selected_cat_cols:
            le = LabelEncoder()
            X_train_df[col] = le.fit_transform(X_train_df[col].astype(str))
            X_test_df[col] = X_test_df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0
            )
            self._label_encoders[col] = le

        # ColumnTransformer: 全部用 RobustScaler（LabelEncoded 列也做缩放）
        ct = ColumnTransformer(transformers=[
            ('num', RobustScaler(quantile_range=(10, 90)), final_selected_cols),
        ])
        X_train = ct.fit_transform(X_train_df)
        X_test  = ct.transform(X_test_df)
        self.column_transformer_ = ct

        # CNN 只处理前 n_continuous_ 列（连续特征），类别特征在后面
        self.n_continuous_ = len(selected_num_cols)  # CNN 只处理前这么多列
        self.n_categorical_ = len(selected_cat_cols)  # 类别特征数量
        # 记录各类别特征的类别数（用于 embedding）
        self.cat_cardinalities_ = [len(self._label_encoders[c].classes_) for c in selected_cat_cols]

        # 打印维度信息
        print(f"\n[预处理] LabelEncoding + RobustScaler")
        print(f"  连续特征排序策略: {self.feature_order}")
        print(f"  连续特征: {len(selected_num_cols)} 列 (→ CNN)")
        print(f"  连续特征顺序前10: {selected_num_cols[:10]}")
        if selected_cat_cols:
            print(f"  类别特征: {selected_cat_cols} → LabelEncoding (→ XGBoost, 不进 CNN)")
            print(f"  类别基数: {dict(zip(selected_cat_cols, self.cat_cardinalities_))}")
        print(f"  最终特征维度: {X_train.shape[1]} (连续: {self.n_continuous_}, 类别: {self.n_categorical_})")

        # 打印类别分布
        class_names = self.label_encoder.classes_
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n训练集类别分布:")
        for u, c in zip(unique, counts):
            name = class_names[u] if u < len(class_names) else str(u)
            print(f"  {name:15s} ({u}): {c:6d} ({c/len(y_train)*100:.2f}%)")

        unique_t, counts_t = np.unique(y_test, return_counts=True)
        print(f"测试集类别分布:")
        for u, c in zip(unique_t, counts_t):
            name = class_names[u] if u < len(class_names) else str(u)
            print(f"  {name:15s} ({u}): {c:6d} ({c/len(y_test)*100:.2f}%)")

        return X_train, y_train, X_test, y_test


class CICIDS2017Preprocessor:
    """
    CIC-IDS2017 数据集预处理器
    数据来源: cicids2017_cleaned.csv (单文件，需内部划分训练/测试集)
    特征: 52个数值型特征，无需 One-Hot 编码
    类别:
      多分类 (multi): Normal=0, DoS=1, DDoS=2, PortScan=3, BruteForce=4, WebAttacks=5, Bots=6
      二分类 (binary): Normal=0, Attack=1
    """

    MULTI_CLASS_NAMES = {
        0: 'Normal',
        1: 'DoS',
        2: 'DDoS',
        3: 'PortScan',
        4: 'BruteForce',
        5: 'WebAttacks',
        6: 'Bots',
    }

    # 原始标签 → 类别名称映射
    _LABEL_MAP_MULTI = {
        'Normal Traffic': 'normal',
        'DoS':            'dos',
        'DDoS':           'ddos',
        'Port Scanning':  'portscan',
        'Brute Force':    'bruteforce',
        'Web Attacks':    'webattacks',
        'Bots':           'bots',
    }

    _INT_MAP_MULTI = {
        'normal':      0,
        'dos':         1,
        'ddos':        2,
        'portscan':    3,
        'bruteforce':  4,
        'webattacks':  5,
        'bots':        6,
    }

    def __init__(self, data_dir='./data', classification='multi',
                 test_size=0.2, sample_size=None, seed=42):
        """
        参数:
            data_dir: 数据根目录，CSV 文件路径为 {data_dir}/CIC-IDS2017/cicids2017_cleaned.csv
            classification: 'binary' 或 'multi'
            test_size: 测试集比例 (default: 0.2，即 80% train+val / 20% test)
            sample_size: 采样数量。None 表示使用全量数据；
                         整数 N 表示从全量数据中按类别比例采样 N 条记录，
                         加快大数据集的训练速度。
            seed: 随机种子
        """
        self.data_dir = data_dir
        self.classification = classification
        self.test_size = test_size
        self.sample_size = sample_size
        self.seed = seed
        self.scaler = RobustScaler(quantile_range=(10, 90))
        self.label_encoder = LabelEncoder()

    def load_and_preprocess(self):
        """
        加载并预处理 CIC-IDS2017 数据集。

        返回:
            X_train (ndarray): 训练特征 (含验证集，main.py 会进一步拆分)
            y_train (ndarray): 训练标签
            X_test  (ndarray): 测试特征
            y_test  (ndarray): 测试标签
        """
        csv_path = f'{self.data_dir}/CIC-IDS2017/cicids2017_cleaned.csv'
        print(f"\n正在读取 CIC-IDS2017: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"原始数据量: {len(df):,} 行, {df.shape[1]} 列")

        # ── 1. 清洗 ──────────────────────────────────────────────────────────
        # 替换 inf/-inf 为 NaN 后丢弃（保险起见）
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        before = len(df)
        df.dropna(inplace=True)
        if len(df) < before:
            print(f"丢弃含 NaN/inf 行: {before - len(df):,}")

        # ── 2. 可选采样 ───────────────────────────────────────────────────────
        if self.sample_size is not None and self.sample_size < len(df):
            # 按类别分层采样，保持原始类别比例
            total = len(df)
            df = df.groupby('Attack Type', group_keys=False).apply(
                lambda g: g.sample(
                    n=max(1, int(round(self.sample_size * len(g) / total))),
                    random_state=self.seed
                ),
                include_groups=True,
            ).reset_index(drop=True)
            print(f"分层采样后数据量: {len(df):,}")

        # ── 3. 特征 / 标签分离 ────────────────────────────────────────────────
        label_col = 'Attack Type'
        feature_cols = [c for c in df.columns if c != label_col]
        X = df[feature_cols].values.astype(float)
        print(f"特征维度: {X.shape[1]}")
        print(f"分类模式: {'二分类 (Normal/Attack)' if self.classification == 'binary' else '多分类 (7类)'}")

        # ── 4. 标签编码 ────────────────────────────────────────────────────────
        raw_labels = df[label_col].str.strip()

        if self.classification == 'binary':
            y = (raw_labels != 'Normal Traffic').astype(int).values
            self.label_encoder.classes_ = np.array(['normal', 'attack'])
        else:
            label_str = raw_labels.map(self._LABEL_MAP_MULTI).fillna('normal')
            y = label_str.map(self._INT_MAP_MULTI).fillna(0).values.astype(int)
            self.label_encoder.classes_ = np.array(
                ['normal', 'dos', 'ddos', 'portscan', 'bruteforce', 'webattacks', 'bots']
            )

        # ── 5. 训练 / 测试划分 (stratified) ────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y
        )
        print(f"划分完成: 训练+验证 {len(y_train):,}  测试 {len(y_test):,}")

        # ── 6. 归一化 ──────────────────────────────────────────────────────────
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        # ── 7. 打印类别分布 ─────────────────────────────────────────────────────
        class_names = self.label_encoder.classes_
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n训练集类别分布:")
        for u, c in zip(unique, counts):
            name = class_names[u] if u < len(class_names) else str(u)
            print(f"  {name:12s} ({u}): {c:7,d} ({c / len(y_train) * 100:.2f}%)")

        return X_train, y_train, X_test, y_test


def partition_iid(X, y, num_clients, seed=42):
    np.random.seed(seed)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    split_indices = np.array_split(indices, num_clients)
    return {i: (X[idx], y[idx]) for i, idx in enumerate(split_indices)}


def partition_non_iid(X, y, num_clients, alpha=0.5, seed=42):
    """
    Non-IID 数据划分：Dirichlet(alpha) + 保底分配
    每个客户端保底获得每个类别至少 min_per_class 个样本，
    剩余按 Dirichlet(alpha) 分配，确保 Non-IID 但不至于缺失类别。
    """
    np.random.seed(seed)
    classes = np.unique(y)
    n_classes = len(classes)
    client_indices = [[] for _ in range(num_clients)]

    # 保底：每个客户端每个类至少分到该类总量的5%（最少5个）
    min_per_class = 2  # keep small to preserve Non-IID heterogeneity

    for c in classes:
        c_indices = np.where(y == c)[0]
        np.random.shuffle(c_indices)

        # 先分保底
        guaranteed = min(min_per_class, len(c_indices) // num_clients)  # at most min_per_class per client, preserving Non-IID
        ptr = 0
        for client_id in range(num_clients):
            if ptr + guaranteed <= len(c_indices):
                client_indices[client_id].extend(c_indices[ptr:ptr+guaranteed].tolist())
                ptr += guaranteed

        # 剩余按 Dirichlet 分配
        remaining = c_indices[ptr:]
        if len(remaining) > 0:
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(remaining)).astype(int)[:-1]
            splits = np.split(remaining, split_points)
            for client_id, split in enumerate(splits):
                client_indices[client_id].extend(split.tolist())

    client_data = {}
    for i in range(num_clients):
        idx = np.array(client_indices[i], dtype=int)
        if len(idx) == 0:
            idx = np.random.choice(len(y), max(1, len(y) // (num_clients * 10)))
        np.random.shuffle(idx)
        client_data[i] = (X[idx], y[idx])

    return client_data


def apply_smote_per_client(client_data, num_classes, dataset_name='NSL-KDD',
                            classification='multi', k_neighbors=5):
    """

    参数:
        client_data : list of (X, y)，每个元素为一个客户端的数据
        num_classes : 类别总数
        dataset_name: 'NSL-KDD' | 'UNSW-NB15'
        classification: 'multi' 或 'binary'，两种模式均支持 SMOTE
        k_neighbors : SMOTE近邻数，不足时自适应降低

    返回:
        augmented_client_data: 过采样后的客户端数据列表
    """
    try:
        from imblearn.over_sampling import BorderlineSMOTE, SMOTE
    except ImportError:
        print("  [Borderline-SMOTE] 警告: 未安装 imbalanced-learn，跳过。"
              "请运行: pip install imbalanced-learn")
        return client_data

    if classification == 'binary':
        # 二分类：少数类(Attack=1)过采样到多数类(Normal=0)的80%
        multipliers = None  # 使用通用逻辑（自动检测少数类并过采样）
    elif dataset_name == 'CIC-IDS2017' and num_classes == 7:
        # CIC-IDS2017 多分类: WebAttacks(5) 和 Bots(6) 是极少数类
        multipliers = {5: 8, 6: 10}
    elif dataset_name == 'NSL-KDD' and num_classes == 5:
        multipliers = {3: 3, 4: 15}
    elif dataset_name == 'UNSW-NB15' and num_classes == 10:
        # Imbalance ratios: Analysis 1:35, Backdoor 1:40, Shellcode 1:62, Worms 1:535
        # Analysis/Backdoor: ×10, DoS/Recon: ×5
        # Shellcode/Worms: 降低极端倍率减少噪声风险
        multipliers = {1: 10, 2: 10, 3: 5, 7: 5, 8: 30, 9: 60}
    elif dataset_name == 'UNSW-NB15' and classification == 'two-stage-attack':
        # 两阶段 Stage2: 9类攻击子集 (去除 Normal 后, 标签 remap 1-9→0-8)
        # Generic(5)=35.7%, Exploits(3)=27%, Fuzzers(4)=14.7%, DoS(2)=9.9%, Recon(6)=8.5%
        # Analysis(0)=1.6%, Backdoor(1)=1.4%, Shellcode(7)=0.9%, Worms(8)=0.1%
        multipliers = {0: 5, 1: 5, 7: 8, 8: 15}
    else:
        multipliers = None

    augmented = []
    smote_log = []

    for client_id, (X, y) in enumerate(client_data):
        class_counts = np.bincount(y, minlength=num_classes)

        if multipliers is not None:
            sampling_strategy = {}
            for cls_id, mult in multipliers.items():
                cnt = class_counts[cls_id]
                if cnt < 2:          # 样本数不足，无法生成近邻，跳过
                    continue
                target = int(cnt * mult)
                # 不超过最大类数量的3倍，防止极端类别反转
                target = min(target, int(class_counts.max() * 3))
                if target > cnt:
                    sampling_strategy[cls_id] = target
        else:
            max_cnt = class_counts.max()
            threshold = int(max_cnt * 0.85)  # 0.5→0.85: 少数类过采样至多数类的85%
            sampling_strategy = {
                c: threshold
                for c in range(num_classes)
                if 2 <= class_counts[c] < threshold
            }

        if not sampling_strategy:
            augmented.append((X, y))
            continue

        # ── 自适应参数 ──────────────────────────────────────────────
        # k_neighbors: 用于合成样本时在同类中找近邻，不超过最小目标类样本数-1
        min_cnt = min(class_counts[c] for c in sampling_strategy)
        k = min(k_neighbors, int(min_cnt) - 1)
        if k < 1:
            augmented.append((X, y))
            continue
        # m_neighbors: 用于判断边界样本，在全局样本中找近邻，必须 >= k
        m = max(k, min(10, int(class_counts.sum()) - 1))

        try:
            smote = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k,
                m_neighbors=m,
                kind='borderline-1',
                random_state=42,
            )
            X_res, y_res = smote.fit_resample(X, y)
            method_used = 'Borderline-SMOTE'
        except Exception as e:
            # Non-IID 下 Borderline-SMOTE 可能失败（少数类全被判为 Noise）
            # 回退到普通 SMOTE，保证仍有过采样效果
            try:
                fallback = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k,
                    random_state=42,
                )
                X_res, y_res = fallback.fit_resample(X, y)
                method_used = 'SMOTE(回退)'
            except Exception as e2:
                augmented.append((X, y))
                smote_log.append(f"  客户端{client_id:2d}: Borderline-SMOTE和SMOTE均失败({e2})，保留原始数据")
                continue

        augmented.append((X_res, y_res))
        new_counts = np.bincount(y_res, minlength=num_classes)
        smote_log.append(
            f"  客户端{client_id:2d} [{method_used}]: "
            f"原{dict(enumerate(class_counts.tolist()))} "
            f"→ 后{dict(enumerate(new_counts.tolist()))}"
        )

    # 打印摘要（前5条）
    if smote_log:
        print(f"\n[Borderline-SMOTE] 各客户端过采样结果（共{len(smote_log)}个）：")
        for line in smote_log[:5]:
            print(line)
        if len(smote_log) > 5:
            print(f"  ... （省略其余 {len(smote_log)-5} 个客户端）")

    return augmented
