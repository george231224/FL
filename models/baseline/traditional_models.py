import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class LIBSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', max_samples=30000):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        self.max_samples = max_samples

    def train(self, X_train, y_train):
        # RBF SVM 时间复杂度 O(n²~n³)，大数据集需采样
        if len(y_train) > self.max_samples:
            from sklearn.model_selection import train_test_split
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=self.max_samples,
                random_state=42, stratify=y_train
            )
            print(f"  LIBSVM 采样: {len(y_train)} -> {len(y_sub)} (分层)")
            self.model.fit(X_sub, y_sub)
        else:
            self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        tn = cm[0, 0] if cm.shape[0] > 1 else 0
        fp = cm[0, 1:].sum() if cm.shape[0] > 1 else 0
        far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0

        return {
            'Accuracy': accuracy_score(y_test, y_pred) * 100,
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
            'FAR': far
        }
