import numpy as np
from itertools import product

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def mse_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred) / len(y_true)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def precision_score(y_true, y_pred, zero_division=0):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return zero_division
    return tp / (tp + fp)


def recall_score(y_true, y_pred, zero_division=0):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return zero_division
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)


def roc_auc_score(y_true, y_proba):
    n_thresholds = 1000
    thresholds = np.linspace(1, 0, n_thresholds)
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)
    auc = np.trapz(tpr_array, fpr_array)
    return auc


def classification_report(y_true, y_pred, target_names=None):
    if target_names is None:
        target_names = ['Class 0', 'Class 1']
    report = {}
    for i, name in enumerate(target_names):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        prec = precision_score(y_true_binary, y_pred_binary)
        rec = recall_score(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)
        support = np.sum(y_true == i)
        report[name] = {
            'precision': prec,
            'recall': rec,
            'f1-score': f1,
            'support': support
        }
    report['accuracy'] = accuracy_score(y_true, y_pred)
    report['macro avg'] = {
        'precision': np.mean([report[name]['precision'] for name in target_names]),
        'recall': np.mean([report[name]['recall'] for name in target_names]),
        'f1-score': np.mean([report[name]['f1-score'] for name in target_names]),
        'support': len(y_true)
    }
    return report


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.history = {'loss': [], 'accuracy': []}
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for iteration in range(self.n_iterations):
            linear_output = X @ self.weights + self.bias
            y_pred = sigmoid(linear_output)
            loss = binary_cross_entropy(y, y_pred)
            if self.regularization == 'l2':
                loss += self.lambda_reg * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += self.lambda_reg * np.sum(np.abs(self.weights))
            self.history['loss'].append(loss)
            y_pred_class = (y_pred >= 0.5).astype(int)
            acc = accuracy_score(y, y_pred_class)
            self.history['accuracy'].append(acc)
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            if self.regularization == 'l2':
                dw += (2 * self.lambda_reg / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_reg / n_samples) * np.sign(self.weights)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self
    
    def predict_proba(self, X):
        linear_output = X @ self.weights + self.bias
        return sigmoid(linear_output)
    
    def predict(self, X, threshold=0.5):
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def k_fold_split(n_samples, n_folds=5, shuffle=True, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    fold_sizes = np.full(n_folds, n_samples // n_folds, dtype=int)
    fold_sizes[:n_samples % n_folds] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, test_indices))
        current = stop
    return folds


def cross_val_score(model, X, y, cv=5, scoring='accuracy'):
    folds = k_fold_split(len(X), n_folds=cv, shuffle=True, random_state=42)
    scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        if isinstance(model, LogisticRegression):
            fold_model = LogisticRegression(learning_rate=model.learning_rate, n_iterations=model.n_iterations, regularization=model.regularization, lambda_reg=model.lambda_reg)
        else:
            raise ValueError("Unknown model type")
        fold_model.fit(X_train_fold, y_train_fold)
        if scoring == 'roc_auc':
            if hasattr(fold_model, 'predict_proba'):
                y_proba = fold_model.predict_proba(X_test_fold)
                score = roc_auc_score(y_test_fold, y_proba)
            else:
                y_pred = fold_model.predict(X_test_fold)
                score = accuracy_score(y_test_fold, y_pred)
        else:
            y_pred = fold_model.predict(X_test_fold)
            if scoring == 'accuracy':
                score = accuracy_score(y_test_fold, y_pred)
            elif scoring == 'precision':
                score = precision_score(y_test_fold, y_pred)
            elif scoring == 'recall':
                score = recall_score(y_test_fold, y_pred)
            elif scoring == 'f1':
                score = f1_score(y_test_fold, y_pred)
            else:
                raise ValueError(f"Unknown scoring: {scoring}")
        scores.append(score)
    return np.array(scores)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_features(X, method='standardization'):
    if method == 'standardization':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / (std + 1e-10)
        return X_normalized, mean, std
    elif method == 'min-max':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        X_normalized = (X - min_val) / (max_val - min_val + 1e-10)
        return X_normalized, min_val, max_val
    else:
        raise ValueError(f"Unknown method: {method}")

def print_classification_report(report):
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT")
    print("="*80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    for key, metrics in report.items():
        if key not in ['accuracy', 'macro avg']:
            print(f"{key:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                  f"{metrics['f1-score']:<12.4f} {metrics['support']:<10}")
    
    print("-"*80)
    metrics = report['macro avg']
    print(f"{'Macro avg':<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
          f"{metrics['f1-score']:<12.4f} {metrics['support']:<10}")
    
    print("-"*80)
    print(f"{'Accuracy':<20} {report['accuracy']:<12.4f}")
    print("="*80)
