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
    desc_score_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_proba_sorted = y_proba[desc_score_indices]
    thresholds = np.unique(y_proba_sorted)
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        y_pred = (y_proba_sorted >= threshold).astype(int)
        tp = np.sum((y_true_sorted == 1) & (y_pred == 1))
        fp = np.sum((y_true_sorted == 0) & (y_pred == 1))
        tn = np.sum((y_true_sorted == 0) & (y_pred == 0))
        fn = np.sum((y_true_sorted == 1) & (y_pred == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    fpr_list = [0] + fpr_list + [1]
    tpr_list = [0] + tpr_list + [1]
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

class LinearRegression:
    def __init__(self, method='normal_equation', learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = {'loss': []}
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.method == 'normal_equation':
            X_b = np.c_[np.ones((n_samples, 1)), X]
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
        elif self.method == 'gradient_descent':
            self.weights = np.zeros(n_features)
            self.bias = 0
            for iteration in range(self.n_iterations):
                y_pred = X @ self.weights + self.bias
                loss = mean_squared_error(y, y_pred)
                self.history['loss'].append(loss)
                dw = (1 / n_samples) * X.T @ (y_pred - y)
                db = (1 / n_samples) * np.sum(y_pred - y)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


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
# K-NEAREST NEIGHBORS (KNN)
# ============================================================================

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=1)
        elif self.metric == 'cosine':
            dot_product = np.sum(x1 * x2, axis=1)
            norm1 = np.linalg.norm(x1, axis=1)
            norm2 = np.linalg.norm(x2)
            return 1 - (dot_product / (norm1 * norm2 + 1e-10))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._compute_distance(self.X_train, x)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)
        return np.array(predictions)
    
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
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if isinstance(model, LogisticRegression):
            fold_model = LogisticRegression(learning_rate=model.learning_rate, n_iterations=model.n_iterations, regularization=model.regularization, lambda_reg=model.lambda_reg)
        elif isinstance(model, KNeighborsClassifier):
            fold_model = KNeighborsClassifier(n_neighbors=model.n_neighbors, metric=model.metric)
        elif isinstance(model, LinearRegression):
            fold_model = LinearRegression(method=model.method, learning_rate=model.learning_rate, n_iterations=model.n_iterations)
        else:
            fold_model = model
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)
        if scoring == 'accuracy':
            score = accuracy_score(y_test, y_pred)
        elif scoring == 'precision':
            score = precision_score(y_test, y_pred)
        elif scoring == 'recall':
            score = recall_score(y_test, y_pred)
        elif scoring == 'f1':
            score = f1_score(y_test, y_pred)
        elif scoring == 'roc_auc':
            if hasattr(fold_model, 'predict_proba'):
                y_proba = fold_model.predict_proba(X_test)
                score = roc_auc_score(y_test, y_proba)
            else:
                score = accuracy_score(y_test, y_pred)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")
        scores.append(score)
    return np.array(scores)

class GridSearchCV:
    def __init__(self, model_class, param_grid, cv=5, scoring='accuracy'):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_model_ = None
        self.results_ = []
    
    def fit(self, X, y):
        
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        param_combinations = list(product(*param_values))
        for param_combo in param_combinations:
            params = dict(zip(param_names, param_combo))
            model = self.model_class(**params)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            self.results_.append({'params': params, 'mean_score': mean_score, 'std_score': std_score, 'scores': scores})
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        self.best_model_ = self.model_class(**self.best_params_)
        self.best_model_.fit(X, y)
        return self
    
    def predict(self, X):
        return self.best_model_.predict(X)
    
    def score(self, X, y):
        return self.best_model_.score(X, y)


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
