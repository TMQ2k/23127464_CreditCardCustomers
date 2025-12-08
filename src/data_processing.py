import numpy as np

def feature_typing(col_names, data):
    missing_tokens = {"", "Unknown", "unknown", "?", "NA"}
    numeric_cols = []
    categorical_cols = []
    for j, name in enumerate(col_names):
        col = data[:, j]
        col = np.char.strip(col)
        mask_missing = np.isin(col, list(missing_tokens))
        non_missing = col[~mask_missing]
        if non_missing.size == 0:
            categorical_cols.append(name)
            continue
        try:
            non_missing.astype(float)
            numeric_cols.append(name)
        except ValueError:
            categorical_cols.append(name)
    return numeric_cols, categorical_cols

def min_max_scale(arr):
    arr = arr.astype(float)
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val == 0: return arr
    return (arr - min_val) / (max_val - min_val)

def standard_scale(arr):
    arr = arr.astype(float)
    mean_val, std_val = np.mean(arr), np.std(arr)
    if std_val == 0: return arr
    return (arr - mean_val) / std_val

def log_transform(arr):
    arr = arr.astype(float)
    return np.log1p(arr)

def one_hot_encode_manual(column_data):
    unique_values = np.unique(column_data)
    n_categories = len(unique_values)
    n_samples = len(column_data)
    encoded = np.zeros((n_samples, n_categories))
    for i, val in enumerate(unique_values):
        mask = (column_data == val)
        encoded[mask, i] = 1
    return encoded, unique_values

def chi_square_test_manual(observed):
    observed = observed.astype(float)
    row_sums = np.sum(observed, axis=1)
    col_sums = np.sum(observed, axis=0)
    total = np.sum(observed)
    expected = np.outer(row_sums, col_sums) / total
    chi2 = np.sum((observed - expected)**2 / expected)
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    if df > 0:
        x = chi2 / df
        z = (x**(1/3) - (1 - 2/(9*df))) / np.sqrt(2/(9*df))
        p_value = 1 - 0.5 * (1 + np.tanh(z / np.sqrt(2)))
    else:
        p_value = 1.0
    return chi2, p_value, df, expected

def t_test_independent_manual(group1, group2):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
    df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    z_approx = abs(t_stat)
    p_value = 2 * (1 - 0.5 * (1 + np.tanh(z_approx / np.sqrt(2))))
    return t_stat, p_value, df
    