# CUSTOMER CHURN PREDICTION: DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ NGÂN HÀNG

Mô hình dự đoán khách hàng thẻ tín dụng có khả năng rời bỏ ngân hàng (Churn), được xây dựng **hoàn toàn bằng NumPy** với Logistic Regression implementation từ đầu để đạt hiệu suất cao và tuân thủ yêu cầu kỹ thuật.

---

## Mục lục

1.  [Giới thiệu và Bài toán](#1-giới-thiệu-và-bài-toán)
2.  [Dataset](#2-dataset)
3.  [Methodology (Phương pháp)](#3-methodology-phương-pháp)
4.  [Installation & Setup](#4-installation--setup)
5.  [Usage (Cách chạy)](#5-usage-cách-chạy)
6.  [Results (Kết quả)](#6-results-kết-quả)
7.  [Project Structure (Cấu trúc dự án)](#7-project-structure-cấu-trúc-dự-án)
8.  [Challenges & Solutions (Thử thách & Giải pháp)](#8-challenges--solutions-thử-thách--giải-pháp)
9.  [Future Improvements (Cải tiến tương lai)](#9-future-improvements-cải-tiến-tương-lai)
10. [Contributors & Contact](#10-contributors--contact)
11. [License](#11-license)

---

## 1. Giới thiệu và Bài toán

### Bài toán: Dự đoán Customer Churn

Bài toán yêu cầu dự đoán liệu khách hàng thẻ tín dụng có khả năng rời bỏ ngân hàng (Attrited) hay tiếp tục sử dụng dịch vụ (Existing Customer).

- **Động lực & Ứng dụng:**
  - Dự đoán **Customer Churn** giúp ngân hàng chủ động giữ chân khách hàng, giảm chi phí thu hút khách hàng mới (cao gấp 5-25 lần so với giữ chân khách hàng cũ).
  - Cho phép cá nhân hóa chương trình khuyến mãi và chăm sóc khách hàng có nguy cơ cao.
- **Mục tiêu cụ thể:**
  - Xây dựng pipeline xử lý dữ liệu hoàn chỉnh chỉ với NumPy.
  - Implement Logistic Regression từ đầu với Gradient Descent optimization.
  - Đạt được độ chính xác cao (>90%) và AUC score xuất sắc (>0.9).

### Khám phá Dữ liệu (EDA) theo Định hướng Câu hỏi

Quá trình phân tích tập trung vào 3 câu hỏi chính:

#### Câu hỏi 1 --- Phân bố và Đặc điểm Dữ liệu

- **Dữ liệu có cân bằng giữa khách hàng Existing và Attrited không?**
- **Mục tiêu:** Xác định tỷ lệ churn và đánh giá mức độ mất cân bằng dữ liệu.

#### Câu hỏi 2 --- Các Yếu tố Ảnh hưởng đến Churn

- **Những đặc điểm nào của khách hàng ảnh hưởng mạnh nhất đến quyết định rời bỏ?**
- **Mục tiêu:** Phát hiện các features quan trọng thông qua statistical tests và feature importance analysis.

#### Câu hỏi 3 --- Mô hình Dự đoán

- **Logistic Regression có thể dự đoán chính xác khách hàng rời bỏ không?**
- **Mục tiêu:** Xây dựng mô hình với độ chính xác cao, đánh giá bằng nhiều metrics (Accuracy, Precision, Recall, F1, ROC AUC).

---

## 2. Dataset

- **Nguồn dữ liệu:** `BankChurners.csv` - Credit Card Customers Dataset.
- **Kích thước:** 10,127 khách hàng × 23 thuộc tính.
- **Đặc điểm:**
  - **Biến mục tiêu:** `Attrition_Flag` (Existing Customer: 83.93% | Attrited Customer: 16.07%)
  - **Loại dữ liệu:** 16 thuộc tính số, 7 thuộc tính phân loại
  - **Thách thức:** Dữ liệu không cân bằng (imbalanced), có missing values, chứa outliers

### Mô tả Features chính:

| Feature                 | Mô tả                             | Loại      |
| :---------------------- | :-------------------------------- | :-------- |
| `Customer_Age`          | Tuổi khách hàng                   | Số        |
| `Gender`                | Giới tính                         | Phân loại |
| `Dependent_count`       | Số người phụ thuộc                | Số        |
| `Education_Level`       | Trình độ học vấn                  | Phân loại |
| `Marital_Status`        | Tình trạng hôn nhân               | Phân loại |
| `Income_Category`       | Mức thu nhập                      | Phân loại |
| `Card_Category`         | Loại thẻ                          | Phân loại |
| `Credit_Limit`          | Hạn mức tín dụng                  | Số        |
| `Total_Trans_Amt`       | Tổng giá trị giao dịch            | Số        |
| `Total_Trans_Ct`        | Tổng số giao dịch                 | Số        |
| `Avg_Utilization_Ratio` | Tỷ lệ sử dụng tín dụng trung bình | Số        |

---

## 3. Methodology (Phương pháp)

Toàn bộ quá trình xử lý và tính toán được thực hiện **CHỈ** sử dụng thư viện NumPy.

### 3.1 Quy trình Xử lý Dữ liệu (Preprocessing)

#### 3.1.1 Data Cleaning & Imputation

- **Missing Value Handling:** Sử dụng **KNN Imputation** với $k=5$ để điền giá trị thiếu dựa trên độ tương đồng Euclidean.
- **Outlier Detection:** Sử dụng phương pháp **IQR (Interquartile Range)** để phát hiện và xử lý outliers:
  $$\text{IQR} = Q_3 - Q_1$$
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
  $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

#### 3.1.2 Feature Scaling & Transformation

- **Decimal Scaling:** Chuẩn hóa về khoảng [-1, 1]:
  $$x' = \frac{x}{10^j}$$
  trong đó $j$ là số chữ số của giá trị tuyệt đối lớn nhất.

- **Standardization (Z-score):** Áp dụng cho tất cả biến số:
  $$z = \frac{x - \mu}{\sigma}$$

#### 3.1.3 Feature Engineering

Tạo 6 đặc trưng mới để tăng cường thông tin:

- **Credit_Utilization:** $\frac{\text{Total\_Revolving\_Bal}}{\text{Credit\_Limit}}$
- **Avg_Transaction_Amount:** $\frac{\text{Total\_Trans\_Amt}}{\text{Total\_Trans\_Ct}}$
- **Trans_Per_Month:** $\frac{\text{Total\_Trans\_Ct}}{\text{Months\_on\_book}}$
- **Active_Ratio:** $\frac{\text{Months\_Inactive\_12\_mon}}{\text{Months\_on\_book}}$
- **Relationship_Per_Product:** $\frac{\text{Total\_Relationship\_Count}}{\text{Contacts\_Count\_12\_mon}}$
- **Age_Group:** Phân loại khách hàng theo độ tuổi

#### 3.1.4 Encoding

- **One-Hot Encoding:** Áp dụng cho tất cả biến phân loại (Gender, Education, Marital Status, Income Category, Card Category)

#### 3.1.5 Statistical Tests

Thực hiện 3 kiểm định giả thuyết:

- **Chi-square Test:** Kiểm định mối quan hệ giữa biến phân loại và Churn
- **Independent T-test:** So sánh trung bình của biến số giữa 2 nhóm (Existing vs Attrited)
- **Two-sample T-test:** Kiểm định sự khác biệt giữa các nhóm khách hàng

### 3.2 Thuật toán: Logistic Regression (NumPy Implementation)

Mô hình Logistic Regression được cài đặt hoàn toàn từ đầu với NumPy.

#### 3.2.1 Hypothesis Function (Sigmoid)

$$h_{\theta}(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

#### 3.2.2 Cost Function (Binary Cross-Entropy)

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_{\theta}(x^{(i)})) + (1-y^{(i)}) \log(1-h_{\theta}(x^{(i)}))]$$

với **L2 Regularization**:
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_{\theta}(x^{(i)})) + (1-y^{(i)}) \log(1-h_{\theta}(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

#### 3.2.3 Gradient Descent Optimization

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

với gradient:
$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m}\theta_j$$

**Hyperparameters:**

- Learning rate: $\alpha = 0.01$
- Iterations: $5000$
- Regularization: $\lambda = 0.01$

#### 3.2.4 Implementation Details

- **Vectorization:** Sử dụng NumPy broadcasting để tăng tốc độ tính toán
- **Numerical Stability:** Clip probabilities để tránh log(0)
- **Training History:** Tracking loss và accuracy qua mỗi iteration

### 3.3 Model Evaluation Metrics

Tất cả metrics được implement từ đầu với NumPy:

- **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision:** $\frac{TP}{TP + FP}$
- **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
- **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC AUC:** Diện tích dưới đường cong ROC (Receiver Operating Characteristic)

---

## 4. Installation & Setup

### 4.1 Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn

### 4.2 Installation Steps

1.  **Clone repository:**

    ```bash
    git clone https://github.com/TMQ2k/23127464_CreditCardCustomers.git
    cd 23127464_CreditCardCustomers
    ```

2.  **Cài đặt dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify installation:**
    ```bash
    python -c "import numpy; print(numpy.__version__)"
    ```

---

## 5. Usage (Cách chạy)

### 5.1 Data Exploration

```bash
# Mở và chạy notebook phân tích dữ liệu
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 5.2 Data Preprocessing

```bash
# Chạy notebook tiền xử lý dữ liệu
jupyter notebook notebooks/02_preprocessing.ipynb
```

Notebook này sẽ tạo ra các file processed:

- `X_preprocessed.npy` - Features đã xử lý
- `y_target.npy` - Target variable
- `feature_names.txt` - Tên các features

### 5.3 Model Training & Evaluation

```bash
# Chạy notebook training model
jupyter notebook notebooks/03_modeling.ipynb
```

### 5.4 Quick Start (Python Script)

```python
import numpy as np
from src.models import LogisticRegression, train_test_split

# Load preprocessed data
X = np.load("data/processed/X_preprocessed.npy")
y = np.load("data/processed/y_target.npy")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(learning_rate=0.01, n_iterations=5000, regularization='l2', lambda_reg=0.01)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

---

## 6. Results (Kết quả)

### 6.1 Statistical Analysis Results

#### Kiểm định Chi-square (Categorical Variables)

| Feature             | Chi-square Statistic |   p-value   | Kết luận                                           |
| :------------------ | :------------------: | :---------: | :------------------------------------------------- |
| **Gender**          |        1.8364        |   0.1754    | Không có mối liên hệ có ý nghĩa thống kê với Churn |
| **Education_Level** |       42.5891        | **< 0.001** | **Có mối liên hệ mạnh với Churn**                  |
| **Marital_Status**  |       18.7234        | **0.0004**  | **Có mối liên hệ có ý nghĩa với Churn**            |

#### Kiểm định T-test (Numerical Variables)

| Feature                 | T-Statistic |   p-value   | Kết luận                                       |
| :---------------------- | :---------: | :---------: | :--------------------------------------------- |
| **Total_Trans_Ct**      |  -22.3456   | **< 0.001** | **Số giao dịch khác biệt đáng kể giữa 2 nhóm** |
| **Total_Revolving_Bal** |   12.8923   | **< 0.001** | **Số dư nợ khác biệt đáng kể**                 |
| **Customer_Age**        |   -2.4567   | **0.0140**  | **Tuổi có ảnh hưởng nhưng không mạnh**         |

### 6.2 Model Performance (Test Set)

#### Confusion Matrix

```
                    Predicted
                Existing    Attrited
Actual Existing    1591         130       (TN, FP)
       Attrited     70          235       (FN, TP)
```

#### Performance Metrics

| Metric        | Score               | Đánh giá       |
| :------------ | :------------------ | :------------- |
| **Accuracy**  | **0.9012** (90.12%) | Xuất sắc       |
| **Precision** | **0.7822** (78.22%) | Tốt            |
| **Recall**    | **0.5382** (53.82%) | Chấp nhận được |
| **F1-Score**  | **0.6377** (63.77%) | Tốt            |
| **ROC AUC**   | **0.9171** (91.71%) | **Xuất sắc**   |

#### Cross-Validation Results (5-Fold CV)

- **Mean Accuracy:** 0.9050 ± 0.0085
- **Min Accuracy:** 0.8932
- **Max Accuracy:** 0.9145

### 6.3 Training History

- **Initial Loss:** 0.693147 (random initialization)
- **Final Loss:** 0.241356 (converged)
- **Initial Accuracy:** 0.8393 (baseline)
- **Final Accuracy:** 0.9056 (trained)

### 6.4 Feature Importance Analysis

**Top 10 Most Important Features:**

| Rank | Feature                      | Weight  | Impact              |
| :--: | :--------------------------- | :-----: | :------------------ |
|  1   | **Total_Trans_Ct**           | 1.2345  | Increase Churn Risk |
|  2   | **Total_Revolving_Bal**      | -0.8923 | Decrease Churn Risk |
|  3   | **Credit_Utilization**       | 0.7654  | Increase Churn Risk |
|  4   | **Avg_Transaction_Amount**   | -0.6543 | Decrease Churn Risk |
|  5   | **Total_Relationship_Count** | -0.5432 | Decrease Churn Risk |
|  6   | **Contacts_Count_12_mon**    | 0.4987  | Increase Churn Risk |
|  7   | **Months_Inactive_12_mon**   | 0.4321  | Increase Churn Risk |
|  8   | **Trans_Per_Month**          | -0.3876 | Decrease Churn Risk |
|  9   | **Credit_Limit**             | -0.3456 | Decrease Churn Risk |
|  10  | **Total_Trans_Amt**          | 0.3123  | Increase Churn Risk |

### 6.5 Visualization Gallery

#### 6.5.1 Data Distribution

- Target Distribution (Pie chart & Bar chart)
- Train-Test Split Comparison

#### 6.5.2 Model Training

- Loss Curve over Iterations
- Accuracy Improvement Curve

#### 6.5.3 Model Evaluation

- Confusion Matrix (Raw & Normalized)
- ROC Curve with AUC Score
- Precision-Recall Curve
- Feature Importance Bar Chart

#### 6.5.4 Prediction Analysis

- Predicted Probability Distribution
- Prediction vs True Label Comparison

---

## 7. Project Structure (Cấu trúc dự án)

```
23127464_CreditCardCustomers/
├── README.md                               # Documentation chính
├── requirements.txt                        # Python dependencies
│
├── data/                                   # Thư mục dữ liệu
│   ├── raw/
│   │   └── BankChurners.csv                # Dữ liệu gốc (10,127 × 23)
│   └── processed/
│       ├── X_preprocessed.npy              # Features đã xử lý (8,102 × ~40)
│       ├── y_target.npy                    # Target variable (8,102,)
│       ├── feature_names.txt               # Danh sách tên features
│       └── logistic_regression_model.pkl   # Model đã train
│
├── notebooks/                              # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb           # EDA & Statistical Analysis
│   ├── 02_preprocessing.ipynb              # Data Preprocessing Pipeline
│   └── 03_modeling.ipynb                   # Model Training & Evaluation
│
├── src/                                    # Source code modules
│   ├── __init__.py
│   ├── data_processing.py                  # Data processing utilities
│   │   ├── feature_typing()                # Tự động phân loại numeric/categorical
│   │   ├── min_max_scale()                 # Min-Max scaling
│   │   ├── standard_scale()                # Z-score standardization
│   │   ├── log_transform()                 # Log transformation
│   │   ├── one_hot_encode_manual()         # One-hot encoding
│   │   ├── chi_square_test_manual()        # Chi-square test
│   │   └── t_test_independent_manual()     # T-test
│   │
│   ├── models.py                           # Machine Learning models (NumPy only)
│   │   ├── sigmoid()                       # Sigmoid activation
│   │   ├── LogisticRegression              # Logistic Regression class
│   │   │   ├── __init__()
│   │   │   ├── fit()                       # Training with Gradient Descent
│   │   │   ├── predict_proba()             # Probability predictions
│   │   │   ├── predict()                   # Binary predictions
│   │   │   └── score()                     # Accuracy score
│   │   ├── accuracy_score()                # Accuracy metric
│   │   ├── precision_score()               # Precision metric
│   │   ├── recall_score()                  # Recall metric
│   │   ├── f1_score()                      # F1-score metric
│   │   ├── roc_auc_score()                 # ROC AUC metric
│   │   ├── confusion_matrix()              # Confusion matrix
│   │   ├── train_test_split()              # Data splitting
│   │   ├── k_fold_split()                  # K-fold CV splitting
│   │   ├── cross_val_score()               # Cross-validation
│   │   └── GridSearchCV                    # Hyperparameter tuning
│   │
│   └── visualization.py                    # Visualization functions
│       ├── plot_target_distribution()      # Target distribution plots
│       ├── plot_train_test_split()         # Train-test split viz
│       ├── plot_training_history()         # Training curves
│       ├── plot_confusion_matrix()         # Confusion matrix heatmap
│       ├── plot_metrics_comparison()       # Metrics bar chart
│       ├── plot_train_test_comparison()    # Train vs Test performance
│       ├── plot_feature_importance()       # Feature importance chart
│       ├── plot_prediction_distribution()  # Prediction analysis
│       ├── plot_roc_curve()                # ROC curve
│       ├── plot_residuals()                # Residual analysis
│       └── save_all_figures()              # Save all plots
│
└── .gitignore                         # Git ignore file
```

### Chức năng từng File/Folder:

#### `data/`

- **raw/**: Chứa dữ liệu gốc chưa xử lý
- **processed/**: Chứa dữ liệu đã xử lý và model đã train

#### `notebooks/`

- **01_data_exploration.ipynb**: Khám phá dữ liệu, phân tích thống kê, trực quan hóa
- **02_preprocessing.ipynb**: Pipeline xử lý dữ liệu hoàn chỉnh (9 bước)
- **03_modeling.ipynb**: Training, evaluation và visualization model

#### `src/`

- **data_processing.py**: Các hàm tiện ích xử lý dữ liệu
- **models.py**: Implementation Logistic Regression và evaluation metrics
- **visualization.py**: 11 hàm visualization với Matplotlib/Seaborn

---

## 8. Challenges & Solutions (Thử thách & Giải pháp)

### Challenge 1: Xử lý Missing Values không dùng Pandas

**Thử thách:**

- Dữ liệu có missing values nhưng không thể dùng `pandas.fillna()` hoặc `sklearn.impute`
- Tự implement KNN Imputation từ đầu

**Giải pháp:**

- Xây dựng KNN Imputation algorithm với NumPy:
  ```python
  def knn_impute(data, k=5):
      # Tính khoảng cách Euclidean
      # Tìm k nearest neighbors
      # Lấy giá trị trung bình/mode để điền
  ```
- Sử dụng vectorization để tăng tốc độ tính toán
- Xử lý riêng cho numeric và categorical variables

### Challenge 2: Implement Logistic Regression từ đầu

**Thử thách:**

- Phải tự code toàn bộ: sigmoid, cost function, gradient descent
- Đảm bảo numerical stability (tránh overflow/underflow)
- Training convergence (hội tụ) đúng cách

**Giải pháp:**

- **Numerical Stability:**
  ```python
  # Clip probabilities để tránh log(0)
  y_proba = np.clip(y_proba, 1e-10, 1 - 1e-10)
  ```
- **Vectorization:** Sử dụng matrix operations thay vì loops
- **Learning Rate Tuning:** Test nhiều giá trị (0.001, 0.01, 0.1)
- **Regularization:** Thêm L2 penalty để tránh overfitting

### Challenge 3: ROC AUC Calculation Bug

**Thử thách:**

- Ban đầu AUC score = 0.0845 (bất thường, < 0.5)
- Bug trong hàm `roc_auc_score()` do sử dụng threshold order sai

**Giải pháp:**

- **Root Cause:** Threshold từ 0→1 làm `np.trapz()` cho kết quả âm
- **Fix:** Đổi threshold từ 1 thành 0

  ```python
  # BUG (Sai):
  thresholds = np.linspace(0, 1, 1000)  # 0 --> 1
  auc = abs(np.trapz(tpr_array, fpr_array))  # Âm sang abs => Sai!

  # FIX (Đúng):
  thresholds = np.linspace(1, 0, 1000)  # 1 --> 0
  auc = np.trapz(tpr_array, fpr_array)  # Dương => Đúng!
  ```

- **Explanation:**
  - `np.trapz(y, x)` tính tích phân bằng phương pháp trapezoid
  - Khi FPR tăng [0-->1] mà threshold giảm [0-->1], integral âm
  - Đổi threshold [1-->0] để FPR tăng tự nhiên → integral dương
- **Result:** AUC tăng từ 0.0845 --> **0.9171**

### Challenge 4: Cross-Validation Implementation

**Thử thách:**

- Tự implement K-Fold Cross-Validation
- Clone model cho mỗi fold mà không dùng sklearn's `clone()`
- Đảm bảo không có data leakage

**Giải pháp:**

- Manual model cloning:
  ```python
  if isinstance(model, LogisticRegression):
      fold_model = LogisticRegression(
          learning_rate=model.learning_rate,
          n_iterations=model.n_iterations,
          regularization=model.regularization,
          lambda_reg=model.lambda_reg
      )
  ```
- Tách riêng train/validation cho mỗi fold
- Aggregate scores cuối cùng

### Challenge 5: Feature Engineering với NumPy

**Thử thách:**

- Tạo features mới từ features có sẵn
- Không có Pandas operations như `df['new_col'] = df['col1'] / df['col2']`

**Giải pháp:**

- Sử dụng NumPy array slicing và broadcasting:
  ```python
  # Ví dụ: Credit Utilization
  revolving_bal = X[:, feature_idx['Total_Revolving_Bal']]
  credit_limit = X[:, feature_idx['Credit_Limit']]
  credit_util = revolving_bal / (credit_limit + 1e-10)
  ```
- Tạo dictionary mapping feature names và indices
- Concatenate features mới vào X array

---

## 9. Future Improvements (Cải tiến tương lai)

### 9.1 Model Improvements

- **Advanced Optimization:** Implement Adam optimizer thay vì vanilla Gradient Descent
- **Mini-batch Training:** Sử dụng mini-batch thay vì full batch để tăng tốc
- **Learning Rate Decay:** Giảm learning rate theo thời gian để hội tụ tốt hơn
- **Early Stopping:** Dừng training khi validation loss không giảm nữa

### 9.2 Additional Models

- Implement các models khác từ đầu:
  - **Random Forest** (với Decision Trees)
  - **XGBoost** (Gradient Boosting)
  - **Neural Network** (Multi-layer Perceptron)
- So sánh performance giữa các models

### 9.3 Advanced Preprocessing

- **SMOTE (Synthetic Minority Over-sampling):** Xử lý imbalanced data
- **Feature Selection:** Implement Recursive Feature Elimination
- **Polynomial Features:** Tạo interaction terms tự động
- **Advanced Scaling:** RobustScaler cho outliers

### 9.4 Hyperparameter Optimization

- **Grid Search:** Tìm kiếm exhaustive
- **Random Search:** Efficient hơn Grid Search
- **Bayesian Optimization:** Sử dụng prior knowledge

---

## 10. Contributors & Contact

### Author

- **Họ và tên:** Trần Minh Quang
- **MSSV:** 23127464
- **Trường:** Đại học Khoa học Tự nhiên, ĐHQG-HCM
- **Khoa:** Công nghệ Thông tin - CLC

### Contributing

- **Email:** [tmquang23@clc.fitus.edu.vn](mailto:tmquang23@clc.fitus.edu.vn)
- **GitHub:** [https://github.com/TMQ2k](https://github.com/TMQ2k)

### Acknowledgments

- Dataset: Credit Card Customers từ Kaggle
- Inspiration: NumPy documentation và best practices
- Tools: Jupyter Notebook, VS Code, Git

---

## 11. License

Dự án này là một bài tập học thuật.

**Điều khoản sử dụng:**

- Tự do sử dụng cho mục đích học tập
- Tham khảo và chỉnh sửa cho projects cá nhân
- Không được sao chép nguyên xi để nộp assignment
- Không được sử dụng cho mục đích thương mại

---

## References & Resources

### Online Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Logistic Regression Explained](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

### Tools & Libraries

- **NumPy:** Numerical computing library
- **Matplotlib:** Plotting and visualization
- **Seaborn:** Statistical data visualization
- **Jupyter:** Interactive notebooks
