# Credit Card Customers - Data Processing Project

## 📋 Tổng quan

Dự án xử lý và phân tích dữ liệu khách hàng thẻ tín dụng (BankChurners.csv) **chỉ sử dụng NumPy** - không sử dụng Pandas, Scikit-learn hay bất kỳ thư viện xử lý dữ liệu nào khác.

## 🎯 Mục tiêu

Thực hiện đầy đủ data processing pipeline bao gồm:

- ✅ Data loading và validation
- ✅ Missing values handling
- ✅ Outlier detection và treatment
- ✅ Normalization (Min-Max, Log, Decimal)
- ✅ Standardization (Z-score)
- ✅ Feature Engineering
- ✅ Dimensionality Reduction (PCA từ scratch)
- ✅ Descriptive Statistics
- ✅ Hypothesis Testing

## 📁 Cấu trúc Thư mục

```
23127464_CreditCardCustomers/
├── data/
│   ├── raw/
│   │   └── BankChurners.csv          # Dữ liệu gốc (10,127 rows × 23 cols)
│   └── processed/
│       ├── numeric_data_processed.npy
│       ├── data_minmax_normalized.npy
│       ├── data_standardized.npy
│       ├── data_pca.npy
│       └── engineered_features.npy
├── notebooks/
│   └── 01_data_exploration.ipynb     # Main notebook (30 cells)
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   └── visualization.py
├── README.md                          # File này
├── RESULTS.md                         # Kết quả chi tiết
├── USAGE.md                           # Hướng dẫn sử dụng
└── requirements.txt
```

## 🚀 Quick Start

### 1. Clone repository

```bash
git clone <repository-url>
cd 23127464_CreditCardCustomers
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

**Lưu ý:** Chỉ cần NumPy!

### 3. Mở notebook

```bash
# Trong VS Code
code notebooks/01_data_exploration.ipynb
```

### 4. Chạy notebook

Chạy tuần tự các cells từ trên xuống dưới (Shift + Enter)

## 📊 Dataset

**BankChurners.csv**

- **Số dòng**: 10,127 customers
- **Số cột**: 23 columns
  - 17 numeric features
  - 6 categorical features

**Một số features quan trọng:**

- Customer_Age
- Gender
- Dependent_count
- Credit_Limit
- Total_Trans_Amt
- Total_Trans_Ct
- Attrition_Flag (target)

## 🔧 Kỹ thuật Đã Sử dụng

### 1. Data Loading

- CSV parsing thủ công chỉ với Python built-in functions
- Xử lý quotes và commas trong CSV format

### 2. Missing Values

- **Numeric**: Median imputation (robust to outliers)
- **Categorical**: Mode imputation
- Total Unknown values handled: 3,380

### 3. Outlier Detection

- **Method**: IQR (Interquartile Range)
- **Treatment**: Capping thay vì removal
- Detected outliers trong 12/16 features

### 4. Normalization

- **Min-Max Scaling**: [0, 1]
- **Log Transformation**: log(x + ε)
- **Decimal Scaling**: x / 10^d

### 5. Standardization

- **Z-score**: mean=0, std=1
- Phù hợp cho gradient-based algorithms

### 6. Feature Engineering

**6 features mới:**

1. Credit_Utilization
2. Avg_Transaction_Amount
3. Trans_Per_Month
4. Customer_Lifetime_Value_Proxy
5. Relationship_Intensity
6. Age_Credit_Ratio

### 7. PCA (từ scratch)

- Implementation: Eigenvalue decomposition
- Giảm từ 16 → 10 dimensions
- Giữ được 94.38% variance

### 8. Statistical Analysis

**Descriptive Statistics:**

- Mean, Median, Std, Variance
- Skewness, Kurtosis

**Hypothesis Testing:**

- One-sample t-test
- Chi-square test for variance
- Two-sample t-test

## 📈 Kết quả Chính

### Missing Values

- Education_Level: 1,519 → filled with "Graduate"
- Marital_Status: 749 → filled with "Married"
- Income_Category: 1,112 → filled with "Less than $40K"

### Outliers

- Total detected: 6,724 outliers across 12 features
- Treatment: Capped at Q1-1.5×IQR and Q3+1.5×IQR

### PCA Results

| PC   | Variance | Cumulative |
| ---- | -------- | ---------- |
| PC1  | 19.71%   | 19.71%     |
| PC2  | 16.16%   | 35.87%     |
| PC3  | 11.51%   | 47.38%     |
| PC10 | 3.92%    | 94.38%     |

### Hypothesis Tests

1. **Age mean ≠ 45**: Bác bỏ H0 (p < 0.05)
2. **Credit_Limit variance ≠ 50M**: Bác bỏ H0 (p < 0.05)
3. **Credit_Limit young vs old**: Chấp nhận H0 (không khác biệt)

## 📚 Documentation

- **RESULTS.md**: Kết quả chi tiết và phân tích
- **USAGE.md**: Hướng dẫn sử dụng từng bước
- **Notebook**: Có markdown cells giải thích từng phần

## 🛠️ Technologies

- **Python 3.x**
- **NumPy** (only library used for data processing)
- **Jupyter Notebook** (for interactive development)

**KHÔNG sử dụng:**

- ❌ Pandas
- ❌ Scikit-learn
- ❌ Scipy
- ❌ Các thư viện xử lý dữ liệu khác

## 💡 Key Features

1. **Pure NumPy Implementation**: Tất cả algorithms được implement từ scratch
2. **Numerical Stability**: Sử dụng float64, epsilon handling, catastrophic cancellation prevention
3. **Comprehensive**: Đầy đủ từ data loading đến hypothesis testing
4. **Reusable**: Code có thể áp dụng cho datasets khác
5. **Well-documented**: Comments và markdown cells đầy đủ

## 🧪 Testing

Tất cả 19 code cells đã được test và chạy thành công:

- Mean của standardized data: ~0.000000 ✓
- Std của standardized data: 1.000000 ✓
- PCA variance explained: 94.38% ✓
- All statistical tests converged ✓

## 📖 Học tập

Project này demonstrate:

- Data processing fundamentals
- Statistical methods implementation
- Numerical computing best practices
- PCA algorithm từ scratch
- Hypothesis testing procedures

## 🤝 Contributing

Project này là assignment nên không accept contributions. Tuy nhiên bạn có thể:

- Fork để học tập
- Sử dụng code như reference
- Adapt cho datasets riêng

## 📝 License

Educational project - tự do sử dụng cho mục đích học tập.

## 👨‍💻 Author

Student ID: 23127464

## 📧 Contact

Có câu hỏi? Xem documentation trong:

- Notebook cells (markdown)
- RESULTS.md
- USAGE.md

---

**⚠️ Lưu ý quan trọng:**

- Project này KHÔNG sử dụng Pandas, Scikit-learn
- Tất cả xử lý dữ liệu chỉ với NumPy
- Các algorithms được implement từ scratch
- Tuân thủ numerical stability principles
