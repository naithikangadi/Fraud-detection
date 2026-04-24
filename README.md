# Fraud-detection

## 1. Exploration and Data Cleaning

To prepare the dataset for modeling, several preprocessing steps were implemented to reduce noise and enhance predictive signals:

### Noise Reduction
High-cardinality and non-predictive identifiers (e.g., `trans_num`, `unix_time`, `zip`, and names) were removed to prevent overfitting.

### Feature Engineering

- **Geospatial Analysis**  
  Used the Haversine formula to calculate `distance_km` between the user and the merchant.

- **Temporal Features**  
  Extracted `trans_hour` and `trans_dayofweek` to capture fraudulent patterns related to specific times or days.

- **Demographics**  
  Converted date-of-birth into `age` and calculated `txn_count_per_card` to track transaction frequency.

### Encoding Strategy

- **Label Encoding**  
  Applied to categorical fields like `category`, `gender`, and `job`.

- **Target Encoding**  
  The `merchant` column was target-encoded (mean fraud rate per merchant) to capture vendor-specific risk without bloating the feature space.

---

## 2. Modeling Approach

We implemented two distinct supervised learning models to address the severe class imbalance (where fraud is a tiny fraction of the total data).

### Model A: Logistic Regression

- **Justification**  
  Chosen as a baseline linear model. It is highly interpretable and efficient for high-dimensional data.

- **Approach**  
  - Used `class_weight='balanced'` to penalize misclassification of fraud cases  
  - Standardized features using `StandardScaler`

### Model B: LightGBM (Gradient Boosting)

- **Justification**  
  Selected for its ability to capture complex, non-linear relationships and efficiency with large datasets.

- **Approach**  
  - Used `scale_pos_weight=174` to handle imbalance  
  - Implemented early stopping to prevent overfitting while maximizing ROC-AUC  

---

## 3. Comparative Performance Analysis

Given the high class imbalance, we prioritized **Recall** and **ROC-AUC** over simple accuracy.

| Metric                     | Logistic Regression | LightGBM |
|--------------------------|--------------------|----------|
| Accuracy                 | 93%                | 100%     |
| ROC-AUC                  | 0.8210             | 0.9809   |
| Fraud Recall (Class 1)   | 0.70               | 0.84     |
| Fraud Precision (Class 1)| 0.05               | 0.74     |
| F1-Score (Class 1)       | 0.10               | 0.78     |

### Analysis of Success and Failure

#### Logistic Regression
- **Success**  
  Achieved a recall of 0.70, meaning it detected 70% of fraud cases.

- **Failure**  
  Extremely low precision (0.05), leading to a high number of false positives.  
  Its linear nature failed to capture complex fraud patterns.

#### LightGBM
- **Success**  
  - Best performance across all metrics  
  - ROC-AUC of 0.98 indicates near-perfect class separation  
  - Strong balance between recall (0.84) and precision (0.74)

- **Failure**  
  Functions as a "black box" model, requiring feature importance analysis for interpretability.

---

## 4. Conclusion

The **LightGBM model** is the clear winner for this application.

- Precision improved dramatically from **5% → 74%**
- Recall improved from **70% → 84%**

This confirms that fraud detection in this dataset relies on **non-linear patterns**, which are better captured by tree-based ensemble models.
