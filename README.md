# 🔒 Online Fraud Detection - Financial Security ML

A **classification system** for detecting fraudulent online transactions using machine learning with anomaly detection, feature engineering, and imbalanced data handling techniques for financial security applications.

## 🎯 Overview

This project implements:
- ✅ Binary fraud detection classification
- ✅ Highly imbalanced dataset handling
- ✅ Anomaly detection techniques
- ✅ Transaction feature analysis
- ✅ Model evaluation for fraud (Precision, Recall)
- ✅ Real-time fraud scoring

## 🏗️ Architecture

### Fraud Detection Pipeline
- **Problem**: Binary classification (Fraud: Yes/No)
- **Data**: Online transaction records
- **Challenge**: Severe class imbalance (~99% legitimate, ~1% fraud)
- **Critical Metric**: Recall (catch all frauds, even at cost of false positives)
- **Algorithms**: Logistic Regression, Random Forest, Isolation Forest, XGBoost

### Tech Stack
| Component | Technology |
|-----------|-----------|
| **Core ML** | scikit-learn, XGBoost |
| **Imbalance** | SMOTE, Class Weights |
| **Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Language** | Python 3.8+ |

## 📊 Dataset Characteristics

### Transaction Features
```
Transaction Details:
├── Transaction_Amount: Dollar value
├── Transaction_Type: Purchase/Withdrawal/Transfer
├── Merchant_Category: Industry type
└── Transaction_Date: Timestamp

Customer Profile:
├── Customer_Age: Cardholder age
├── Account_Age_Days: Duration account exists
├── Num_Transactions_Last_30: Recent activity
└── Average_Transaction_Amount: Baseline spending

Risk Indicators:
├── Transaction_in_Foreign_Country: Flag (0/1)
├── Unusual_Time: Outside normal hours (0/1)
├── Unusual_Amount: Deviation from average
└── Device_Changed: New device (0/1)

Target:
└── Fraud: 0 (Legitimate) / 1 (Fraudulent)
```

### Class Distribution (Imbalanced)
```
Legitimate (Class 0): ~99.8% of data ✓ Normal pattern
Fraudulent (Class 1):  ~0.2% of data ✗ Rare anomalies

This imbalance creates challenges:
- Naive accuracy = 99.8% with all-zero predictions
- Traditional metrics misleading
- Need: Precision, Recall, ROC-AUC, F1-Score
```

## 🔧 Handling Imbalanced Data

### Strategy 1: Class Weighting

```python
from sklearn.ensemble import RandomForestClassifier

# Weight minority class higher
# fraud_weight = n_legit / n_fraud ≈ 500:1
model_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Auto-calculates weights
    random_state=42
)
model_weighted.fit(X_train, y_train)

# Alternative: Manual weights
class_weights = {
    0: 1,      # Legitimate
    1: 500     # Fraud (500x importance)
}
```

**How It Works**
```
Loss = Σ(weight[y_i] × loss(y_true, y_pred))
Better learning from rare fraud cases
```

### Strategy 2: SMOTE (Synthetic Minority Oversampling)

```python
from imblearn.over_sampling import SMOTE

# Synthetically generate minority samples
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original distribution: {(y_train==1).sum()} fraud cases")
print(f"SMOTE distribution: {(y_train_balanced==1).sum()} fraud cases")

# New ratio: 50% legitimate, 50% fraud (balanced for training)
model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_smote.fit(X_train_balanced, y_train_balanced)
```

**SMOTE Process**
```
For each minority sample:
1. Find k nearest neighbors (k=5)
2. Randomly select neighbor
3. Interpolate: new_sample = sample + random(0,1) × (neighbor - sample)
4. Result: Synthetic fraud patterns in feature space
```

### Strategy 3: Threshold Adjustment

```python
# Default threshold = 0.5
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_default = (y_pred_proba >= 0.5).astype(int)

# Lower threshold to catch more frauds
threshold = 0.3  # More sensitive
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Effect:
# Default (0.5): 99% precision, 40% recall (miss many frauds)
# Adjusted (0.3): 95% precision, 70% recall (catch more frauds)
```

## 🤖 Fraud Detection Models

### Model 1: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(
    class_weight='balanced',
    max_iter=500,
    random_state=42
)
model_lr.fit(X_train, y_train)

# Decision boundary: log(p/(1-p)) = β₀ + β₁×amount + β₂×foreign + ...
# Fraud probability: p = 1 / (1 + e^(-z))

from sklearn.metrics import precision_recall_curve, auc
precision, recall, thresholds = precision_recall_curve(y_test, 
    model_lr.predict_proba(X_test)[:, 1])
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")  # ~0.85-0.90
```

### Model 2: Random Forest with Balancing

```python
# Random Forest naturally handles imbalance better than LR
model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight='balanced',
    min_samples_leaf=10,  # Prevent overfitting to synthetic samples
    random_state=42
)
model_rf.fit(X_train_balanced, y_train_balanced)

# Feature importance
importances = model_rf.feature_importances_
for feature, importance in sorted(
    zip(X.columns, importances), key=lambda x: x[1], reverse=True
)[:5]:
    print(f"{feature}: {importance:.4f}")

# Top fraud indicators typically:
# 1. Transaction_in_Foreign_Country: High importance
# 2. Unusual_Amount (deviation from average): Medium-high
# 3. Device_Changed: Medium
# 4. Unusual_Time: Medium
```

### Model 3: Isolation Forest (Anomaly Detection)

```python
from sklearn.ensemble import IsolationForest

# Unsupervised anomaly detection
# Assumes fraud = rare anomalies
iso_forest = IsolationForest(
    contamination=0.005,  # Expect ~0.5% fraud
    random_state=42
)
anomaly_scores = iso_forest.fit_predict(X_train)
# -1 = anomaly (fraud), +1 = normal

# Can be standalone or combined with supervised method
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso_binary = (y_pred_iso == -1).astype(int)

from sklearn.metrics import f1_score
f1_iso = f1_score(y_test, y_pred_iso_binary)
print(f"Isolation Forest F1: {f1_iso:.4f}")
```

**Why Isolation Forest?**
```
- Doesn't assume normal distribution
- Efficient for high-dimensional data
- Can flag transactions "isolated" from normal patterns
- No training on fraud examples needed
```

### Model 4: XGBoost with Custom Evaluation

```python
import xgboost as xgb

# XGBoost with custom scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~500

model_xgb = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

# Use early stopping on hold-out validation set
eval_set = [(X_val, y_val)]
model_xgb.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=10,
    verbose=False
)

# Predictions with fraud probability
fraud_probs = model_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = (fraud_probs >= 0.3).astype(int)  # Lower threshold
```

## 📊 Evaluation Metrics (Critical for Imbalanced Data)

### Why Standard Accuracy Fails

```python
# Naive Classifier: Predict all transactions are legitimate
y_pred_naive = np.zeros(len(y_test))
accuracy_naive = (y_pred_naive == y_test).mean()
# Accuracy = 99.8% ✗ MISLEADING!

# Real issue: Fraud Rate = 0%
fraud_rate_actual = (y_test == 1).mean()
fraud_rate_pred = (y_pred_naive == 1).mean()
# Both 0% - completely useless model!
```

### Correct Metrics for Fraud

#### Precision & Recall

```python
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred_xgb, average=None
)

print(f"Precision (Legit): {precision[0]:.4f}")  # Of predicted legit, % actually legit
print(f"Precision (Fraud): {precision[1]:.4f}")  # Of predicted fraud, % actually fraud
print(f"Recall (Legit):    {recall[0]:.4f}")     # Of actual legit, % correctly caught
print(f"Recall (Fraud):    {recall[1]:.4f}")     # ← KEY METRIC!

# Example results:
# Precision[Fraud] = 0.85 → When we flag fraud, 85% are true fraud
# Recall[Fraud] = 0.70 → We catch 70% of actual fraud cases
```

**Why Recall Matters for Fraud**
```
Cost of False Negative: Customer loses $$$, files dispute
Cost of False Positive: Customer inconvenienced, transaction blocked

Fraud detection priority: Maximize Recall (catch as many frauds as possible)
Accept higher False Positives (block some legitimate transactions)
```

#### ROC-AUC Score

```python
from sklearn.metrics import roc_auc_score, roc_curve

roc_auc = roc_auc_score(y_test, fraud_probs)
fpr, tpr, thresholds = roc_curve(y_test, fraud_probs)

print(f"ROC-AUC: {roc_auc:.4f}")  # Expected: 0.92-0.97

# Interpretation:
# Model probability: 95% chance fraud is higher-scored than legit
# Perfect: AUC=1.0, Random: AUC=0.5

# Visualization
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc:.3f})')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

#### F1-Score

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred_xgb)
# F1 = 2 × (Precision × Recall) / (Precision + Recall)

# Example: Precision=0.85, Recall=0.70
# F1 = 2 × (0.85 × 0.70) / (0.85 + 0.70) = 0.767

# Best when Precision-Recall balanced
# Higher F1 = better model for imbalanced fraud detection
```

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_xgb)
#                 Predicted
#           Legit      Fraud
# Actual Legit [98,000    1,000]  TN=98K, FP=1K
#       Fraud [   600    2,400]   FN=600, TP=2.4K

# Calculations:
# Precision(Fraud) = TP/(TP+FP) = 2400/3400 = 0.706
# Recall(Fraud) = TP/(TP+FN) = 2400/3000 = 0.800
# False Positive Rate = FP/(FP+TN) = 1000/99000 = 0.010

# Cost Analysis:
# 1000 false positives = customer friction
# 600 missed frauds = financial losses
# Business decides: Can we accept 1% false fraud alerts?
```

## 🚀 Installation & Usage

### Setup
```bash
git clone https://github.com/Sunny-commit/online-fraud-detection.git
cd online-fraud-detection

python -m venv env
source env/bin/activate

pip install pandas numpy scikit-learn xgboost imbalanced-learn jupyter matplotlib

jupyter notebook "online fraud detection.ipynb"
```

## 💡 Interview Insights

### Q1: How do you handle the imbalanced fraud dataset?

```
Techniques (in order of preference):
1. Class Weighting: Scale loss by class frequency
   - Simple, no extra data
   - model = RF(class_weight='balanced')

2. SMOTE: Synthetic oversample minority
   - Creates realistic synthetic frauds
   - Doubles data size, longer training
   
3. Threshold Adjustment: Lower decision boundary
   - Catch more frauds at cost of false positives
   - No retraining needed

4. Cost-Sensitive Learning: Custom loss weights
   - Directly model business cost
   - FP_cost=1, FN_cost=100 (fraud expensive)

Combined Approach:
- Use SMOTE for training data balance
- Train with class_weight='balanced'
- Lower threshold to optimize recall
- Validate on real distribution
```

### Q2: Why not just use accuracy as evaluation metric?

```
99.8% accuracy with all-negative predictions is USELESS!

Fraud Detection Requirement:
- Must catch actual frauds (high Recall)
- Must minimize customer friction (high Precision not critical)
- ROC-AUC shows discrimination ability across thresholds
- F1-Score balances precision-recall tradeoff

Analogy:
Medical screening: Cancer detection
- Accuracy: 99% (9 healthy, 1 sick, all negative)
- Recall: 0% (missed the cancer patient!)
- Better model: Lower threshold, catch more positives

Fraud is similar: Cost of missing fraud >> Cost of false alert
```

### Q3: How would you deploy this to production?

```
Real-time Fraud Detection Pipeline:

1. Feature Extraction Layer
   - Extract transaction features (30-50 features)
   - Real-time from transaction database
   - Aggregation: Last 30 days activity

2. Model Serving
   - Pickle pre-trained XGBoost model
   - Load into API server (Flask/FastAPI)
   - Sub-100ms prediction latency required

3. Scoring & Decision
   - Get fraud probability: 0-1 score
   - Apply threshold logic:
     * Score > 0.7: Block (manual review)
     * 0.5-0.7: Challenge (2FA, OTP)
     * <0.5: Approve

4. Feedback Loop
   - Customer disputes → Ground truth updates
   - Monthly retrain on new fraud patterns
   - Adversarial adaptation (fraudsters evolve)

5. Monitoring
   - Track false positive rate (customer complaints)
   - Track fraud recall (fraud that slipped through)
   - Drift detection (spending patterns change)
   - Model decay (older model less accurate)
```

## 🎯 Real-World Applications

**Financial Institutions**
- Credit card fraud prevention
- Online banking security
- Wire transfer verification

**E-commerce Platforms**
- Order checkout fraud detection
- Refund abuse prevention
- High-value purchase verification

**Payment Processors**
- Real-time transaction scoring
- Chargethrough fraud prevention
- Account takeover detection

**Fintech**
- Loan application fraud
- Risk assessment scoring
- KYC verification

## 📚 Key Concepts

**Precision-Recall Tradeoff**
```
Increase Recall (catch more frauds):
- Lower threshold
- Accept more false positives
- Customers inconvenienced

Increase Precision (fewer false alerts):
- Raise threshold
- Miss frauds
- Revenue losses

Sweet spot: Business-specific cost analysis
```

**Why Ensemble Methods Excel**
```
- Random Forest: Captures non-linear fraud patterns
- XGBoost: Iteratively improves on hard cases (sophisticated frauds)
- Isolation Forest: Finds anomalies without labeled fraud data

Combined: Maximum fraud detection capability
```

## 🌟 Portfolio Strengths

✅ Real financial security problem
✅ Addresses imbalanced classification
✅ Uses appropriate evaluation metrics
✅ Multiple handling techniques
✅ Practical anomaly detection (Isolation Forest)
✅ Business-context decision making
✅ Production-ready architecture thinking

## 📄 License

MIT License - Educational Use

---

**Next Steps**:
1. Add cost-based optimization (weight frauds vs false positives)
2. Implement SHAP explainability (why was transaction flagged?)
3. Deploy as microservice with REST API
4. Add concept drift detection (fraud patterns evolve)
5. Build dashboard for monitoring & retraining
