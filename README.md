# 🔍 Online Fraud Detection - Anomaly Detection

A **machine learning system for detecting fraudulent transactions** using anomaly detection, ensemble methods, and real-time pattern recognition.

## 🎯 Overview

This project provides:
- ✅ Fraud detection classification
- ✅ Anomaly detection algorithms
- ✅ Feature engineering for transactions
- ✅ Imbalanced class handling
- ✅ Real-time detection pipeline
- ✅ ROC-AUC evaluation

## 📊 Fraud Dataset

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FraudDataAnalysis:
    """Analyze fraud patterns"""
    
    def __init__(self, filepath='fraud_data.csv'):
        self.df = pd.read_csv(filepath)
    
    def explore_fraud(self):
        """Dataset overview"""
        print(f"Total transactions: {len(self.df)}")
        print(f"\nFraud distribution:")
        print(self.df['isFraud'].value_counts())
        
        fraud_rate = self.df['isFraud'].sum() / len(self.df) * 100
        print(f"Fraud rate: {fraud_rate:.2f}%")
    
    def fraud_characteristics(self):
        """Analyze fraud patterns"""
        fraud_df = self.df[self.df['isFraud'] == 1]
        
        print(f"Average fraud amount: ${fraud_df['amount'].mean():.2f}")
        print(f"Maximum fraud amount: ${fraud_df['amount'].max():.2f}")
        
        # Time patterns
        fraud_df['hour'] = pd.to_datetime(fraud_df['timestamp']).dt.hour
        fraud_by_hour = fraud_df.groupby('hour').size()
        print(f"Fraud by hour:\n{fraud_by_hour}")
```

## 🔧 Feature Engineering

```python
class FraudFeatureEngineer:
    """Engineer fraud detection features"""
    
    @staticmethod
    def transaction_features(df):
        """Transaction-level features"""
        df_copy = df.copy()
        
        # Amount statistics
        df_copy['Amount_Scaled'] = np.log1p(df_copy['amount'])
        
        # Time features
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        df_copy['hour'] = df_copy['timestamp'].dt.hour
        df_copy['dayOfWeek'] = df_copy['timestamp'].dt.dayofweek
        df_copy['month'] = df_copy['timestamp'].dt.month
        
        # Is weekend
        df_copy['is_weekend'] = (df_copy['dayOfWeek'] >= 5).astype(int)
        
        # Is night time (midnight to 6am)
        df_copy['is_night'] = ((df_copy['hour'] >= 0) & (df_copy['hour'] < 6)).astype(int)
        
        return df_copy
    
    @staticmethod
    def user_behavior_features(df):
        """User historical behavior"""
        df_copy = df.copy()
        
        # User transaction count
        df_copy['user_transaction_count'] = df_copy.groupby('user_id')['transaction_id'].transform('count')
        
        # User average amount
        df_copy['user_avg_amount'] = df_copy.groupby('user_id')['amount'].transform('mean')
        
        # User std amount
        df_copy['user_std_amount'] = df_copy.groupby('user_id')['amount'].transform('std')
        
        # Amount deviation from user average
        df_copy['amount_deviation'] = np.abs(
            (df_copy['amount'] - df_copy['user_avg_amount']) / (df_copy['user_std_amount'] + 1)
        )
        
        # Transactions per day
        df_copy['transactions_per_day'] = df_copy.groupby(
            [df_copy['user_id'], df_copy['timestamp'].dt.date]
        )['transaction_id'].transform('count')
        
        return df_copy
    
    @staticmethod
    def merchant_features(df):
        """Merchant behavior features"""
        df_copy = df.copy()
        
        # Merchant fraud rate
        merchant_fraud = df_copy.groupby('merchant_id')['isFraud'].agg(['sum', 'count'])
        merchant_fraud['fraud_rate'] = merchant_fraud['sum'] / merchant_fraud['count']
        df_copy['merchant_fraud_rate'] = df_copy['merchant_id'].map(
            merchant_fraud['fraud_rate']
        )
        
        # Merchant typical amount
        df_copy['merchant_avg_amount'] = df_copy.groupby('merchant_id')['amount'].transform('mean')
        
        return df_copy
```

## 🤖 Anomaly Detection Algorithms

```python
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.covariance import EllipticEnvelope

class AnomalyDetectors:
    """Multiple anomaly detection algorithms"""
    
    @staticmethod
    def isolation_forest(X_train, contamination=0.05):
        """Isolation Forest for anomaly detection"""
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = iso_forest.fit_predict(X_train)
        anomaly_scores = iso_forest.score_samples(X_train)
        
        return iso_forest, anomaly_labels, anomaly_scores
    
    @staticmethod
    def elliptic_envelope(X_train, contamination=0.05):
        """Gaussian covariance estimation"""
        elliptic = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        
        anomaly_labels = elliptic.fit_predict(X_train)
        anomaly_scores = elliptic.decision_function(X_train)
        
        return elliptic, anomaly_labels, anomaly_scores
    
    @staticmethod
    def random_forest_anomaly(X_train, y_train):
        """Random Forest for fraud classification"""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        fraud_prob = rf.predict_proba(X_train)[:, 1]
        
        return rf, fraud_prob
```

## 💳 Real-Time Detection Pipeline

```python
from sklearn.preprocessing import StandardScaler
import joblib

class FraudDetectionPipeline:
    """Real-time fraud detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.threshold = 0.5
    
    def train_ensemble(self, X_train, y_train):
        """Train multiple models"""
        # Isolation Forest
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.iso_forest.fit(X_train)
        
        # Random Forest
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.rf.fit(X_train, y_train)
        
        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        X_scaled = self.scaler.fit_transform(X_train)
        self.lr = LogisticRegression(random_state=42, max_iter=1000)
        self.lr.fit(X_scaled, y_train)
    
    def predict_fraud(self, transaction):
        """Predict fraud for single transaction"""
        # Get predictions from each model
        iso_score = -self.iso_forest.score_samples(transaction.reshape(1, -1))[0]
        iso_score = 1 / (1 + np.exp(-iso_score))  # Convert to probability
        
        rf_prob = self.rf.predict_proba(transaction.reshape(1, -1))[0, 1]
        
        transaction_scaled = self.scaler.transform(transaction.reshape(1, -1))
        lr_prob = self.lr.predict_proba(transaction_scaled)[0, 1]
        
        # Ensemble voting
        ensemble_prob = (iso_score + rf_prob + lr_prob) / 3
        
        # Decision
        is_fraud = ensemble_prob > self.threshold
        
        return {
            'fraud_probability': ensemble_prob,
            'is_fraud': is_fraud,
            'model_scores': {
                'isolation_forest': iso_score,
                'random_forest': rf_prob,
                'logistic_regression': lr_prob
            }
        }
    
    def flag_suspicious_patterns(self, user_history):
        """Flag suspicious user behavior"""
        flags = []
        
        if user_history['transactions_per_day'] > 10:
            flags.append('Unusual transaction frequency')
        
        if user_history['amount_deviation'] > 3:
            flags.append('Amount significantly different from user average')
        
        if user_history['is_night'] and user_history['merchant_fraud_rate'] > 0.1:
            flags.append('High-risk merchant at unusual hour')
        
        return flags
```

## 📊 Evaluation Metrics

```python
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, f1_score
)

class FraudEvaluator:
    """Evaluate fraud detection"""
    
    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba):
        """Comprehensive evaluation"""
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"\nROC-AUC: {roc_auc:.4f}")
        
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        print(f"Max F1 Score: {max([f1_score(y_true, (y_pred_proba > thresholds)) for thresholds in np.arange(0, 1, 0.01)]):.4f}")
```

## 💡 Interview Talking Points

**Q: Handle class imbalance?**
```
Answer:
- SMOTE oversampling
- Class weights in models
- Adjust decision threshold
- F1/precision-recall metrics
- Cost-sensitive learning
```

**Q: Real-time performance?**
```
Answer:
- Fast inference (feature scaling)
- Model caching
- Batch processing for efficiency
- Threshold tuning (sensitivity vs specificity)
```

## 🌟 Portfolio Value

✅ Anomaly detection
✅ Ensemble methods
✅ Class imbalance handling
✅ Real-time detection
✅ Feature engineering
✅ Fraud domain expertise
✅ Production pipeline

---

**Technologies**: Scikit-learn, Pandas, NumPy

