Hybrid Bank Transaction Fraud Risk Detection

1. Overview
This project builds a hybrid fraud risk scoring system for bank transactions.
It combines domain rules, engineered features, and tree‑based machine learning models to classify each transaction into High, Moderate, or Low fraud risk, with a strong focus on explainability using SHAP.

2. Dataset and Feature Engineering
Source: Bank transaction data with customer, transaction, device, and behavioral attributes.

Key steps:

Cleaning missing/invalid values, fixing date formats, and removing obvious data leaks.

Deriving time features: TransactionHour, TransactionDayOfWeek, TimeSinceLastTransaction, TransactionDuration, LongDuration flag.

Behavioral features: LoginAttempts, HighLoginAttempts, recent activity gaps.

Customer & account features: CustomerAge, CustomerOccupation, AccountBalance, IsCredit.

Rule engine output: RuleBasedFraud flag capturing strong, interpretable fraud heuristics.

Categorical encoding:

Label‑encoding: TransactionType, Channel, CustomerOccupation, Location.

Target:

FraudRiskLevel ∈ {High, Moderate, Low}, label‑encoded for modeling.

Train/test splits are pre‑saved as fraud_train_data.csv and fraud_test_data.csv to keep evaluation reproducible.

3. Modeling Pipeline
All models use the same feature set (IDs, dates, rule labels, and cluster labels dropped).

3.1 Decision Tree (baseline)
Model: DecisionTreeClassifier(max_depth=5, random_state=42)

Performance:

Accuracy ≈ 0.92

High: strong precision, recall ≈ 0.82 (many High → Low misses)

Low & Moderate: near‑perfect.

Insight: Good interpretable baseline, but under‑detects High‑risk transactions.

3.2 Random Forest (baseline ensemble)
Model: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

Performance:

Accuracy ≈ 0.96

High & Low: F1 ≈ 0.95

Moderate: perfect.

Confusion matrix: almost diagonal; High → Low errors drop to 30.

Insight: Ensemble trees significantly improve robustness over a single tree.

3.3 Tuned Random Forest
Tuning: GridSearchCV (5‑fold, f1_weighted) over:

n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.

Performance (best model):

Accuracy ≈ 0.98

High: precision 1.00, recall ≈ 0.95

Low: precision ≈ 0.95, recall 1.00

Moderate: perfect.

Confusion matrix: only 15 High cases misclassified as Low; no confusion for Low/Moderate.

Insight: Strong, production‑ready tree ensemble with excellent recall on High risk.

3.4 XGBoost (baseline)
Model: XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss').

Performance:

Accuracy ≈ 0.92 (similar to Decision Tree).

High‑risk recall ≈ 0.82 (same pattern: High → Low misses).

3.5 Tuned XGBoost (final model)
Tuning: RandomizedSearchCV (3‑fold, f1_weighted) over:

n_estimators, max_depth, learning_rate, subsample, colsample_bytree.

Best hyperparameters:

n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=1.

Performance:

Accuracy ≈ 0.99

High: precision 1.00, recall 0.98, F1 0.99

Low: precision 0.98, recall 1.00, F1 0.99

Moderate: precision/recall/F1 = 1.00.

Confusion matrix:

316/322 High correctly predicted as High.

Only 1 Low predicted as High.

No errors for Moderate.

Insight: Best trade‑off between accuracy and error profile; chosen as champion model.

4. Model Explainability (Feature Importance & SHAP)
4.1 Global importance
Across tuned Random Forest and tuned XGBoost, both feature_importances_ and SHAP agree on the key drivers:

RuleBasedFraud (dominant driver across all classes).

CustomerAge.

AccountBalance and TransactionAmount.

IsCredit and TransactionType.

CustomerOccupation, Channel.

Behavior & timing: LoginAttempts, HighLoginAttempts, TransactionDuration, TimeSinceLastTransaction.

These results confirm that the models rely on transparent, business‑aligned signals rather than obscure artefacts.

4.2 Multiclass SHAP analysis
Multiclass SHAP bar charts show per‑class contributions:

Rule‑based flag and transaction amount/balance have strong impact on High vs Low separation.

Age, credit/debit, and transaction type shape Moderate vs Low/High decisions.

Combined SHAP bar plots (mean |SHAP| across classes) provide a single global ranking for quick communication.

5. Key Outcomes
Built a hybrid fraud risk detector combining:

Domain rules and clustering (RuleBasedFraud, engineered risk features).

Supervised tree ensembles (Random Forest, XGBoost).

SHAP‑based explainability for governance.

Achieved ≈ 99% test accuracy with the tuned XGBoost model and extremely low miss‑rate on High‑risk transactions.

Delivered interpretable insights on why transactions are scored as High / Moderate / Low, ready for fraud‑ops review and documentation.

