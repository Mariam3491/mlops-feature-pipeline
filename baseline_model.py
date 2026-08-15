# baseline_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('dataset/cleaned_creditcard.csv')

# Separate features and target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    results[name] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save best model
    if name == 'XGBoost':  # Assuming XGBoost performs best
        joblib.dump(model, 'models/baseline_model.pkl')
        print(f"✅ Baseline model saved to models/baseline_model.pkl")

# Feature importance for best model
best_model = models['XGBoost']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Create visualization
plt.figure(figsize=(12, 6))
plt.barh(feature_importance.head(10)['feature'], 
         feature_importance.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances - XGBoost')
plt.tight_layout()
plt.savefig('docs/feature_importance.png')