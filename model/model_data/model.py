import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "breast-cancer-wisconsin-data_data.csv")

df = pd.read_csv(DATA_PATH)

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print(f"Missing values: {df.isnull().sum().sum()}")

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

best_model = None
best_score = 0
best_name = ""

print("\n" + "="*50)
print("Model Evaluation Results")
print("="*50)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='recall')
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  CV Recall: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Confusion Matrix: [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    
    if recall > best_score:
        best_score = recall
        best_model = model
        best_name = name

with open(os.path.join(BASE_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("\n" + "="*50)
print(f"Best Model: {best_name} (Recall: {best_score:.4f})")
print("Saved: model.pkl and scaler.pkl")
print("="*50)