import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import joblib, json, os

X = pd.read_parquet("data/processed/features.parquet")  # build later
y = X.pop("bypass")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

clf = LogisticRegression(max_iter=1000, n_jobs=None)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:,1]
metrics = {"auc": roc_auc_score(y_test, proba), "logloss": log_loss(y_test, proba)}
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/logreg.joblib")
with open("models/logreg_metrics.json","w") as f: json.dump(metrics, f, indent=2)
print(metrics)
