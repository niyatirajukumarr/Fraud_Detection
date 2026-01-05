import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/fraud_data.csv")

full_features = [
    "amount", "merchant_category", "country", "hour", "payment_method",
    "device_risk", "velocity", "past_chargebacks"
]

X = df[full_features]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7
)

model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
print("Delayed model AUC:", roc_auc_score(y_test, preds))

joblib.dump(model, "models/delayed_model.pkl")
print("Delayed model saved")
