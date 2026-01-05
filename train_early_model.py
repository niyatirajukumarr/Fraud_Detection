import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv("data/fraud_data.csv")

early_features = [
    "amount",
    "merchant_category",
    "country",
    "hour",
    "payment_method"
]

X = df[early_features]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6
)

model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, preds))

joblib.dump(model, "models/early_model.pkl")
print("Early model saved")
