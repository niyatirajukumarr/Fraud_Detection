import shap
import joblib
import pandas as pd

model = joblib.load("models/delayed_model.pkl")
df = pd.read_csv("data/fraud_data.csv")

features = [
    "amount", "merchant_category", "country", "hour", "payment_method",
    "device_risk", "velocity", "past_chargebacks"
]

sample = df.iloc[:1][features]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

print("Top contributing features:")
for f, v in zip(features, shap_values[1][0]):
    print(f, round(v, 3))

