import shap
import joblib
import pandas as pd

model = joblib.load("models/early_model.pkl")
df = pd.read_csv("data/fraud_data.csv")

sample = df.iloc[:1][[
    "amount", "merchant_category", "country", "hour", "payment_method"
]]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

print("SHAP values:", shap_values)

