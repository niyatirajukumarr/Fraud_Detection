import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

data = pd.DataFrame({
    "amount": np.random.exponential(scale=200, size=n),
    "merchant_category": np.random.randint(0, 10, size=n),
    "country": np.random.randint(0, 5, size=n),
    "hour": np.random.randint(0, 24, size=n),
    "payment_method": np.random.randint(0, 4, size=n),

    # delayed features
    "device_risk": np.random.rand(n),
    "velocity": np.random.poisson(2, size=n),
    "past_chargebacks": np.random.binomial(1, 0.1, size=n)
})

# Fraud logic (hidden from model)
data["is_fraud"] = (
    (data["amount"] > 500).astype(int) |
    (data["device_risk"] > 0.8).astype(int) |
    (data["past_chargebacks"] == 1)
).astype(int)

# Simulate missing delayed data
mask = np.random.rand(n) < 0.4
data.loc[mask, ["device_risk", "velocity", "past_chargebacks"]] = np.nan

data.to_csv("data/fraud_data.csv", index=False)
print("Dataset generated")
