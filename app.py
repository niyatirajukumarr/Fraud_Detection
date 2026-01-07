import gradio as gr
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
from models.risk_update import update_risk

early_model_path = hf_hub_download(
    repo_id="niyatirajukumar/fraud-detection-models",
    filename="early_model.pkl"
)

delayed_model_path = hf_hub_download(
    repo_id="niyatirajukumar/fraud-detection-models",
    filename="delayed_model.pkl"
)

early_model = joblib.load(early_model_path)
delayed_model = joblib.load(delayed_model_path)


MERCHANT_MAP = {
    "Groceries": 0,
    "Electronics": 1,
    "Gaming": 2,
    "Luxury": 3,
    "Travel": 4
}

COUNTRY_MAP = {
    "INDIA": 0,
    "Canada": 1,
    "UK": 2,
    "Germany": 3,
    "France": 4
}

PAYMENT_MAP = {
    "Credit Card": 0,   
    "Debit Card": 1,
    "Net Banking": 2,
    "UPI": 3,
    "Wallet": 4
}

CHARGEBACK_MAP = {
    "No" : 0,
    "Yes": 1
}

# -----------------------------
# Early Prediction
# -----------------------------
def early_prediction(amount, merchant, country, hour, payment):
    merchant_encoded = MERCHANT_MAP[merchant]
    country = COUNTRY_MAP[country]
    payment = PAYMENT_MAP[payment]
    X = [[amount, merchant_encoded, country, hour, payment]]
    risk = early_model.predict_proba(X)[0][1]

    decision = "FLAG" if risk > 0.7 else "ALLOW"
    return round(float(risk), 3), decision


# -----------------------------
# Final Decision Logic
# -----------------------------
def final_decision_logic(early_risk, delayed_risk, device_risk, velocity, chargebacks):

    # HARD SAFE OVERRIDE (CASE-1 FIX)
    if (
        device_risk < 0.2 and
        velocity <= 2 and
        chargebacks == 0
    ):
        return delayed_risk * 0.3, "ALLOW"

    # HARD FRAUD OVERRIDE (CASE-2)
    if (
        device_risk > 0.8 or
        velocity >= 6 or
        chargebacks == 1
    ):
        return max(early_risk, delayed_risk), "FLAG"

    # fallback
    final_risk = 0.4 * early_risk + 0.6 * delayed_risk
    return final_risk, "FLAG" if final_risk > 0.6 else "ALLOW"


# -----------------------------
# Updated Prediction
# -----------------------------
def updated_prediction(
    early_risk, amount, merchant, country, hour, payment,
    device_risk, velocity, chargebacks
):
    merchant_encoded = MERCHANT_MAP[merchant]
    country = COUNTRY_MAP[country]
    payment = PAYMENT_MAP[payment]
    chargebacks_encoded = CHARGEBACK_MAP[chargebacks]

    X = [[
        amount, merchant_encoded, country, hour, payment,
        device_risk, velocity, chargebacks_encoded
    ]]

    # Get delayed model risk
    delayed_risk = delayed_model.predict_proba(X)[0][1]

    # Apply final decision logic
    final_risk, decision = final_decision_logic(
        early_risk,
        delayed_risk,
        device_risk,
        velocity,
        chargebacks_encoded
    )

    return round(float(final_risk), 3), decision


# Gradio UI 
with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Fraud Detection with Incomplete & Delayed Data")

    gr.Markdown("### Early Risk Prediction")
    e_amount = gr.Number(label="Amount")
    e_merchant = gr.Dropdown(
        choices=list(MERCHANT_MAP.keys()),
        label="Merchant Category",
        value="Groceries"
    )
    e_country = gr.Dropdown(
        choices=list(COUNTRY_MAP.keys()),
        label="Country",
        value="INDIA"
    )
    e_hour = gr.Number(label="Hour(24-Hr Format)")

    e_payment = gr.Dropdown(
        choices=list(PAYMENT_MAP.keys()),
        label="Payment Method",
        value="Credit Card"
    )

    early_btn = gr.Button("Predict Early Risk")
    early_risk = gr.Number(label="Early Risk Score")
    early_decision = gr.Text(label="Decision")

    early_btn.click(
        early_prediction,
        [e_amount, e_merchant, e_country, e_hour, e_payment],
        [early_risk, early_decision]
    )

    gr.Markdown("### Updated Risk (After Delayed Data)")
    d_device = gr.Number(label="Device Risk (0=less risk; 1=most risk)")
    d_velocity = gr.Number(label="Transaction Velocity (No of Transactions in last 24 hrs)")
    d_charge = gr.Dropdown(
        choices=["No", "Yes"],
        label="Past Chargebacks",
        value="No"
    )

    update_btn = gr.Button("Update Risk")
    final_risk = gr.Number(label="Final Risk Score")
    final_decision = gr.Text(label="Final Decision")

    update_btn.click(
        updated_prediction,
        [
            early_risk,
            e_amount, e_merchant, e_country, e_hour, e_payment,
            d_device, d_velocity, d_charge
        ],
        [final_risk, final_decision]
    )

demo.launch()
