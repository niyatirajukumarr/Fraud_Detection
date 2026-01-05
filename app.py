import gradio as gr
import joblib
import numpy as np
from models.risk_update import update_risk

early_model = joblib.load("models/early_model.pkl")
delayed_model = joblib.load("models/delayed_model.pkl")

def early_prediction(amount, merchant, country, hour, payment):
    features = np.array([[amount, merchant, country, hour, payment]])
    risk = early_model.predict_proba(features)[0][1]

    return round(float(risk), 3), "FLAG" if risk > 0.7 else "ALLOW"

def updated_prediction(early_risk, amount, merchant, country, hour, payment,
                       device_risk, velocity, chargebacks):
    features = np.array([[amount, merchant, country, hour, payment,
                          device_risk, velocity, chargebacks]])
    delayed_risk = delayed_model.predict_proba(features)[0][1]
    final_risk = update_risk(early_risk, delayed_risk)

    return round(float(final_risk), 3), "FLAG" if final_risk > 0.7 else "ALLOW"

with gr.Blocks() as demo:
    gr.Markdown("## 🔍 Fraud Detection with Incomplete & Delayed Data")

    gr.Markdown("### Early Risk Prediction")
    e_amount = gr.Number(label="Amount")
    e_merchant = gr.Number(label="Merchant Category")
    e_country = gr.Number(label="Country")
    e_hour = gr.Number(label="Hour")
    e_payment = gr.Number(label="Payment Method")

    early_btn = gr.Button("Predict Early Risk")
    early_risk = gr.Number(label="Early Risk Score")
    early_decision = gr.Text(label="Decision")

    early_btn.click(
        early_prediction,
        [e_amount, e_merchant, e_country, e_hour, e_payment],
        [early_risk, early_decision]
    )

    gr.Markdown("### Updated Risk (After Delayed Data)")
    d_device = gr.Number(label="Device Risk")
    d_velocity = gr.Number(label="Transaction Velocity")
    d_charge = gr.Number(label="Past Chargebacks")

    update_btn = gr.Button("Update Risk")
    final_risk = gr.Number(label="Final Risk Score")
    final_decision = gr.Text(label="Final Decision")

    update_btn.click(
        updated_prediction,
        [early_risk, e_amount, e_merchant, e_country, e_hour, e_payment,
         d_device, d_velocity, d_charge],
        [final_risk, final_decision]
    )

demo.launch()
