#  Fraud Detection with Incomplete & Delayed Data

---

##  Problem Statement

Fraud detection systems in the real world must often make decisions **before all information is available**.  
Critical signals such as **device trust**, **behavioral patterns**, and **historical activity** typically arrive **after** a transaction has already been initiated.

### Traditional Fraud Models Assume:
- All features are available upfront  
- A single prediction is final  
- Decisions do not change over time  

### This Leads To:
- ❌ High false positives (blocking genuine users)  
- ❌ High false negatives (missing fraudulent activity)  
- ❌ Poor explainability and low user trust  

---

##  Objective

To build a fraud detection system that:

- Detects fraud **early** using incomplete transaction data  
- **Dynamically updates risk** as delayed signals become available  
- **Explains every decision clearly** to ensure transparency and trust  

---

##  Solution Overview

We designed a **two-stage, dynamic fraud detection system** that closely mimics how **real financial institutions** operate.

---

### 🔹 Stage 1: Early Risk Prediction

- Uses **immediately available** transaction features  
- Generates an **early risk score**  
- Intentionally **conservative** to minimize missed fraud  

---

### 🔹 Stage 2: Updated Risk Prediction

- Incorporates **delayed signals**, such as:
  - Device risk
  - Transaction velocity
  - Past chargebacks  
- Updates the fraud risk score  
- Allows decisions to **change over time**

This enables:

- ✅ **FLAG → ALLOW** (false positive correction)  
- ✅ **ALLOW → FLAG** (late fraud detection)  

---

##  System Architecture


<img width="373" height="366" alt="image" src="https://github.com/user-attachments/assets/90b67c12-d281-44a9-a8cc-bf2e875a41a9" />

---

##  Features Used

### Early Features (Available Immediately)
- Transaction amount  
- Merchant category  
- Country  
- Hour of transaction  
- Payment method  

### Delayed Features (Arrive Later)
- Device risk score  
- Transaction velocity  
- Past chargebacks  

---

##  Decision Logic

The system combines **machine learning predictions** with **business rules**:

- Trusted devices and normal behavior can **override early flags**  
- Strong fraud signals can **override early allows**  
- Balances **security** with **customer experience**

This **hybrid ML + rules design** reflects real-world fraud systems, where machine learning models are **not used in isolation**.

---

##  Explainability with SHAP

Fraud detection systems require **transparency, trust, and auditability**.

We use **SHAP (SHapley Additive exPlanations)** to:

- Explain predictions **at a per-transaction level**  
- Show how each feature **contributes to the risk score**  
- Transform the model from a **black box** into a **glass box**

### Why SHAP?

- Works seamlessly with **tree-based models** like LightGBM  
- Provides **consistent and theoretically grounded explanations**  
- Is an **industry standard** in financial machine learning  

This enables explanations such as:

> *“The transaction was initially flagged due to high amount and late hour, but later allowed because the device was trusted and behavior was normal.”*

---

##  User Interface

The system is deployed as an **interactive web application**, allowing users to:

1. Enter transaction details to obtain an **early risk score**  
2. Provide delayed signals to **update the risk**  
3. Observe how decisions **evolve in real time**

This makes the system **intuitive, transparent, and easy to demo**.

---

##  Tech Stack

- **Python**  
- **LightGBM** – Fraud risk modeling  
- **SHAP** – Explainability  
- **Gradio** – Interactive UI  
- **Hugging Face Spaces** – Deployment  

---

##  Key Innovations

- Handles **incomplete and delayed data**  
- Dynamic risk updates instead of static predictions  
- Explicit handling of both **false positives** and **false negatives**  
- Fully **explainable decisions** using SHAP  
- Hybrid **ML + rule-based** system design  

---

##  Conclusion

This project goes beyond traditional fraud detection by modeling fraud as a **risk journey**, not a one-time classification.

By combining **early detection**, **delayed updates**, and **clear explainability**, the system is:

- ✅ Realistic  
- ✅ Trustworthy  
- ✅ Production-inspired

---
