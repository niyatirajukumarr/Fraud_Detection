# Fraud Detection with Incomplete & Delayed Data

## Problem
Detect fraudulent transactions early when limited data is available and
update the risk score as delayed signals arrive, while explaining every decision.

## Solution Overview
- Early-stage fraud detection using partial transaction data
- Dynamic risk score updates when delayed features become available
- Explainable predictions using SHAP
- Deployed as an interactive web app

## Architecture
1. Early fraud model (LightGBM)
2. Delayed fraud model with richer signals
3. Risk update logic combining both stages
4. Gradio UI for live demonstration

## Tech Stack
- Python
- LightGBM
- SHAP
- FastAPI
- Gradio
- Hugging Face Spaces

## Demo Flow
1. Enter basic transaction details → get early risk
2. Add delayed information → updated risk score
3. See transparent fraud decision

