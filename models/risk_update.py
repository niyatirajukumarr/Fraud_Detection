def update_risk(early_risk, delayed_risk, alpha=0.6):
    return alpha * early_risk + (1 - alpha) * delayed_risk
