def update_risk(early_risk, delayed_risk, alpha=0.6):
    """
    alpha = trust in early signal
    (1-alpha) = trust in delayed signal
    """
    return round(alpha * early_risk + (1 - alpha) * delayed_risk, 3)
