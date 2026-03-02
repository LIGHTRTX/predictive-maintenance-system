def overload_ratio(current, rated_current=5.0):
    return current / rated_current

def unified_risk(failure_prob, overload_ratio, rms):
    normalized_overload = min(overload_ratio/2, 1)
    normalized_rms = min(rms/0.5, 1)
    return 0.4*failure_prob + 0.3*normalized_overload + 0.3*normalized_rms

def classify(risk_score):
    if risk_score < 0.4:
        return "Normal", "green"
    elif risk_score < 0.7:
        return "Warning", "orange"
    else:
        return "Critical", "red"