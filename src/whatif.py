"""What-if analysis: how would the MODEL's churn probability change if a customer's setup changed?

This shows what the model predicts, not what would really happen. The data shows associations,
not causes, and monthly charges are kept fixed (the model does not use them).
"""
import pandas as pd

from src.prediction import score_customers

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = ["Electronic check", "Mailed check",
                   "Bank transfer (automatic)", "Credit card (automatic)"]
ADDONS = {"tech_support": "tech support", "online_security": "online security",
          "online_backup": "online backup", "device_protection": "device protection"}


def single_changes(row: pd.Series) -> list[tuple[str, dict]]:
    """Every single change that makes sense for this customer, as (label, changes)."""
    options = []
    for contract in CONTRACTS:
        if contract != row["contract"]:
            options.append((f"Contract: {contract}", {"contract": contract}))
    for method in PAYMENT_METHODS:
        if method != row["payment_method"]:
            options.append((f"Payment: {method}", {"payment_method": method}))
    if row["internet_service"] != "No":      # add-ons only exist with internet service
        for column, name in ADDONS.items():
            if row[column] != "Yes":
                options.append((f"Add {name}", {column: "Yes"}))
    if int(row["paperless_billing"]) == 1:
        options.append(("Switch to paper billing", {"paperless_billing": 0}))
    else:
        options.append(("Switch to paperless billing", {"paperless_billing": 1}))
    return options


def score_options(row: pd.Series, options: list[tuple[str, dict]], model, metadata) -> pd.DataFrame:
    """Score the current setup and each option. Row 0 of the result is the current setup.

    row: one customer in the cleaned database format (as in the dashboard's data frame).
    """
    base = row.to_dict()
    records = [base] + [{**base, **changes} for _, changes in options]
    scored = score_customers(pd.DataFrame(records), model, metadata)
    result = pd.DataFrame({
        "change": ["Current setup"] + [label for label, _ in options],
        "churn_probability": scored["churn_probability"].to_numpy(),
        "risk_level": scored["risk_level"].to_numpy(),
        "revenue_at_risk": scored["revenue_at_risk"].to_numpy(),
    })
    baseline = float(result.loc[0, "churn_probability"])
    result["change_pct_points"] = ((result["churn_probability"] - baseline) * 100).round(1)
    return result