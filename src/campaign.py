"""Campaign simulator: what a retention campaign is worth under adjustable assumptions.

Nothing here retrains or changes the saved model or its threshold. It only re-prices the
model's predictions with different business assumptions. The assumptions are estimates.
"""
import numpy as np
import pandas as pd

from src.business_metrics import best_threshold, evaluate_flags


def rule_flags(df: pd.DataFrame) -> np.ndarray:
    """The Phase 2 SQL rule: month-to-month, 12 months or less, electronic check."""
    return ((df["contract"] == "Month-to-month")
            & (df["tenure_months"] <= 12)
            & (df["payment_method"] == "Electronic check")).to_numpy()


def expected_value_flags(proba, monthly_charges, offer_cost: float, success_rate: float,
                         horizon_months: float) -> np.ndarray:
    """Contact a customer when the expected gain from the offer exceeds its cost.

    Expected gain = churn probability x success rate x monthly charge x horizon.
    With well-calibrated probabilities this is the cost-optimal rule for each customer.
    It uses only the user's assumptions, nothing is learned from the evaluation data.
    """
    p = np.asarray(proba, dtype=float)
    charges = np.asarray(monthly_charges, dtype=float)
    return p * success_rate * charges * horizon_months >= offer_cost


def break_even_probability(monthly_charge: float, offer_cost: float, success_rate: float,
                           horizon_months: float) -> float:
    """Churn probability above which contacting a customer paying monthly_charge pays off.

    A value above 1 means no customer with that charge is worth contacting.
    """
    gain_if_certain = success_rate * monthly_charge * horizon_months
    return float("inf") if gain_if_certain <= 0 else offer_cost / gain_if_certain


def simulate(df: pd.DataFrame, threshold: float, offer_cost: float, success_rate: float,
             horizon_months: float):
    """Return (strategy table, threshold sweep, best threshold) for the given customers.

    df needs: churn, churn_probability, monthly_charges, contract, tenure_months, payment_method.
    """
    y = df["churn"].to_numpy()
    proba = df["churn_probability"].to_numpy()
    charges = df["monthly_charges"].to_numpy()
    assumptions = {"offer_cost": offer_cost, "success_rate": success_rate,
                   "horizon_months": horizon_months}

    best, sweep = best_threshold(y, proba, charges, **assumptions)
    strategies = [
        ("Contact nobody", np.zeros(len(df), dtype=bool)),
        ("Contact everyone", np.ones(len(df), dtype=bool)),
        ("SQL rule baseline", rule_flags(df)),
        (f"Model at selected threshold {threshold:.2f}", proba >= threshold),
        (f"Model at best threshold {best:.2f}", proba >= best),
        ("Expected-value rule (per customer)",
         expected_value_flags(proba, charges, **assumptions)),
    ]
    rows = []
    for label, flagged in strategies:
        result = evaluate_flags(y, flagged, charges, **assumptions)
        rows.append({
            "strategy": label,
            "contacted": result["flagged"],
            "recall": round(result["recall"], 3),
            "precision": round(result["precision"], 3),
            "expected_saved": round(result["expected_saved"], 1),
            "cost": round(result["cost"]),
            "net_benefit": round(result["net_benefit"]),
        })
    return pd.DataFrame(rows), sweep, best