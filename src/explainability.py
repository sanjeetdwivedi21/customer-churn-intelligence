"""Phase 6: SHAP explanations for the saved churn model.

Run from the project root:  python -m src.explainability
Optional:                   python -m src.explainability --customer-id 7590-VHVEG

SHAP values are in log-odds: positive pushes the prediction towards churn, negative away.
They describe what the MODEL does, not what causes churn.
"""
import argparse
import logging
import sys

import matplotlib
matplotlib.use("Agg")  # draw to files only
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR
from src.data_preprocessing import load_customers
from src.feature_engineering import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from src.model_training import load_split
from src.prediction import load_artifacts, prepare_features, score_customers

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_TREES = (GradientBoostingClassifier,) + ((XGBClassifier,) if XGBClassifier else ())

# (text when value is 1, text when value is 0)
BINARY_TEXT = {
    "senior_citizen": ("Senior citizen", "Not a senior citizen"),
    "has_partner": ("Has a partner", "No partner"),
    "has_dependents": ("Has dependents", "No dependents"),
    "phone_service": ("Has phone service", "No phone service"),
    "paperless_billing": ("Uses paperless billing", "Does not use paperless billing"),
    "multiple_lines": ("Has multiple phone lines", "No multiple phone lines"),
    "online_security": ("Has online security", "No online security add-on"),
    "online_backup": ("Has online backup", "No online backup add-on"),
    "device_protection": ("Has device protection", "No device protection add-on"),
    "tech_support": ("Has tech support", "No tech support add-on"),
    "streaming_tv": ("Has streaming TV", "No streaming TV"),
    "streaming_movies": ("Has streaming movies", "No streaming movies"),
    "autopay": ("Pays by automatic payment", "Does not use automatic payment"),
}


# ---------- SHAP computation ----------

def build_explainer(pipeline, X_train: pd.DataFrame):
    """Pick the right SHAP explainer for the model. Background data = training set only."""
    preprocessor, classifier = pipeline[:-1], pipeline[-1]
    if isinstance(classifier, LogisticRegression):
        background = preprocessor.transform(X_train)
        masker = shap.maskers.Independent(background, max_samples=len(background))
        return shap.LinearExplainer(classifier, masker)
    if isinstance(classifier, SUPPORTED_TREES):
        return shap.TreeExplainer(classifier)
    raise RuntimeError(f"No SHAP explainer set up for {type(classifier).__name__}")


def compute_shap(pipeline, explainer, X: pd.DataFrame):
    """Return (transformed features, SHAP values per encoded column, base value)."""
    X_t = pipeline[:-1].transform(X)
    values = explainer.shap_values(X_t)
    if isinstance(values, list):          # some versions return one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    base = float(np.ravel(explainer.expected_value)[-1])
    shap_t = pd.DataFrame(values, index=X.index, columns=X_t.columns)

    # Safety check: base value + SHAP values must rebuild the model's own log-odds
    p = np.clip(pipeline.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    gap = np.abs(base + shap_t.sum(axis=1).to_numpy() - np.log(p / (1 - p))).max()
    if gap > 0.01:
        raise RuntimeError(f"SHAP additivity check failed (max gap {gap:.4f})")
    logger.info("Additivity check passed (max gap %.6f)", gap)
    return X_t, shap_t, base


def group_shap(shap_t: pd.DataFrame) -> pd.DataFrame:
    """Merge one-hot columns back into their original feature (contract_* -> contract)."""
    mapping = {col: next((f for f in CATEGORICAL_FEATURES if col.startswith(f + "_")), col)
               for col in shap_t.columns}
    return shap_t.T.groupby(mapping).sum().T[FEATURE_COLUMNS]


def global_importance(grouped: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP per feature, with its share of the total."""
    importance = grouped.abs().mean().sort_values(ascending=False)
    table = pd.DataFrame({"feature": importance.index, "mean_abs_shap": importance.values.round(4)})
    table["share_pct"] = (100 * table["mean_abs_shap"] / table["mean_abs_shap"].sum()).round(1)
    return table


# ---------- Business-language explanation ----------

def describe_feature(feature: str, value) -> str:
    """Turn a feature and the customer's actual value into a readable phrase.

    monthly_charges and total_charges are not model features anymore, so they are not described.
    """
    if feature == "tenure_months":
        return f"Tenure of {int(value)} months"
    if feature == "contract":
        return f"{value} contract"
    if feature == "internet_service":
        return "No internet service" if value == "No" else f"{value} internet service"
    if feature == "payment_method":
        return f"Pays by {str(value).lower()}"
    yes_text, no_text = BINARY_TEXT[feature]
    return yes_text if int(value) == 1 else no_text


def explain_customer(customer_id: str, X: pd.DataFrame, grouped: pd.DataFrame,
                     scored: pd.DataFrame, top_n: int = 4) -> dict:
    """Structured explanation for one customer. Phase 7 passes this to the LLM."""
    matches = scored.index[scored["customer_id"] == customer_id]
    if len(matches) == 0:
        raise ValueError(f"Customer '{customer_id}' not found")
    idx = matches[0]
    contributions = grouped.loc[idx]

    def pick(series: pd.Series) -> list[dict]:
        return [{"feature": f, "description": describe_feature(f, X.loc[idx, f]),
                 "impact": round(float(v), 3)} for f, v in series.items()]

    return {
        "customer_id": customer_id,
        "churn_probability": float(scored.loc[idx, "churn_probability"]),
        "risk_level": str(scored.loc[idx, "risk_level"]),
        "monthly_charges": float(scored.loc[idx, "monthly_charges"]),
        "increases_risk": pick(contributions[contributions > 0].sort_values(ascending=False).head(top_n)),
        "decreases_risk": pick(contributions[contributions < 0].sort_values().head(top_n)),
    }


def format_explanation(exp: dict) -> str:
    """Plain-text version. Clearly labelled as a MODEL prediction, not a recommendation."""
    lines = [
        f"MODEL PREDICTION for customer {exp['customer_id']}",
        f"  Churn probability: {exp['churn_probability']:.0%} | Risk level: {exp['risk_level']}",
        "  Factors pushing churn risk UP:",
    ]
    lines += [f"    + {d['description']} (impact {d['impact']:+.2f})" for d in exp["increases_risk"]]
    lines.append("  Factors pushing churn risk DOWN:")
    lines += [f"    - {d['description']} (impact {d['impact']:+.2f})" for d in exp["decreases_risk"]]
    return "\n".join(lines)


# ---------- Plots ----------

def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_global(table: pd.DataFrame) -> None:
    top = table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4c72b0")
    ax.set_xlabel("Mean |SHAP value| (average impact on churn log-odds)")
    ax.set_title("Global feature importance")
    save(fig, "14_shap_global_importance")


def plot_summary(shap_t: pd.DataFrame, X_t: pd.DataFrame) -> None:
    shap.summary_plot(shap_t.to_numpy(), X_t, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    save(fig, "15_shap_summary")


def plot_dependence(feature: str, color_by: str, X: pd.DataFrame,
                    grouped: pd.DataFrame, name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for value, part in X.groupby(color_by):
        ax.scatter(part[feature], grouped.loc[part.index, feature], s=10, alpha=0.5, label=str(value))
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value (log-odds)")
    ax.set_title(f"How {feature} moves the churn prediction")
    ax.legend(title=color_by)
    save(fig, name)


def plot_customer(exp: dict) -> None:
    items = sorted(exp["increases_risk"] + exp["decreases_risk"], key=lambda d: d["impact"])
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(items) + 1.6))
    ax.barh([d["description"] for d in items], [d["impact"] for d in items],
            color=["#c44e52" if d["impact"] > 0 else "#4c72b0" for d in items])
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Impact on churn log-odds (red = raises risk, blue = lowers risk)")
    ax.set_title(f"Customer {exp['customer_id']}: {exp['churn_probability']:.0%} churn probability "
                 f"({exp['risk_level']} risk)")
    save(fig, "17_customer_explanation")


# ---------- Main ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="SHAP explanations for the churn model.")
    parser.add_argument("--customer-id", default=None,
                        help="Customer to explain (default: highest-risk active customer)")
    args = parser.parse_args()

    try:
        model, metadata = load_artifacts()
        customers = load_customers()
        X = prepare_features(customers)
        X_train = load_split()[0]  # training rows only, used as SHAP background

        scored = score_customers(customers, model, metadata)
        explainer = build_explainer(model, X_train)
        X_t, shap_t, _ = compute_shap(model, explainer, X)
        grouped = group_shap(shap_t)
        table = global_importance(grouped)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        scored.to_csv(PROCESSED_DIR / "scored_customers.csv", index=False)
        grouped.assign(customer_id=scored["customer_id"]).to_csv(
            PROCESSED_DIR / "shap_values.csv", index=False)
        table.to_csv(REPORTS_DIR / "shap_global_importance.csv", index=False)

        plot_global(table)
        plot_summary(shap_t, X_t)
        plot_dependence("tenure_months", "contract", X, grouped, "16a_shap_dependence_tenure")
        plot_dependence("tenure_months", "internet_service", X, grouped,
                        "16b_shap_dependence_tenure_by_internet")

        if args.customer_id:
            target_id = args.customer_id
        else:
            active = scored[scored["actual_churn"] == 0]
            target_id = scored.loc[active["churn_probability"].idxmax(), "customer_id"]
        exp = explain_customer(target_id, X, grouped, scored)
        plot_customer(exp)

        print(f"\nModel: {metadata['model_name']} | SHAP units: log-odds\n")
        print("Global feature importance (top 10):\n")
        print(table.head(10).to_string(index=False))
        print("\n" + format_explanation(exp))

        active = scored[scored["actual_churn"] == 0]
        print("\nActive customers by risk level (in-sample scores, preview of the dashboard KPIs):")
        print(active["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).to_string())
        print(f"Expected monthly revenue at risk (active customers): "
              f"${active['revenue_at_risk'].sum():,.0f}")
        logger.info("Saved scored_customers.csv, shap_values.csv and figures 14 to 17")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("Failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())