"""Phase 7: personalised retention recommendations.

Works with NO API key (rule-based engine). If ANTHROPIC_API_KEY is set in .env, an LLM
writes the recommendation from the same facts, and falls back to the rules on any failure.

Run from the project root:
    python -m src.ai_recommendations                      # top 10 high-risk active customers
    python -m src.ai_recommendations --customer-id 5150-ITWWB
    python -m src.ai_recommendations --top 25 --no-llm

Model predictions and recommendations are always shown as separate, labelled parts.
"""
import argparse
import json
import logging
import os
import re
import sys

import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR
from src.data_preprocessing import load_customers
from src.explainability import describe_feature
from src.prediction import prepare_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MIN_IMPACT = 0.05  # drivers weaker than this are noise and are not passed on
SOURCE_RULES = "Rule-based fallback"
SOURCE_LLM = "AI-generated (LLM)"
REQUIRED_KEYS = ("recommendation", "suggested_action", "expected_objective")


# ---------- Setup ----------

def load_env() -> None:
    """Read KEY=VALUE lines from .env into the environment (no extra package needed)."""
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_inputs():
    """Return (X, scored, grouped_shap), aligned row by row. Uses the Phase 6 outputs."""
    scored_path = PROCESSED_DIR / "scored_customers.csv"
    shap_path = PROCESSED_DIR / "shap_values.csv"
    if not scored_path.exists() or not shap_path.exists():
        raise FileNotFoundError("Run first: python -m src.explainability")

    customers = load_customers()
    X = prepare_features(customers).reset_index(drop=True)
    scored = pd.read_csv(scored_path)
    shap_df = pd.read_csv(shap_path)

    same_order = (scored["customer_id"].tolist() == customers["customer_id"].tolist()
                  == shap_df["customer_id"].tolist())
    if not (len(X) == len(scored) == len(shap_df)) or not same_order:
        raise RuntimeError("Saved scores do not match the database. "
                           "Re-run: python -m src.explainability")
    return X, scored, shap_df.drop(columns="customer_id")


# ---------- Facts (everything the recommendation may use) ----------

def build_facts(i: int, X: pd.DataFrame, scored: pd.DataFrame, grouped: pd.DataFrame) -> dict:
    """Real data and SHAP drivers for customer at row i. Nothing else is allowed downstream."""
    x, contrib = X.iloc[i], grouped.iloc[i]

    def pick(series: pd.Series) -> list[dict]:
        return [{"feature": f, "description": describe_feature(f, x[f]),
                 "impact": round(float(v), 3)} for f, v in series.items()]

    return {
        "customer_id": scored.at[i, "customer_id"],
        "churn_probability": float(scored.at[i, "churn_probability"]),
        "risk_level": str(scored.at[i, "risk_level"]),
        "monthly_charges": float(scored.at[i, "monthly_charges"]),
        "revenue_at_risk": float(scored.at[i, "revenue_at_risk"]),
        "profile": {
            "tenure_months": int(x["tenure_months"]),
            "contract": str(x["contract"]),
            "internet_service": str(x["internet_service"]),
            "payment_method": str(x["payment_method"]),
            "has_tech_support": bool(x["tech_support"]),
            "has_online_security": bool(x["online_security"]),
        },
        "drivers_up": pick(contrib[contrib >= MIN_IMPACT].sort_values(ascending=False).head(5)),
        "drivers_down": pick(contrib[contrib <= -MIN_IMPACT].sort_values().head(3)),
    }


# ---------- Rule-based engine ----------

# feature -> (applies to this customer?, suggested action, objective)
RULES = {
    "contract": (
        lambda p: p["contract"] == "Month-to-month",
        "Offer a loyalty incentive to move to a one-year or two-year contract",
        "increase commitment length"),
    "tenure_months": (
        lambda p: p["tenure_months"] <= 12,
        "Arrange a proactive check-in call during the first year of service",
        "strengthen the early customer relationship"),
    "payment_method": (
        lambda p: p["payment_method"] == "Electronic check",
        "Encourage switching to automatic payment",
        "reduce payment friction"),
    "internet_service": (
        lambda p: p["internet_service"] == "Fiber optic",
        "Review whether the fiber plan still fits the customer's needs",
        "confirm the plan matches how the customer uses the service"),
    "tech_support": (
        lambda p: p["internet_service"] != "No" and not p["has_tech_support"],
        "Offer a trial of the tech support add-on",
        "increase engagement with support services"),
    "online_security": (
        lambda p: p["internet_service"] != "No" and not p["has_online_security"],
        "Offer a trial of the online security add-on",
        "increase engagement with protective services"),
}
MAX_ACTIONS = 3


def rule_based_recommendation(facts: dict) -> dict:
    """Actions only where the model flags a driver AND the customer's real data matches."""
    chosen = []
    for driver in facts["drivers_up"]:  # already sorted, strongest first
        rule = RULES.get(driver["feature"])
        if rule and rule[0](facts["profile"]):
            chosen.append(rule)
        if len(chosen) == MAX_ACTIONS:
            break

    if not chosen:
        if facts["risk_level"] == "Low":
            return {"recommendation": "No targeted retention action is needed. Keep monitoring.",
                    "suggested_action": "Monitor the account",
                    "expected_objective": "Keep the customer's current low churn risk"}
        return {"recommendation": "The model sees elevated risk but no single driver matches a "
                                  "standard action. Review the account manually.",
                "suggested_action": "Manual account review",
                "expected_objective": "Identify the cause of the elevated risk"}

    steps = [f"{i}. {action}" for i, (_, action, _) in enumerate(chosen, 1)]
    objectives = ", ".join(obj for _, _, obj in chosen)
    return {
        "recommendation": "Suggested steps: " + "; ".join(steps) + ".",
        "suggested_action": chosen[0][1],
        "expected_objective": (f"Lower churn risk by aiming to {objectives}, "
                               f"protecting about ${facts['monthly_charges']:,.2f} per month "
                               "in revenue"),
    }


# ---------- Optional LLM ----------

SYSTEM_PROMPT = (
    "You write retention recommendations for a telecom company. Use ONLY the facts in the "
    "JSON you are given. Never invent customer behaviour, complaints, usage, prices, discount "
    "amounts or reasons for churn: the data has none of these. The drivers come from a "
    "statistical model and show associations, not causes. Only suggest actions linked to "
    "drivers in drivers_up. Reply with a JSON object only, with exactly these string keys: "
    "recommendation (2 to 3 sentences), suggested_action (one short sentence), "
    "expected_objective (one short sentence). Write in English."
)


def llm_recommendation(facts: dict) -> dict:
    """Ask the LLM. Raises on any problem so the caller can fall back."""
    import anthropic  # imported here so the project runs without the package

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                 timeout=30.0, max_retries=1)
    message = client.messages.create(
        model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(facts)}],
    )
    text = message.content[0].text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    if not all(isinstance(data.get(k), str) and data[k].strip() for k in REQUIRED_KEYS):
        raise ValueError("LLM reply is missing required fields")
    return {k: data[k].strip() for k in REQUIRED_KEYS}


def generate_recommendation(facts: dict, use_llm: bool = True) -> dict:
    """Full record: model prediction fields + recommendation + its source label."""
    result, source = None, SOURCE_RULES
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            result, source = llm_recommendation(facts), SOURCE_LLM
        except Exception as error:  # any API, network, package or parsing problem
            logger.warning("LLM unavailable (%s). Using rule-based fallback.", type(error).__name__)
    if result is None:
        result = rule_based_recommendation(facts)

    return {
        "customer_id": facts["customer_id"],
        "churn_probability": facts["churn_probability"],
        "risk_level": facts["risk_level"],
        "priority": {"High": "High", "Medium": "Medium", "Low": "Low"}[facts["risk_level"]],
        "monthly_charges": facts["monthly_charges"],
        "revenue_at_risk": facts["revenue_at_risk"],
        "main_reasons": [d["description"] for d in facts["drivers_up"]],
        **result,
        "source": source,
    }


def format_report(rec: dict) -> str:
    """Prediction and recommendation are clearly separated and labelled."""
    reasons = "\n".join(f"    + {r}" for r in rec["main_reasons"]) or "    (none above threshold)"
    return (
        f"MODEL PREDICTION (statistical model) - customer {rec['customer_id']}\n"
        f"  Churn probability: {rec['churn_probability']:.0%} | Risk level: {rec['risk_level']} | "
        f"Expected monthly revenue at risk: ${rec['revenue_at_risk']:,.2f}\n"
        f"  Main factors raising the risk:\n{reasons}\n"
        f"RECOMMENDATION [{rec['source']}] - priority: {rec['priority']}\n"
        f"  {rec['recommendation']}\n"
        f"  Suggested action: {rec['suggested_action']}\n"
        f"  Expected objective: {rec['expected_objective']}"
    )


# ---------- Main ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="Retention recommendations.")
    parser.add_argument("--customer-id", default=None)
    parser.add_argument("--top", type=int, default=10,
                        help="Number of high-risk active customers (default 10)")
    parser.add_argument("--no-llm", action="store_true", help="Force the rule-based engine")
    args = parser.parse_args()

    try:
        load_env()
        X, scored, grouped = load_inputs()
        use_llm = not args.no_llm
        if use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
            logger.info("No ANTHROPIC_API_KEY found: using the rule-based engine.")

        if args.customer_id:
            rows = scored.index[scored["customer_id"] == args.customer_id].tolist()
            if not rows:
                raise ValueError(f"Customer '{args.customer_id}' not found")
        else:
            active_high = scored[(scored["actual_churn"] == 0) & (scored["risk_level"] == "High")]
            rows = active_high.sort_values("revenue_at_risk", ascending=False).head(args.top).index.tolist()

        records = [generate_recommendation(build_facts(i, X, scored, grouped), use_llm) for i in rows]

        if args.customer_id:
            print("\n" + format_report(records[0]))
        else:
            for rec in records[:3]:
                print("\n" + format_report(rec))
            print(f"\n(Showing 3 of {len(records)}. All are in the CSV.)")

            out = pd.DataFrame(records)
            out["main_reasons"] = out["main_reasons"].apply("; ".join)
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out.to_csv(REPORTS_DIR / "retention_recommendations.csv", index=False)
            print(f"Sources used: {out['source'].value_counts().to_dict()}")
            logger.info("Saved reports/retention_recommendations.csv")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("Failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())