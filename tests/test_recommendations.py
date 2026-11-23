import pytest

from src import ai_recommendations as ai
from src.explainability import describe_feature


def make_facts(**overrides) -> dict:
    facts = {
        "customer_id": "T-001", "churn_probability": 0.8, "risk_level": "High",
        "monthly_charges": 90.0, "revenue_at_risk": 72.0,
        "profile": {"tenure_months": 3, "contract": "Month-to-month",
                    "internet_service": "Fiber optic", "payment_method": "Electronic check",
                    "has_tech_support": False, "has_online_security": False},
        "drivers_up": [
            {"feature": "tenure_months", "description": "Tenure of 3 months", "impact": 0.9},
            {"feature": "internet_service", "description": "Fiber optic internet service", "impact": 0.6},
            {"feature": "contract", "description": "Month-to-month contract", "impact": 0.4},
        ],
        "drivers_down": [],
    }
    facts.update(overrides)
    return facts


# ---------- rule-based engine ----------

def test_rules_use_drivers_that_match_real_data():
    rec = ai.rule_based_recommendation(make_facts())
    assert "check-in call" in rec["recommendation"]
    assert "fiber plan" in rec["recommendation"]
    assert "loyalty incentive" in rec["recommendation"]
    assert rec["suggested_action"].startswith("Arrange a proactive check-in")


def test_driver_without_matching_data_creates_no_action():
    # Model flags "contract" but this customer is already on a two-year contract
    facts = make_facts(drivers_up=[{"feature": "contract", "description": "x", "impact": 0.5}])
    facts["profile"]["contract"] = "Two year"
    rec = ai.rule_based_recommendation(facts)
    assert "loyalty incentive" not in rec["recommendation"]
    assert rec["suggested_action"] == "Manual account review"


def test_addon_rule_needs_internet_service():
    facts = make_facts(drivers_up=[{"feature": "tech_support", "description": "x", "impact": 0.5}])
    facts["profile"]["internet_service"] = "No"
    assert ai.rule_based_recommendation(facts)["suggested_action"] == "Manual account review"


def test_driver_with_no_rule_is_ignored():
    facts = make_facts(drivers_up=[{"feature": "multiple_lines", "description": "x", "impact": 0.5}])
    assert ai.rule_based_recommendation(facts)["suggested_action"] == "Manual account review"


def test_low_risk_without_drivers_is_monitored():
    rec = ai.rule_based_recommendation(make_facts(risk_level="Low", drivers_up=[]))
    assert rec["suggested_action"] == "Monitor the account"


def test_at_most_three_steps():
    rec = ai.rule_based_recommendation(make_facts())
    assert "4." not in rec["recommendation"]


def test_no_invented_facts_in_text():
    rec = ai.rule_based_recommendation(make_facts())
    text = " ".join(rec.values()).lower()
    assert "complaint" not in text
    assert "%" not in text   # the dataset has no discount percentages


# ---------- source labels and fallback ----------

def test_no_api_key_uses_rule_engine(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rec = ai.generate_recommendation(make_facts())
    assert rec["source"] == ai.SOURCE_RULES
    assert rec["priority"] == "High"
    for key in ai.REQUIRED_KEYS:
        assert rec[key]


def test_llm_failure_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")

    def broken(_facts):
        raise RuntimeError("network down")

    monkeypatch.setattr(ai, "llm_recommendation", broken)
    assert ai.generate_recommendation(make_facts())["source"] == ai.SOURCE_RULES


def test_llm_success_is_labelled_as_llm(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    fake = {"recommendation": "r", "suggested_action": "a", "expected_objective": "o"}
    monkeypatch.setattr(ai, "llm_recommendation", lambda _facts: fake)
    rec = ai.generate_recommendation(make_facts())
    assert rec["source"] == ai.SOURCE_LLM
    assert rec["suggested_action"] == "a"


def test_use_llm_false_forces_rules(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setattr(ai, "llm_recommendation", lambda _f: pytest.fail("LLM must not be called"))
    assert ai.generate_recommendation(make_facts(), use_llm=False)["source"] == ai.SOURCE_RULES


# ---------- readable descriptions ----------

def test_tenure_grammar():
    assert describe_feature("tenure_months", 1) == "Tenure of 1 month"
    assert describe_feature("tenure_months", 5) == "Tenure of 5 months"


def test_service_descriptions():
    assert describe_feature("internet_service", "No") == "No internet service"
    assert describe_feature("internet_service", "Fiber optic") == "Fiber optic internet service"
    assert describe_feature("contract", "Month-to-month") == "Month-to-month contract"
    assert describe_feature("tech_support", 0) == "No tech support add-on"