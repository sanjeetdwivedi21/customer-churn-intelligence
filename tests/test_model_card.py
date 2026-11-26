import re

import pandas as pd
import pytest

from src.config import DB_PATH, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR
from src.model_card import held_out_metrics


def test_metrics_hand_calculation():
    # Flagged at 0.5: customers 0 and 1. Only customer 0 churned.
    m = held_out_metrics([1, 0, 1, 0], [0.9, 0.8, 0.4, 0.1], threshold=0.5)
    assert m["customers"] == 4
    assert m["churn_rate"] == 0.5
    assert m["flagged"] == 2
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["accuracy"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)       # 3 of the 4 churner/non-churner pairs ranked right
    assert m["brier"] == pytest.approx(0.255)


def test_no_flagged_customers_does_not_crash():
    m = held_out_metrics([1, 0, 1, 0], [0.4, 0.3, 0.2, 0.1], threshold=0.9)
    assert m["flagged"] == 0
    assert m["precision"] == 0 and m["recall"] == 0


@pytest.mark.skipif(
    not (DB_PATH.exists() and (MODELS_DIR / "final_model.joblib").exists()
         and (PROCESSED_DIR / "scored_customers.csv").exists()
         and (REPORTS_DIR / "model_evaluation.md").exists()),
    reason="Build the project first (db_setup, model_tuning, model_evaluation, explainability)")
def test_card_metrics_agree_with_the_evaluation_report():
    from src.data_preprocessing import load_customers
    from src.model_training import load_split
    from src.prediction import load_artifacts

    _, metadata = load_artifacts()
    customers = load_customers()
    scored = pd.read_csv(PROCESSED_DIR / "scored_customers.csv")
    _, X_test, _, _ = load_split()
    m = held_out_metrics(customers.loc[X_test.index, "churn"],
                         scored.loc[X_test.index, "churn_probability"], metadata["threshold"])

    report = (REPORTS_DIR / "model_evaluation.md").read_text(encoding="utf-8")
    found = re.search(r"ROC-AUC (\d\.\d{3}), PR-AUC (\d\.\d{3})", report)
    assert found, "could not find the ROC-AUC line in reports/model_evaluation.md"
    assert m["customers"] == 1409
    assert m["roc_auc"] == pytest.approx(float(found.group(1)), abs=0.002)
    assert m["pr_auc"] == pytest.approx(float(found.group(2)), abs=0.002)