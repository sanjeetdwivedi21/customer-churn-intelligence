# Architecture

## Data flow

```
data/raw/telco_churn.csv
        |  src/db_setup.py + sql/01, 02
        v
db/churn.db   raw_customers (all text)  ->  customers (typed, constrained)  ->  vw_customer_features
        |
        |-- sql/03, 04 via src/run_queries.py        business queries and segmentation
        |-- src/data_preprocessing.py                loads tables into pandas
        v
src/feature_engineering.py     features + preprocessor (inside the Pipeline)
        |
        v
src/model_training.py          split (80/20, stratified, seed 42) and baseline comparison
src/model_tuning.py            tuning, threshold, model choice  ->  models/final_model.joblib
        |                                                           models/model_metadata.json
        v
src/model_evaluation.py        one test-set evaluation, plots, reports/model_evaluation.md
src/prediction.py              inference only: score customers, assign risk tier
src/explainability.py          SHAP values, scored_customers.csv, shap_values.csv, figures
src/ai_recommendations.py      rule engine, optional LLM, labelled output
        |
        v
dashboard/app.py               reads saved files only (no training, no SHAP at runtime)
```

## Layers

| Layer | Files | Responsibility |
|---|---|---|
| Configuration | `src/config.py` | Paths, seed, tuning settings, business assumptions |
| Data | `sql/`, `db_setup.py`, `data_preprocessing.py` | Raw and clean tables, validation, loading |
| Features | `feature_engineering.py` | Deterministic feature building; learned transforms live in the pipeline |
| Training | `model_training.py`, `model_tuning.py` | Split, comparison, tuning, threshold, model selection |
| Evaluation | `model_evaluation.py`, `business_metrics.py` | Test metrics, curves, calibration, net benefit |
| Inference | `prediction.py` | Load artifacts, validate input, score, risk tiers |
| Explainability | `explainability.py` | SHAP, grouped drivers, plain-language descriptions |
| Recommendations | `ai_recommendations.py` | Facts, rules, optional LLM, source labels |
| Presentation | `dashboard/` | Five tabs, filters, charts, downloadable report |
| Quality | `tests/` | 36 unit and integration tests |

## Key decisions

| Decision | Reason |
|---|---|
| SQLite with standard SQL | No setup; queries port to PostgreSQL |
| Raw layer plus clean layer | Cleaning can be rebuilt without re-downloading; constraints stop bad data |
| One scikit-learn `Pipeline` | Scaling and encoding fit on training folds only, so no leakage |
| Test set used once | Honest final score; model and threshold fixed beforehand |
| Threshold from net benefit, not 0.5 | An offer costs far less than the value of a saved customer |
| No class weights | Keeps probabilities calibrated for revenue at risk |
| `monthly_charges` and `total_charges` excluded from the model | Collinear with tenure, internet type and add-ons; gave misleading coefficients |
| Rule engine as default recommender | Works offline, never invents facts; LLM is an optional upgrade with fallback |
| Dashboard reads saved files | Fast, and the model shown is exactly the evaluated one |

## Artifacts

| File | Produced by | Used by |
|---|---|---|
| `models/final_model.joblib`, `model_metadata.json` | `model_tuning` | prediction, explainability, dashboard |
| `models/tuning_summary.csv`, `threshold_sweep_oof.csv` | `model_tuning` | evaluation report |
| `data/processed/scored_customers.csv`, `shap_values.csv` | `explainability` | recommendations, dashboard |
| `reports/model_evaluation.md`, `reports/figures/` | `model_evaluation`, EDA, SHAP | README |
| `reports/retention_recommendations.csv` | `ai_recommendations` | review |

## Rebuilding

The pipeline is deterministic (fixed seeds). Deleting `db/`, `models/` and `data/processed/` and re-running the commands in the README reproduces the same model, threshold and metrics.