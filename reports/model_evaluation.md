# Model Evaluation Report

## Summary
- **Selected model:** Logistic Regression (chosen on training data only)
- **Decision threshold:** 0.26, chosen by maximising estimated net benefit on
  out-of-fold training predictions. Risk tiers: Medium >= 0.26,
  High >= 0.50.
- **Test set:** evaluated once, after model and threshold were fixed.

## Business assumptions (not from the dataset)
Offer cost $50, offer success rate 25%,
value of a saved customer = 12 months of their monthly charge.
All net-benefit figures depend on these and are estimates.

## Model comparison (training data, out-of-fold)
| model | cv_roc_auc | oof_pr_auc | oof_brier | threshold | recall | precision | flagged_pct | net_benefit |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8466 | 0.6581 | 0.1349 | 0.26 | 0.809 | 0.515 | 41.7 | 164889.0 |
| XGBoost | 0.8499 | 0.6677 | 0.1333 | 0.21 | 0.86 | 0.49 | 46.5 | 163955.0 |
| Gradient Boosting | 0.8496 | 0.6634 | 0.1338 | 0.2 | 0.861 | 0.488 | 46.8 | 162922.0 |
| Random Forest | 0.8476 | 0.6569 | 0.1348 | 0.28 | 0.803 | 0.525 | 40.6 | 160872.0 |

## Test-set metrics
| setting | threshold | accuracy | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|
| threshold 0.50 | 0.5 | 0.803 | 0.653 | 0.553 | 0.599 | 0.841 | 0.628 |
| business threshold | 0.26 | 0.737 | 0.503 | 0.794 | 0.616 | 0.841 | 0.628 |

The first row uses the default 0.5 threshold, the second the business threshold.
ROC-AUC 0.841, PR-AUC 0.628, Brier score 0.138.

## Strategy comparison (test set, estimated net benefit)
| strategy | customers_flagged | recall | precision | net_benefit_usd |
|---|---|---|---|---|
| Contact nobody | 0 | 0.0 | 0.0 | 0 |
| Contact everyone | 1409 | 1.0 | 0.265 | 11,195 |
| SQL rule baseline | 191 | 0.313 | 0.613 | 14,818 |
| Logistic Regression @ 0.26 | 591 | 0.794 | 0.503 | 38,055 |

## Calibration check
Expected churners from model probabilities: 377; actual: 374.
Expected monthly revenue at risk: $28,024; actual churned: $27,215.
If these are close, probability-based "revenue at risk" is trustworthy.

## Limitations
- Associations in this dataset, not causes.
- The dataset is a snapshot with no usage or complaint data.
- Offer cost, success rate and horizon are assumptions.
- One 80/20 split: test metrics carry sampling noise (about +/- 0.02).

## Figures
`reports/figures/`: 09_confusion_matrix, 10_roc_curve, 11_precision_recall_curve,
12_calibration, 13_net_benefit_vs_threshold.
