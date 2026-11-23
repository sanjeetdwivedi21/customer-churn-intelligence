# Model Evaluation Report

## Summary
- **Selected model:** Logistic Regression (chosen on training data only)
- **Decision threshold:** 0.27, chosen by maximising estimated net benefit on
  out-of-fold training predictions. Risk tiers: Medium >= 0.27,
  High >= 0.50.
- **Test set:** evaluated once, after model and threshold were fixed.

## Business assumptions (not from the dataset)
Offer cost $50, offer success rate 25%,
value of a saved customer = 12 months of their monthly charge.
All net-benefit figures depend on these and are estimates.

## Model comparison (training data, out-of-fold)
| model | cv_roc_auc | oof_pr_auc | oof_brier | threshold | recall | precision | flagged_pct | net_benefit |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8452 | 0.6571 | 0.1353 | 0.27 | 0.793 | 0.529 | 39.8 | 165177.0 |
| XGBoost | 0.85 | 0.6673 | 0.1334 | 0.19 | 0.88 | 0.477 | 48.9 | 163444.0 |
| Gradient Boosting | 0.8499 | 0.6652 | 0.1336 | 0.26 | 0.805 | 0.525 | 40.7 | 162266.0 |
| Random Forest | 0.8477 | 0.6571 | 0.1349 | 0.27 | 0.814 | 0.516 | 41.9 | 161086.0 |

## Test-set metrics
| setting | threshold | accuracy | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|
| threshold 0.50 | 0.5 | 0.796 | 0.642 | 0.527 | 0.579 | 0.84 | 0.631 |
| business threshold | 0.27 | 0.746 | 0.514 | 0.778 | 0.619 | 0.84 | 0.631 |

The first row uses the default 0.5 threshold, the second the business threshold.
ROC-AUC 0.840, PR-AUC 0.631, Brier score 0.139.

## Strategy comparison (test set, estimated net benefit)
| strategy | customers_flagged | recall | precision | net_benefit_usd |
|---|---|---|---|---|
| Contact nobody | 0 | 0.0 | 0.0 | 0 |
| Contact everyone | 1409 | 1.0 | 0.265 | 11,195 |
| SQL rule baseline | 191 | 0.313 | 0.613 | 14,818 |
| Logistic Regression @ 0.27 | 566 | 0.778 | 0.514 | 38,221 |

## Calibration check
Expected churners from model probabilities: 377; actual: 374.
Expected monthly revenue at risk: $27,935; actual churned: $27,215.
If these are close, probability-based "revenue at risk" is trustworthy.

## Limitations
- Associations in this dataset, not causes.
- The dataset is a snapshot with no usage or complaint data.
- Offer cost, success rate and horizon are assumptions.
- One 80/20 split: test metrics carry sampling noise (about +/- 0.02).

## Figures
`reports/figures/`: 09_confusion_matrix, 10_roc_curve, 11_precision_recall_curve,
12_calibration, 13_net_benefit_vs_threshold.
