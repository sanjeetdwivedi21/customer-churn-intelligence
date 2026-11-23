# Executive Summary: Customer Churn Intelligence

## The situation

Of 7,043 customers, 1,869 (26.5%) have churned. Because churned customers paid more than average, they held 30.5% of monthly revenue: about $139,000 a month, or roughly $1.67 million a year.

## What the data shows

1. **Contract is the strongest lever.** Month-to-month customers churn at 42.7%, against 11.3% on one-year and 2.8% on two-year contracts. They are 55% of customers but 89% of churners.
2. **The first year is the danger zone.** Churn is 47.4% in the first 12 months and 9.5% after four years.
3. **One segment stands out.** Month-to-month customers in their first year on fiber optic churn at 70.2%. They are 13% of customers and 34% of all churners.
4. **Payment method and add-ons are risk markers.** Electronic check users churn at 45.3% (others 15% to 19%). Internet customers without tech support churn at 41.6% against 15.2% with it.
5. **Gender has no visible effect.**

These are associations. The dataset cannot show that changing a contract or adding tech support would reduce churn, so those ideas should be tested before being rolled out widely.

## What the model adds

A Logistic Regression model ranks customers by churn probability. On a held-out test set it reaches ROC-AUC 0.84 and finds 78% of churners while contacting 40% of customers. Its probabilities are well calibrated: 377 churners expected, 374 actual.

Under the assumptions below, contacting customers the model flags is estimated to be worth about **$38,200** on the 1,409-customer test set, against $14,800 for a simple SQL rule and $11,200 for contacting everyone. Contacting nobody is worth $0.

**Assumptions (not from the data):** $50 per retention offer, 25% of would-be churners accept, a saved customer is worth 12 months of their bill.

## Who to contact first

Among customers who have not yet churned, the model flags 514 as high risk and 815 as medium risk. Their expected monthly revenue at risk is about **$68,000** (probability times monthly charge, an in-sample estimate). The dashboard ranks active customers by revenue at risk, so the list starts with high-value customers who are most likely to leave.

A simple starting list, even without the model: 352 active customers are on a month-to-month contract with 12 months or less of tenure and pay by electronic check. They bill about $22,400 a month in total.

## Recommended actions

| Priority | Action | Target | Why |
|---|---|---|---|
| 1 | Proactive check-in call | Customers in their first 12 months with high risk | Churn is highest early |
| 2 | Contract incentive (one or two year) | High-risk month-to-month customers | Commitment is associated with lower churn |
| 3 | Review plan fit | High-risk fiber customers | Fiber has the highest churn among internet types |
| 4 | Encourage automatic payment | Electronic check users | Highest-churn payment method |
| 5 | Offer a trial of tech support or online security | Internet customers without them | Much lower churn among customers who have them |

Start with a pilot on part of the contact list and keep a control group. That measures the real success rate, which this analysis can only assume.

## How to reduce revenue loss

- Focus effort on the high-risk, high-bill customers first.
- Move month-to-month customers to longer contracts where they are willing.
- Protect new customers in their first year.
- Measure results and replace the assumptions above with real campaign numbers.

## Limits of this analysis

- Associations, not causes.
- No usage, complaint or pricing history in the data.
- Business assumptions are estimates.
- Dashboard scores are slightly optimistic for customers the model trained on; test-set results above are the honest measure.