-- Phase 2: customer segmentation and a rule-based risk preview.

-- Q1: Top 10 churn segments (contract x tenure band x internet service, minimum 50 customers)
SELECT
    contract,
    tenure_band,
    internet_service,
    COUNT(*)                                AS customers,
    SUM(churn)                              AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(100.0 * SUM(churn) / (SELECT SUM(churn) FROM customers), 2) AS share_of_all_churners_pct
FROM vw_customer_features
GROUP BY contract, tenure_band, internet_service
HAVING COUNT(*) >= 50
ORDER BY churn_rate_pct DESC
LIMIT 10;

-- Q2: Spend quartiles using NTILE (1 = lowest monthly charges, 4 = highest)
WITH ranked AS (
    SELECT *, NTILE(4) OVER (ORDER BY monthly_charges) AS spend_quartile
    FROM customers
)
SELECT
    spend_quartile,
    COUNT(*)                                AS customers,
    MIN(monthly_charges)                    AS min_charge,
    MAX(monthly_charges)                    AS max_charge,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost
FROM ranked
GROUP BY spend_quartile
ORDER BY spend_quartile;

-- Q3: Rule-based high-risk profile versus everyone else
SELECT
    CASE
        WHEN contract = 'Month-to-month'
         AND tenure_months <= 12
         AND payment_method = 'Electronic check'
        THEN 'Risk profile: month-to-month, 12 months or less, electronic check'
        ELSE 'All other customers'
    END AS segment,
    COUNT(*)                                AS customers,
    SUM(churn)                              AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(100.0 * SUM(churn) / (SELECT SUM(churn) FROM customers), 2) AS share_of_all_churners_pct
FROM customers
GROUP BY segment
ORDER BY churn_rate_pct DESC;

-- Q4: Active customers who match the risk profile (a preview of the contact list)
SELECT
    internet_service,
    COUNT(*)                         AS active_customers,
    ROUND(SUM(monthly_charges), 2)   AS monthly_revenue_exposed,
    ROUND(AVG(monthly_charges), 2)   AS avg_monthly_charges
FROM customers
WHERE churn = 0
  AND contract = 'Month-to-month'
  AND tenure_months <= 12
  AND payment_method = 'Electronic check'
GROUP BY internet_service
ORDER BY monthly_revenue_exposed DESC;