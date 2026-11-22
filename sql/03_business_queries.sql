-- Phase 2: core business queries. Each statement ends with a semicolon.

-- Q1: Executive KPIs
SELECT
    COUNT(*)                                                     AS total_customers,
    SUM(churn)                                                   AS churned_customers,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2)                      AS churn_rate_pct,
    ROUND(SUM(monthly_charges), 2)                               AS total_monthly_revenue,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost,
    ROUND(100.0 * SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END)
          / SUM(monthly_charges), 2)                             AS revenue_lost_pct
FROM customers;

-- Q2: Churn by contract type
SELECT
    contract,
    COUNT(*)                                AS customers,
    SUM(churn)                              AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)          AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- Q3: Churn by tenure band
SELECT
    tenure_band,
    COUNT(*)                                AS customers,
    SUM(churn)                              AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM vw_customer_features
GROUP BY tenure_band
ORDER BY MIN(tenure_months);

-- Q4: Churn by internet service
SELECT
    internet_service,
    COUNT(*)                                AS customers,
    SUM(churn)                              AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)          AS avg_monthly_charges
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- Q5: Churn by payment method
SELECT
    payment_method,
    COUNT(*)                                AS customers,
    SUM(churn)                              AS churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- Q6: Do support and security add-ons relate to churn? (internet customers only)
SELECT 'TechSupport' AS service, tech_support AS has_service,
       COUNT(*) AS customers,
       ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE internet_service <> 'No'
GROUP BY tech_support
UNION ALL
SELECT 'OnlineSecurity', online_security,
       COUNT(*),
       ROUND(100.0 * SUM(churn) / COUNT(*), 2)
FROM customers
WHERE internet_service <> 'No'
GROUP BY online_security
ORDER BY service, has_service;

-- Q7: Riskiest contract and payment combinations (window function, minimum 30 customers)
WITH combo AS (
    SELECT
        contract,
        payment_method,
        COUNT(*)                                AS customers,
        SUM(churn)                              AS churned,
        ROUND(100.0 * SUM(churn) / COUNT(*), 2) AS churn_rate_pct
    FROM customers
    GROUP BY contract, payment_method
    HAVING COUNT(*) >= 30
)
SELECT RANK() OVER (ORDER BY churn_rate_pct DESC) AS risk_rank, *
FROM combo
ORDER BY risk_rank;