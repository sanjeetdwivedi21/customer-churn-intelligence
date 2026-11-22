-- Phase 2: raw -> clean transformation, then a view with derived analysis columns.

INSERT INTO customers (
    customer_id, gender, senior_citizen, has_partner, has_dependents, tenure_months,
    phone_service, multiple_lines, internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv, streaming_movies, contract,
    paperless_billing, payment_method, monthly_charges, total_charges, churn
)
SELECT
    TRIM(customerID),
    TRIM(gender),
    CASE TRIM(SeniorCitizen) WHEN '1' THEN 1 WHEN '0' THEN 0 END,
    CASE TRIM(Partner)       WHEN 'Yes' THEN 1 WHEN 'No' THEN 0 END,
    CASE TRIM(Dependents)    WHEN 'Yes' THEN 1 WHEN 'No' THEN 0 END,
    CAST(TRIM(tenure) AS INTEGER),
    CASE TRIM(PhoneService)  WHEN 'Yes' THEN 1 WHEN 'No' THEN 0 END,
    TRIM(MultipleLines),
    TRIM(InternetService),
    TRIM(OnlineSecurity),
    TRIM(OnlineBackup),
    TRIM(DeviceProtection),
    TRIM(TechSupport),
    TRIM(StreamingTV),
    TRIM(StreamingMovies),
    TRIM(Contract),
    CASE TRIM(PaperlessBilling) WHEN 'Yes' THEN 1 WHEN 'No' THEN 0 END,
    TRIM(PaymentMethod),
    CAST(TRIM(MonthlyCharges) AS REAL),
    -- Blank TotalCharges (new customers with tenure 0) becomes NULL, not a guessed number.
    CASE WHEN TRIM(TotalCharges) = '' THEN NULL ELSE CAST(TRIM(TotalCharges) AS REAL) END,
    CASE TRIM(Churn) WHEN 'Yes' THEN 1 WHEN 'No' THEN 0 END
FROM raw_customers;

-- Analysis view: adds bands and service counts so queries stay short and readable.
CREATE VIEW vw_customer_features AS
SELECT
    c.*,
    CASE
        WHEN tenure_months <= 12 THEN '0-12 months'
        WHEN tenure_months <= 24 THEN '13-24 months'
        WHEN tenure_months <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END AS tenure_band,
    CASE
        WHEN monthly_charges < 35 THEN '<$35'
        WHEN monthly_charges < 70 THEN '$35-70'
        WHEN monthly_charges < 90 THEN '$70-90'
        ELSE '$90+'
    END AS charge_band,
    -- How many "stickiness" services does the customer actually have?
    (CASE WHEN online_security   = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN online_backup     = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN device_protection = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN tech_support      = 'Yes' THEN 1 ELSE 0 END) AS protective_services,
    (CASE WHEN streaming_tv      = 'Yes' THEN 1 ELSE 0 END
   + CASE WHEN streaming_movies  = 'Yes' THEN 1 ELSE 0 END) AS streaming_services
FROM customers AS c;