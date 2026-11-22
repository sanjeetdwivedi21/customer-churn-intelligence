-- Phase 2: table definitions. Running this script rebuilds both tables from scratch.

DROP VIEW  IF EXISTS vw_customer_features;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS raw_customers;

-- Raw layer: every column is TEXT, so nothing is altered or rejected on load.
CREATE TABLE raw_customers (
    customerID       TEXT,
    gender           TEXT,
    SeniorCitizen    TEXT,
    Partner          TEXT,
    Dependents       TEXT,
    tenure           TEXT,
    PhoneService     TEXT,
    MultipleLines    TEXT,
    InternetService  TEXT,
    OnlineSecurity   TEXT,
    OnlineBackup     TEXT,
    DeviceProtection TEXT,
    TechSupport      TEXT,
    StreamingTV      TEXT,
    StreamingMovies  TEXT,
    Contract         TEXT,
    PaperlessBilling TEXT,
    PaymentMethod    TEXT,
    MonthlyCharges   TEXT,
    TotalCharges     TEXT,
    Churn            TEXT
);

-- Clean layer: snake_case names, proper types, binary flags as 0/1, and constraints.
CREATE TABLE customers (
    customer_id        TEXT PRIMARY KEY,
    gender             TEXT    NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen     INTEGER NOT NULL CHECK (senior_citizen IN (0, 1)),
    has_partner        INTEGER NOT NULL CHECK (has_partner IN (0, 1)),
    has_dependents     INTEGER NOT NULL CHECK (has_dependents IN (0, 1)),
    tenure_months      INTEGER NOT NULL CHECK (tenure_months >= 0),
    phone_service      INTEGER NOT NULL CHECK (phone_service IN (0, 1)),
    multiple_lines     TEXT    NOT NULL CHECK (multiple_lines IN ('Yes', 'No', 'No phone service')),
    internet_service   TEXT    NOT NULL CHECK (internet_service IN ('DSL', 'Fiber optic', 'No')),
    online_security    TEXT    NOT NULL CHECK (online_security    IN ('Yes', 'No', 'No internet service')),
    online_backup      TEXT    NOT NULL CHECK (online_backup      IN ('Yes', 'No', 'No internet service')),
    device_protection  TEXT    NOT NULL CHECK (device_protection  IN ('Yes', 'No', 'No internet service')),
    tech_support       TEXT    NOT NULL CHECK (tech_support       IN ('Yes', 'No', 'No internet service')),
    streaming_tv       TEXT    NOT NULL CHECK (streaming_tv       IN ('Yes', 'No', 'No internet service')),
    streaming_movies   TEXT    NOT NULL CHECK (streaming_movies   IN ('Yes', 'No', 'No internet service')),
    contract           TEXT    NOT NULL CHECK (contract IN ('Month-to-month', 'One year', 'Two year')),
    paperless_billing  INTEGER NOT NULL CHECK (paperless_billing IN (0, 1)),
    payment_method     TEXT    NOT NULL,
    monthly_charges    REAL    NOT NULL CHECK (monthly_charges >= 0),
    total_charges      REAL    CHECK (total_charges IS NULL OR total_charges >= 0),
    churn              INTEGER NOT NULL CHECK (churn IN (0, 1))
);
