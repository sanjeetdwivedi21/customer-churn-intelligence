"""Shared test data: four tiny customers in the same format as the cleaned database table."""
import pandas as pd
import pytest


@pytest.fixture
def sample_customers() -> pd.DataFrame:
    base = dict(gender="Male", senior_citizen=0, has_partner=0, has_dependents=0,
                phone_service=1, multiple_lines="No", paperless_billing=1, total_charges=100.0)
    rows = [
        # High-risk profile: new, fiber, month-to-month, electronic check
        dict(base, customer_id="T-001", tenure_months=3, internet_service="Fiber optic",
             online_security="No", online_backup="No", device_protection="No", tech_support="No",
             streaming_tv="Yes", streaming_movies="Yes", contract="Month-to-month",
             payment_method="Electronic check", monthly_charges=95.0, churn=1),
        # Loyal profile: long tenure, two-year contract, autopay, protective add-ons
        dict(base, customer_id="T-002", tenure_months=60, internet_service="DSL",
             online_security="Yes", online_backup="Yes", device_protection="Yes", tech_support="Yes",
             streaming_tv="No", streaming_movies="No", contract="Two year",
             payment_method="Credit card (automatic)", monthly_charges=70.0, churn=0),
        # No internet and no phone: add-ons must become 0
        dict(base, customer_id="T-003", tenure_months=24, internet_service="No",
             phone_service=0, multiple_lines="No phone service",
             online_security="No internet service", online_backup="No internet service",
             device_protection="No internet service", tech_support="No internet service",
             streaming_tv="No internet service", streaming_movies="No internet service",
             contract="One year", payment_method="Mailed check", monthly_charges=20.0, churn=0),
        # Brand-new customer with no bill yet
        dict(base, customer_id="T-004", tenure_months=0, internet_service="DSL",
             online_security="No", online_backup="Yes", device_protection="No", tech_support="No",
             streaming_tv="No", streaming_movies="No", contract="Month-to-month",
             payment_method="Bank transfer (automatic)", monthly_charges=50.0,
             total_charges=None, churn=0),
    ]
    return pd.DataFrame(rows)