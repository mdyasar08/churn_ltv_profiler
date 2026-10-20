import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") 

INPUT_FILE = os.path.join(DATA_DIR, "raw_customers.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "cohort_summary.csv")


def run_churn_ltv_analysis():
    print("=" * 60)
    print("CUSTOMER CHURN & LIFETIME VALUE (LTV) PROFILER")
    print("=" * 60)

    # --------------------------------------------------
    # 2. LOAD DATA
    # --------------------------------------------------

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Make sure raw_customers.csv is inside the data folder."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"\n[1] Data loaded successfully: {df.shape}")
    print("\nAvailable columns:")
    print(df.columns.tolist()) # Converts the column index to a standard list for cleaner reading

    # --------------------------------------------------
    # 3. DATA CLEANING
    # --------------------------------------------------

    df["signup_date"] = pd.to_datetime(
        df["signup_date"], errors="coerce"
    )

    df["monthly_spend"] = pd.to_numeric(
        df["monthly_spend"], errors="coerce"
    )

    df["estimated_monthly_cost"] = pd.to_numeric(
        df["estimated_monthly_cost"], errors="coerce"
    )

    df["tenure_months"] = pd.to_numeric(
        df["tenure_months"], errors="coerce"
    )

    df["churned"] = pd.to_numeric(
        df["churned"], errors="coerce"
    )

    # Fill missing numerical values
    df["monthly_spend"] = df["monthly_spend"].fillna(
        df["monthly_spend"].median() 
    )

    df["estimated_monthly_cost"] = df[
        "estimated_monthly_cost"
    ].fillna(df["estimated_monthly_cost"].median())

    df["tenure_months"] = df["tenure_months"].fillna(
        df["tenure_months"].median()
    )

    num_cols = ["monthly_spend", "estimated_monthly_cost", "tenure_months"]

    # Fills all 3 columns using their respective medians in one go
    # for col in num_cols:
    #     df[col] = df[col].fillna(df[col].median())

    # Remove rows missing essential identifiers or dates
    df = df.dropna(subset=["customer_id", "signup_date"])

    # --------------------------------------------------
    # 4. CHURN ANALYSIS
    # --------------------------------------------------

    # The supplied dataset already contains churn labels:
    # 1 = Churned, 0 = Not churned.
    df["is_churned"] = df["churned"].fillna(0).astype(int) 

    churn_count = int(df["is_churned"].sum())
    active_count = len(df) - churn_count  #Subtracts the churned count from the total number of rows (len(df)) to find how many active customers remain.
    churn_rate = df["is_churned"].mean()

    print("\n[2] CHURN BREAKDOWN")
    print(f"Total customers : {len(df)}")
    print(f"Active customers: {active_count}")
    print(f"Churned customers: {churn_count}")
    print(f"Overall churn rate: {churn_rate * 100:.2f}%")

    # --------------------------------------------------
    # 5. CUSTOMER VALUE SEGMENTATION
    # --------------------------------------------------

    # Divide spending into up to three value tiers.
    # rank(method='first') helps qcut when values repeat.
    tier_labels = ["Bronze", "Silver", "Gold"]

    df["value_tier"] = pd.qcut(
        df["monthly_spend"].rank(method="first"),
        q=3,
        labels=tier_labels
    )

    tier_summary = df.groupby(
        "value_tier", observed=False
    ).agg(
        total_customers=("customer_id", "count"),
        avg_monthly_spend=("monthly_spend", "mean"),
        churn_rate=("is_churned", "mean")
    )

    tier_summary["churn_rate"] = (
        tier_summary["churn_rate"] * 100
    ).round(2)

    tier_summary["avg_monthly_spend"] = (
        tier_summary["avg_monthly_spend"].round(2)
    )

    print("\n[3] CUSTOMER VALUE TIERS")
    print(tier_summary)

    # --------------------------------------------------
    # 6. COHORT ANALYSIS
    # --------------------------------------------------

    df["signup_cohort"] = (
        df["signup_date"].dt.to_period("M").astype(str)
    )

    # Estimated monthly profit per customer
    df["monthly_profit_estimate"] = (
        df["monthly_spend"] - df["estimated_monthly_cost"]
    )

    cohort_summary = df.groupby(
        "signup_cohort", as_index=False
    ).agg(
        cohort_size=("customer_id", "count"),
        churned_customers=("is_churned", "sum"),
        churn_rate=("is_churned", "mean"),
        total_revenue=("monthly_spend", "sum"),
        avg_monthly_spend=("monthly_spend", "mean"),
        avg_tenure_months=("tenure_months", "mean"),
        estimated_monthly_profit=(
            "monthly_profit_estimate", "sum"
        )
    )

    # Convert churn rate to a percentage
    cohort_summary["churn_rate"] = (
        cohort_summary["churn_rate"] * 100
    ).round(2)

    numeric_columns = [
        "total_revenue",
        "avg_monthly_spend",
        "avg_tenure_months",
        "estimated_monthly_profit"
    ]

    cohort_summary[numeric_columns] = (
        cohort_summary[numeric_columns].round(2)
    )

    print("\n[4] COHORT SUMMARY")
    print(cohort_summary.to_string(index=False))

    # --------------------------------------------------
    # 7. ESTIMATED CUSTOMER LIFETIME VALUE (LTV)
    # --------------------------------------------------

    # Simplified revenue-based LTV proxy:
    # Average monthly spend per customer / observed churn rate.
    # This is an estimate, not a predictive LTV model.

    avg_monthly_spend = df["monthly_spend"].mean()

    if churn_rate > 0:
        estimated_ltv = avg_monthly_spend / churn_rate
    else:
        estimated_ltv = np.nan

    avg_monthly_profit = df["monthly_profit_estimate"].mean()

    print("\n[5] KEY BUSINESS METRICS")
    print(f"Average monthly spend: ${avg_monthly_spend:.2f}")
    print(f"Average monthly profit: ${avg_monthly_profit:.2f}")
    print(f"Observed churn rate: {churn_rate * 100:.2f}%")

    if pd.notna(estimated_ltv):
        print(f"Estimated revenue-based LTV: ${estimated_ltv:.2f}")
    else:
        print("Estimated LTV unavailable: no churn observed.")

    # --------------------------------------------------
    # 8. EXPORT COHORT REPORT
    # --------------------------------------------------

    cohort_summary.to_csv(OUTPUT_FILE, index=False)

    print("\n[6] REPORT EXPORTED SUCCESSFULLY")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    run_churn_ltv_analysis()
