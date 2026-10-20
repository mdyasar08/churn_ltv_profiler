Customer Churn & Lifetime Value (LTV) Profiler
An automated Python pipeline using pandas and NumPy to analyze customer retention, calculate Lifetime Value (LTV), and perform cohort analysis.

🚀 Key Features
Data Cleaning & Date Handling: Parses timestamps and formats transaction metrics.
Churn Detection: Uses np.where() to flag accounts with no purchases in >90 days.
Customer Segmentation: Divides users into revenue tiers (Gold, Silver, Bronze) using pd.qcut().
Cohort Analysis: Groups retention and revenue metrics by signup month.
Automated Export: Generates and saves a cohort_summary.csv report.

📂 Project Structurechurn_ltv_profiler/ 
├── data/
│   ├── raw_customers.csv       # Input dataset
│   └── cohort_summary.csv      # Exported report
├── churn_analysis.py           # Main executable script
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation

🛠️ Quick StartInstall dependencies:
pip install -r requirements.txt

Run the analysis:
python churn_analysis.py

📊 Core Business Metrics Calculated
Days Since Last Purchase: $\text{Analysis Date} - \text{Last Purchase Date}
Average Order Value (AOV): $\frac{\text{Total Revenue}}{\text{Total Orders}}
Customer Lifetime Value (LTV): $\frac{\text{AOV} \times \text{Avg Purchase Frequency}}{\text{Churn Rate}}


🧰 Tech Stack
Python | pandas | NumPy | os (Path Management)
