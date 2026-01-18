# %% [markdown]
# # Customer Lifetime Value Cohort Analysis
# ## Business Analyst Project - Ready.io Interview Task
# 
# **Technologies:** Python, SQL (SQLite), Data Visualization
# **Dataset:** Synthetic Retail Transaction Data
# **Duration:** 4 days project (compressed in this notebook)

# %% [markdown]
# ## 1. DATA GENERATION & CLEANING

# %%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns

print(" Libraries imported successfully")

# %%
# Generate synthetic transaction data
np.random.seed(42)
n_transactions = 10000
customer_ids = np.random.randint(1000, 1200, n_transactions)

# Create dates over 2-year period
dates = []
for i in range(n_transactions):
    date = datetime(2022, 1, 1) + timedelta(days=np.random.randint(0, 730))
    dates.append(date)

# Create transaction amounts
amounts = np.round(np.random.exponential(50, n_transactions) + 10, 2)

# Create DataFrame
df = pd.DataFrame({
    'CustomerID': customer_ids,
    'InvoiceDate': dates,
    'Amount': amounts
})

print(f" Data Generated: {len(df)} transactions, {df['CustomerID'].nunique()} unique customers")

# %%
# Calculate cohorts
df['FirstPurchase'] = df.groupby('CustomerID')['InvoiceDate'].transform('min')
df['Cohort'] = df['FirstPurchase'].dt.strftime('%Y-%m')
df['MonthsSinceFirst'] = (df['InvoiceDate'].dt.year - df['FirstPurchase'].dt.year) * 12
df['MonthsSinceFirst'] += (df['InvoiceDate'].dt.month - df['FirstPurchase'].dt.month)

# Filter to first 12 months for analysis
df_analysis = df[df['MonthsSinceFirst'] <= 12]

# Display sample
display(df_analysis.head())

# %% [markdown]
# ## 2. COHORT ANALYSIS IN PYTHON

# %%
# Calculate retention and revenue metrics
cohort_analysis = df_analysis.groupby(['Cohort', 'MonthsSinceFirst']).agg(
    ActiveCustomers=('CustomerID', 'nunique'),
    Revenue=('Amount', 'sum')
).reset_index()

# Calculate cumulative LTV
cohort_analysis['CumulativeLTV'] = cohort_analysis.groupby('Cohort')['Revenue'].cumsum()

# Pivot for heatmap visualization
retention_pivot = cohort_analysis.pivot_table(
    index='Cohort', 
    columns='MonthsSinceFirst', 
    values='ActiveCustomers',
    fill_value=0
)

# Calculate retention rates
cohort_size = retention_pivot.iloc[:, 0]
retention_rate = retention_pivot.divide(cohort_size, axis=0)

# Display metrics
print(" Cohort Analysis Summary:")
print(f"   • Total Cohorts: {cohort_analysis['Cohort'].nunique()}")
print(f"   • Time Period: {df_analysis['InvoiceDate'].min().date()} to {df_analysis['InvoiceDate'].max().date()}")
print(f"   • Total Revenue: ${cohort_analysis['Revenue'].sum():,.2f}")

display(cohort_analysis.head(10))

# %% [markdown]
# ## 3. SQL DATABASE CREATION & QUERIES

# %%
# Create SQLite database in memory
conn = sqlite3.connect(':memory:')

# Load data into SQL database
df_analysis.to_sql('transactions', conn, index=False, if_exists='replace')
print(" SQL Database created with transactions table")

# %%
# SQL Query 1: Cohort Summary
query1 = """
SELECT 
    Cohort,
    COUNT(DISTINCT CustomerID) as CohortSize,
    SUM(Amount) as TotalRevenue,
    AVG(Amount) as AvgTransactionValue
FROM transactions
WHERE MonthsSinceFirst = 0
GROUP BY Cohort
ORDER BY Cohort;
"""

cohort_summary = pd.read_sql_query(query1, conn)
print(" Cohort Size & Revenue:")
display(cohort_summary)

# %%
# SQL Query 2: Monthly Retention
query2 = """
SELECT 
    Cohort,
    MonthsSinceFirst,
    COUNT(DISTINCT CustomerID) as ActiveCustomers,
    SUM(Amount) as MonthlyRevenue,
    SUM(SUM(Amount)) OVER (PARTITION BY Cohort ORDER BY MonthsSinceFirst) as CumulativeLTV
FROM transactions
WHERE MonthsSinceFirst BETWEEN 0 AND 12
GROUP BY Cohort, MonthsSinceFirst
ORDER BY Cohort, MonthsSinceFirst;
"""

monthly_retention = pd.read_sql_query(query2, conn)
print(" Monthly Retention & LTV:")
display(monthly_retention.head(15))

# %%
# SQL Query 3: Best Performing Cohorts
query3 = """
SELECT 
    Cohort,
    CohortSize,
    TotalRevenue,
    ROUND(TotalRevenue * 1.0 / CohortSize, 2) as LTVPerCustomer,
    ROUND(M12_Customers * 100.0 / CohortSize, 2) as Month12RetentionRate
FROM (
    SELECT 
        Cohort,
        COUNT(DISTINCT CustomerID) as CohortSize,
        SUM(Amount) as TotalRevenue,
        SUM(CASE WHEN MonthsSinceFirst = 12 THEN 1 ELSE 0 END) as M12_Customers
    FROM transactions
    GROUP BY Cohort
) sub
ORDER BY LTVPerCustomer DESC
LIMIT 5;
"""

best_cohorts = pd.read_sql_query(query3, conn)
print(" Top 5 Performing Cohorts:")
display(best_cohorts)

# %% [markdown]
# ## 4. VISUALIZATIONS (Tableau Alternative)

# %%
# Visualization 1: Cohort Retention Heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(retention_rate, annot=True, fmt='.0%', cmap='YlOrRd', linewidths=0.5, linecolor='gray')
plt.title('Cohort Retention Heatmap (% of Original Customers Active)', fontsize=16, pad=20)
plt.xlabel('Months Since First Purchase', fontsize=12)
plt.ylabel('Cohort Month', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Visualization 2: LTV Growth by Cohort
plt.figure(figsize=(14, 8))

# Prepare data for line chart
ltv_pivot = monthly_retention.pivot_table(
    index='MonthsSinceFirst', 
    columns='Cohort', 
    values='CumulativeLTV',
    fill_value=0
)

# Plot top 5 cohorts for clarity
top_cohorts = best_cohorts['Cohort'].head(5).tolist()
for cohort in top_cohorts:
    if cohort in ltv_pivot.columns:
        plt.plot(ltv_pivot.index, ltv_pivot[cohort], marker='o', linewidth=2, label=cohort)

plt.title('Customer Lifetime Value Growth (Top 5 Cohorts)', fontsize=16, pad=20)
plt.xlabel('Months Since First Purchase', fontsize=12)
plt.ylabel('Cumulative LTV ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Cohort', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
# Visualization 3: Cohort Size vs LTV Scatter
plt.figure(figsize=(12, 6))
plt.scatter(cohort_summary['CohortSize'], cohort_summary['TotalRevenue'], 
            s=cohort_summary['AvgTransactionValue']*10, alpha=0.6, edgecolors='black')

for idx, row in cohort_summary.iterrows():
    plt.text(row['CohortSize']+1, row['TotalRevenue']+50, row['Cohort'], fontsize=9)

plt.title('Cohort Size vs Total Revenue', fontsize=16, pad=20)
plt.xlabel('Cohort Size (Number of Customers)', fontsize=12)
plt.ylabel('Total Revenue ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. BUSINESS INSIGHTS & RECOMMENDATIONS

# %%
# Calculate key business metrics
avg_ltv = cohort_summary['TotalRevenue'].mean() / cohort_summary['CohortSize'].mean()
avg_3mo_retention = retention_rate[3].mean() * 100
best_cohort_ltv = best_cohorts['LTVPerCustomer'].iloc[0]
worst_cohort_ltv = best_cohorts['LTVPerCustomer'].iloc[-1]

insights_html = f"""
<div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border-left:5px solid #007bff;">
<h3> KEY BUSINESS INSIGHTS</h3>

<h4> Financial Performance:</h4>
<ul>
<li><b>Average Customer LTV:</b> ${avg_ltv:,.2f}</li>
<li><b>Best Cohort LTV:</b> ${best_cohort_ltv:,.2f} ({best_cohorts['Cohort'].iloc[0]})</li>
<li><b>Worst Cohort LTV:</b> ${worst_cohort_ltv:,.2f} ({best_cohorts['Cohort'].iloc[-1]})</li>
<li><b>Performance Gap:</b> {((best_cohort_ltv-worst_cohort_ltv)/worst_cohort_ltv*100):.1f}% difference</li>
</ul>

<h4> Customer Behavior:</h4>
<ul>
<li><b>Average 3-Month Retention:</b> {avg_3mo_retention:.1f}%</li>
<li><b>Average Cohort Size:</b> {cohort_summary['CohortSize'].mean():.0f} customers</li>
<li><b>Peak Revenue Month:</b> Month {monthly_retention.groupby('MonthsSinceFirst')['MonthlyRevenue'].sum().idxmax()} after acquisition</li>
</ul>

<h4> Recommendations:</h4>
<ol>
<li><b>Focus on High-Performing Cohorts:</b> Replicate acquisition strategies from {best_cohorts['Cohort'].iloc[0]} cohort</li>
<li><b>Improve Retention at Month 3:</b> Launch engagement campaigns before the {avg_3mo_retention:.1f}% retention drop</li>
<li><b>Monitor Cohort Quality:</b> Track LTV trends month-over-month to adjust CAC targets</li>
<li><b>Segment by Value:</b> Implement tiered marketing for high vs low LTV customer segments</li>
</ol>
</div>
"""

display(HTML(insights_html))

# %% [markdown]
# ## 6. EXPORT FILES FOR TABLEAU

# %%
# Export data for Tableau
cohort_analysis.to_csv('cohort_analysis_for_tableau.csv', index=False)
monthly_retention.to_csv('monthly_retention_for_tableau.csv', index=False)
cohort_summary.to_csv('cohort_summary_for_tableau.csv', index=False)

# Create SQL script file
sql_script = """-- Customer Lifetime Value Cohort Analysis SQL Script
-- Ready.io Business Analyst Interview Task

-- 1. CREATE TRANSACTIONS TABLE
CREATE TABLE transactions (
    CustomerID INT,
    InvoiceDate DATETIME,
    Amount DECIMAL(10,2),
    FirstPurchase DATETIME,
    Cohort VARCHAR(7),
    MonthsSinceFirst INT
);

-- 2. COHORT SUMMARY
SELECT 
    Cohort,
    COUNT(DISTINCT CustomerID) as CohortSize,
    SUM(Amount) as TotalRevenue,
    AVG(Amount) as AvgTransactionValue
FROM transactions
WHERE MonthsSinceFirst = 0
GROUP BY Cohort
ORDER BY Cohort;

-- 3. MONTHLY RETENTION & LTV
SELECT 
    Cohort,
    MonthsSinceFirst,
    COUNT(DISTINCT CustomerID) as ActiveCustomers,
    SUM(Amount) as MonthlyRevenue,
    SUM(SUM(Amount)) OVER (PARTITION BY Cohort ORDER BY MonthsSinceFirst) as CumulativeLTV
FROM transactions
WHERE MonthsSinceFirst BETWEEN 0 AND 12
GROUP BY Cohort, MonthsSinceFirst
ORDER BY Cohort, MonthsSinceFirst;

-- 4. RETENTION RATE CALCULATION
SELECT 
    Cohort,
    ROUND(SUM(CASE WHEN MonthsSinceFirst = 3 THEN 1 ELSE 0 END) * 100.0 / 
          SUM(CASE WHEN MonthsSinceFirst = 0 THEN 1 ELSE 0 END), 2) as Month3RetentionRate,
    ROUND(SUM(CASE WHEN MonthsSinceFirst = 6 THEN 1 ELSE 0 END) * 100.0 / 
          SUM(CASE WHEN MonthsSinceFirst = 0 THEN 1 ELSE 0 END), 2) as Month6RetentionRate,
    ROUND(SUM(CASE WHEN MonthsSinceFirst = 12 THEN 1 ELSE 0 END) * 100.0 / 
          SUM(CASE WHEN MonthsSinceFirst = 0 THEN 1 ELSE 0 END), 2) as Month12RetentionRate
FROM transactions
GROUP BY Cohort;
"""

with open('cohort_analysis_sql_script.sql', 'w') as f:
    f.write(sql_script)

print(" Files exported for Tableau:")
print("   • cohort_analysis_for_tableau.csv")
print("   • monthly_retention_for_tableau.csv")
print("   • cohort_summary_for_tableau.csv")
print("   • cohort_analysis_sql_script.sql")

# %% [markdown]
# ## 7. PROJECT SUMMARY & DELIVERABLES

# %%
# Generate final summary
deliverables_html = f"""
<div style="background-color:#e8f5e8; padding:20px; border-radius:10px; border:2px solid #28a745;">
<h2 style="color:#28a745;"> PROJECT COMPLETED SUCCESSFULLY</h2>

<h3> DELIVERABLES GENERATED:</h3>
<ol>
<li><b>Data Files:</b>
    <ul>
    <li>cohort_analysis_for_tableau.csv - For Tableau visualization</li>
    <li>monthly_retention_for_tableau.csv - Monthly cohort metrics</li>
    <li>cohort_summary_for_tableau.csv - Cohort-level summary</li>
    </ul>
</li>

<li><b>SQL Script:</b>
    <ul>
    <li>cohort_analysis_sql_script.sql - Complete SQL analysis queries</li>
    </ul>
</li>

<li><b>Visualizations:</b>
    <ul>
    <li>Cohort Retention Heatmap</li>
    <li>LTV Growth Curves</li>
    <li>Cohort Size vs Revenue Analysis</li>
    </ul>
</li>

<li><b>Business Insights:</b>
    <ul>
    <li>Average Customer LTV: ${avg_ltv:,.2f}</li>
    <li>3-Month Retention Rate: {avg_3mo_retention:.1f}%</li>
    <li>Top Cohort: {best_cohorts['Cohort'].iloc[0]} with ${best_cohort_ltv:,.2f} LTV/customer</li>
    <li>4 actionable recommendations provided</li>
    </ul>
</li>
</ol>

<h3> TECHNOLOGIES USED:</h3>
<ul>
<li>Python (pandas, numpy, matplotlib, seaborn) - Data generation & analysis</li>
<li>SQL (SQLite) - Database queries & cohort calculations</li>
<li>Tableau-ready exports - For advanced visualization</li>
</ul>

<h3> KEY METRICS CALCULATED:</h3>
<table border="1" style="border-collapse: collapse; width: 100%;">
<tr style="background-color: #007bff; color: white;">
    <th>Metric</th>
    <th>Value</th>
</tr>
<tr>
    <td>Total Transactions Analyzed</td>
    <td>{len(df_analysis):,}</td>
</tr>
<tr>
    <td>Unique Customers</td>
    <td>{df_analysis['CustomerID'].nunique()}</td>
</tr>
<tr>
    <td>Number of Cohorts</td>
    <td>{cohort_analysis['Cohort'].nunique()}</td>
</tr>
<tr>
    <td>Total Revenue</td>
    <td>${cohort_analysis['Revenue'].sum():,.2f}</td>
</tr>
<tr>
    <td>Average Customer LTV</td>
    <td>${avg_ltv:,.2f}</td>
</tr>
</table>

<h3> NEXT STEPS FOR TABLEAU:</h3>
<ol>
<li>Open Tableau Desktop or Tableau Public</li>
<li>Connect to cohort_analysis_for_tableau.csv</li>
<li>Create:
    <ul>
    <li><b>Heatmap:</b> Cohort (Rows) vs MonthsSinceFirst (Columns) colored by ActiveCustomers</li>
    <li><b>Line Chart:</b> MonthsSinceFirst (X) vs CumulativeLTV (Y) with Cohort as color</li>
    <li><b>Dashboard:</b> Combine both charts with filters</li>
    </ul>
</li>
</ol>
</div>
"""

display(HTML(deliverables_html))
