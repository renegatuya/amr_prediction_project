import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta

# random seed for reproducibility
np.random.seed(42)

# Define countries and districts
regions = {
    "Rwanda": ["Kigali", "Huye", "Rubavu"],
    "Uganda": ["Kampala", "Gulu", "Mbale"],
    "Kenya": ["Nairobi", "Kisumu", "Mombasa"],
    "Tanzania": ["Dar es Salaam", "Arusha", "Mwanza"],
    "Burundi": ["Bujumbura", "Ngozi", "Gitega"],
    "DRC": ["Kinshasa", "Goma", "Lubumbashi"]
}

pathogens = ["Cholera", "E. coli", "Salmonella", "Shigella", "Rotavirus"]
start_date = pd.to_datetime("2023-01-01")
weeks = pd.date_range(start=start_date, periods=52, freq='W')

# Simulate data
data = []

for country, districts in regions.items():
    for district in districts:
        for week in weeks:
            for pathogen in pathogens:
                # Simulate seasonal trend
                week_of_year = week.isocalendar().week
                seasonal_effect = np.sin(2 * np.pi * week_of_year / 52)
                lam = max(0, 5 + 15 * seasonal_effect)  # Ensure lambda is non-negative
                base_cases = np.random.poisson(lam)

                # Add outbreak 
                outbreak = np.random.binomial(1, 0.05)
                cases = base_cases + (np.random.randint(20, 50) if outbreak else 0)

                # Simulate AMR gene abundance from cases
                amr_abundance = round(np.random.normal(loc=0.2 * cases, scale=5), 2)
                amr_abundance = max(0, amr_abundance)

                data.append({
                    "week": week,
                    "country": country,
                    "district": district,
                    "pathogen": pathogen,
                    "cases": cases,
                    "amr_abundance": amr_abundance
                })

# Create DataFrame
df = pd.DataFrame(data)

# Save data
output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "synthetic_git_amr_data.csv"), index=False)

# Plot example: total cases per week
total_cases_weekly = df.groupby("week")["cases"].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=total_cases_weekly, x="week", y="cases", color="blue")
plt.title("Total GIT Infection Cases Over Time")
plt.xlabel("Week")
plt.ylabel("Cases")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "total_cases_over_time.png"))
plt.close()

# Plot heatmap of correlation (by country)
avg_amr = df.groupby("country")["amr_abundance"].mean()
avg_cases = df.groupby("country")["cases"].mean()
summary_df = pd.DataFrame({"cases": avg_cases, "amr_abundance": avg_amr})
sns.heatmap(summary_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation between AMR Abundance and Cases by Country")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

print("Synthetic data and visuals generated and saved to outputs folder.")

