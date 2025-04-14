import pandas as pd
import numpy as np
import os
from datetime import timedelta

# Defensive programming: Ensure directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to generate synthetic weekly data
def generate_synthetic_data():
    countries = ["Rwanda", "Burundi", "Kenya", "Uganda", "Tanzania", "DRC"]
    start_date = pd.to_datetime("2022-01-01")
    weeks = 104  # 2 years
    data = []

    for country in countries:
        date = start_date
        for _ in range(weeks):
            temperature = np.random.normal(25, 3)  # Avg temp
            rainfall = np.random.gamma(2, 15)
            humidity = np.random.uniform(50, 90)
            sanitation_score = np.random.uniform(0, 1)
            resistance_gene_score = np.random.uniform(0, 1)

            git_infection_cases = int(
                np.random.poisson(50 + 10 * (1 - sanitation_score))
            )

            # AMR cases depend on infections & resistance gene score
            amr_rate = 0.1 + 0.6 * resistance_gene_score
            amr_cases = int(git_infection_cases * amr_rate)

            data.append({
                "week": date,
                "country": country,
                "temperature": round(temperature, 2),
                "rainfall": round(rainfall, 2),
                "humidity": round(humidity, 2),
                "sanitation_score": round(sanitation_score, 2),
                "resistance_gene_score": round(resistance_gene_score, 2),
                "git_infection_cases": git_infection_cases,
                "amr_cases": amr_cases
            })

            date += timedelta(weeks=1)

    return pd.DataFrame(data)

# Main function
def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data()

    ensure_dir("data")
    file_path = "data/amr_synthetic_data.csv"
    df.to_csv(file_path, index=False)
    print(f"✅ Data saved to {file_path}")

if __name__ == "__main__":
    main()

