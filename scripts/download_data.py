import pandas as pd
import numpy as np
import os

# Create data folder
os.makedirs('data/raw', exist_ok=True)

# Create sample churn dataset
np.random.seed(42)
n_samples = 7043

data = {
    'CustomerID': [f'CUST_{i:05d}' for i in range(n_samples)],
    'age': np.random.randint(18, 80, n_samples),
    'tenure': np.random.randint(0, 72, n_samples),
    'MonthlyCharges': np.random.uniform(20, 120, n_samples),
    'TotalCharges': np.random.uniform(100, 8500, n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
    'OnlineBackup': np.random.choice(['Yes', 'No'], n_samples),
    'DeviceProtection': np.random.choice(['Yes', 'No'], n_samples),
    'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No'], n_samples),
    'StreamingMovies': np.random.choice(['Yes', 'No'], n_samples),
    'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])  # 27% churn rate
}

df = pd.DataFrame(data)

# Save to CSV
filepath = 'data/raw/churn_data.csv'
df.to_csv(filepath, index=False)

print(f"✓ Dataset created: {filepath}")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nChurn distribution:")
print(df['Churn'].value_counts())