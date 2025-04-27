import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker for generating realistic data
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 20000

# Generate synthetic data
data = {
    'project_id': [fake.uuid4() for _ in range(n_samples)],
    'start_date': [fake.date_between(start_date='-5y', end_date='today') for _ in range(n_samples)],
    'end_date': [fake.date_between(start_date='today', end_date='+5y') for _ in range(n_samples)],
    'emission_reduction': np.random.normal(loc=10000, scale=2000, size=n_samples).astype(int),
    'project_size': np.random.choice(['small', 'medium', 'large'], size=n_samples, p=[0.5, 0.3, 0.2]),
    'successful_projects': np.random.randint(0, 50, size=n_samples),
    'total_projects': np.random.randint(1, 100, size=n_samples),
    'verification_score': np.random.normal(loc=75, scale=10, size=n_samples).clip(0, 100),
    'risk_assessment': np.random.normal(loc=50, scale=15, size=n_samples).clip(0, 100),
    'compliance_score': np.random.normal(loc=80, scale=10, size=n_samples).clip(0, 100),
    'validated': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])  # Target variable
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate project duration in days
df['project_duration'] = (pd.to_datetime(df['end_date']) - pd.to_datetime(df['start_date'])).dt.days

# Calculate historical success rate
df['historical_success_rate'] = df['successful_projects'] / df['total_projects']

# Add some noise to make the data more realistic
df['emission_reduction'] = df['emission_reduction'] * (1 + np.random.normal(0, 0.1, size=n_samples))
df['verification_score'] = df['verification_score'] * (1 + np.random.normal(0, 0.05, size=n_samples))
df['risk_assessment'] = df['risk_assessment'] * (1 + np.random.normal(0, 0.05, size=n_samples))
df['compliance_score'] = df['compliance_score'] * (1 + np.random.normal(0, 0.05, size=n_samples))

# Ensure all values are within realistic bounds
df['emission_reduction'] = df['emission_reduction'].clip(0, None)
df['verification_score'] = df['verification_score'].clip(0, 100)
df['risk_assessment'] = df['risk_assessment'].clip(0, 100)
df['compliance_score'] = df['compliance_score'].clip(0, 100)
df['historical_success_rate'] = df['historical_success_rate'].clip(0, 1)

# Save to CSV
df.to_csv('carbon_credit_dataset.csv', index=False)

print("Dataset generated and saved to 'carbon_credit_dataset.csv'")