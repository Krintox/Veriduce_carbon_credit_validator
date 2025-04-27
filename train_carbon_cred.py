import pandas as pd
from sklearn.model_selection import train_test_split
from cred_val_main_upd import CreditValidator

# Load dataset
data = pd.read_csv('carbon_credit_dataset.csv')

# Define features and target
FEATURES = [
    'start_date', 'end_date', 'emission_reduction', 'project_size',
    'successful_projects', 'total_projects', 'verification_score',
    'risk_assessment', 'compliance_score'
]
TARGET = 'validated'

# Initialize model
validator = CreditValidator()

# Train model
validator.train(data, TARGET, FEATURES)

# Prepare test data for evaluation
X = validator._engineer_features(data[FEATURES])
y = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate model
metrics = validator.evaluate(X_test, y_test)
print("Model Performance:", metrics)

# Save trained model
MODEL_PATH = 'carbon_credit_validator.pkl'
validator.save_model(MODEL_PATH)

# Load model (optional)
validator.load_model(MODEL_PATH)

# Validate new project data
new_project_data = pd.DataFrame({
    'start_date': ['2023-01-01'],
    'end_date': ['2025-01-01'],
    'emission_reduction': [12000],
    'project_size': ['medium'],
    'successful_projects': [10],
    'total_projects': [15],
    'verification_score': [80],
    'risk_assessment': [50],
    'compliance_score': [90]
})

validation_score = validator.validate(new_project_data)
print("Validation Score:", validation_score)
