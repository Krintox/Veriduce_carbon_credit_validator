from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Load trained model
MODEL_PATH = 'carbon_credit_validator.pkl'
model_data = joblib.load(MODEL_PATH)

# Extract components
validator = model_data['ensemble']
feature_transformer = model_data['feature_transformer']
categorical_encoder = model_data['categorical_encoder']

def preprocess_input(data):
    """Preprocess input data before feeding it into the model."""
    df = pd.DataFrame([data])
    
    # Feature Engineering
    df['project_duration'] = (pd.to_datetime(df['end_date']) - pd.to_datetime(df['start_date'])).dt.days
    df['emission_reduction_rate'] = df['emission_reduction'] / df['project_duration']
    df['success_risk_interaction'] = (df['successful_projects'] / df['total_projects']) * df['risk_assessment']
    
    # Encode categorical features
    categorical_features = ['project_size']
    numeric_features = df.select_dtypes(include='number').columns.tolist()
    
    encoded_cats = categorical_encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_cats, columns=categorical_encoder.get_feature_names_out(categorical_features))
    
    df = pd.concat([df[numeric_features], encoded_df], axis=1)
    
    # Apply feature transformation
    transformed_features = feature_transformer.transform(df)
    return pd.DataFrame(transformed_features, columns=df.columns)

@app.route('/')
def home():
    return jsonify({'message': 'API is up and running'})

@app.route('/validate', methods=['POST'])
def validate():
    """API endpoint to validate a carbon credit project."""
    try:
        data = request.json
        processed_data = preprocess_input(data)
        score = validator.predict_proba(processed_data)[:, 1]
        return jsonify({'validation_score': float(score[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9654, debug=True)
