import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import optuna
import logging

class CreditValidator:
    def __init__(self, n_folds=5, random_state=42, n_trials=50):
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_trials = n_trials
        self.models = {}
        self.feature_transformer = RobustScaler()
        self.categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.ensemble = None
        self.logger = self._setup_logging()

    def _setup_logging(self):
        logger = logging.getLogger('CreditValidator')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def _engineer_features(self, data):
        """Feature engineering for numerical and categorical variables."""
        data = data.copy()
        data['project_duration'] = (pd.to_datetime(data['end_date']) - pd.to_datetime(data['start_date'])).dt.days
        data['emission_reduction_rate'] = data['emission_reduction'] / data['project_duration']
        data['success_risk_interaction'] = (data['successful_projects'] / data['total_projects']) * data['risk_assessment']
        
        # Handle categorical features
        categorical_features = ['project_size']
        numeric_features = data.select_dtypes(include=np.number).columns.tolist()
        
        if not hasattr(self, 'cat_encoder_fitted') or not self.cat_encoder_fitted:
            self.categorical_encoder.fit(data[categorical_features])
            self.cat_encoder_fitted = True

        encoded_cats = self.categorical_encoder.transform(data[categorical_features])
        encoded_df = pd.DataFrame(encoded_cats, columns=self.categorical_encoder.get_feature_names_out(categorical_features))

        data = pd.concat([data[numeric_features], encoded_df], axis=1)
        return pd.DataFrame(self.feature_transformer.fit_transform(data), columns=data.columns)

    def _objective(self, trial, X, y):
        """Hyperparameter tuning objective function."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in cv.split(X, y):
            model = xgb.XGBClassifier(**params, objective='binary:logistic', random_state=self.random_state)
            model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                      early_stopping_rounds=50, verbose=False)
            scores.append(roc_auc_score(y.iloc[val_idx], model.predict_proba(X.iloc[val_idx])[:, 1]))

        return np.mean(scores)

    def train(self, data, target, features):
        """Train the model using hyperparameter tuning and ensemble learning."""
        X = self._engineer_features(data[features])
        y = data[target]

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)

        self.models = {
            'xgb': xgb.XGBClassifier(**study.best_params, objective='binary:logistic', random_state=self.random_state),
            'lgb': LGBMClassifier(n_estimators=500, learning_rate=0.01, random_state=self.random_state),
            'log': LogisticRegression(max_iter=1000, random_state=self.random_state)
        }

        self.ensemble = VotingClassifier(estimators=[(k, v) for k, v in self.models.items()], voting='soft')
        self.ensemble.fit(X, y)
        self.logger.info("Training completed.")
        return {'best_params': study.best_params, 'best_score': study.best_value}

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if self.ensemble is None:
            raise ValueError("Model is not trained yet.")
        
        y_pred = self.ensemble.predict(X_test)
        y_prob = self.ensemble.predict_proba(X_test)[:, 1]
        return {
            'Accuracy': accuracy_score(y_test, y_pred),
            'ROC_AUC': roc_auc_score(y_test, y_prob)
        }

    def save_model(self, filepath):
        """Save trained model to disk."""
        joblib.dump({
            'ensemble': self.ensemble,
            'feature_transformer': self.feature_transformer,
            'categorical_encoder': self.categorical_encoder
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        self.ensemble = model_data['ensemble']
        self.feature_transformer = model_data['feature_transformer']
        self.categorical_encoder = model_data['categorical_encoder']
        self.logger.info("Model loaded successfully.")

    def validate(self, project_data):
        """Validate new project data and return probability of credit validation."""
        X = self._engineer_features(project_data)
        return self.ensemble.predict_proba(X)[:, 1]
