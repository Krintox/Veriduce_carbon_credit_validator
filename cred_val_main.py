import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from typing import Dict, List, Tuple, Optional, Union
import shap
import optuna
import logging
import json
import joblib

class AdvancedCreditValidator:
    """
    Sophisticated credit validation system using ensemble learning,
    advanced feature engineering, and explainable AI.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        n_trials: int = 100,
        calibrate_probabilities: bool = True
    ):
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_trials = n_trials
        self.calibrate_probabilities = calibrate_probabilities
        
        # Initialize models
        self.models = {}
        self.feature_transformers = {}
        self.feature_importance = {}
        self.shap_values = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure detailed logging."""
        logger = logging.getLogger('CreditValidator')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def _create_feature_transformers(self) -> Dict:
        """Creates dictionary of feature transformers."""
        return {
            'robust_scaler': RobustScaler(),
            'quantile_transformer': QuantileTransformer(
                output_distribution='normal',
                n_quantiles=1000,
                random_state=self.random_state
            )
        }
        
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering pipeline with domain-specific features
        for carbon credit validation.
        """
        features = pd.DataFrame()
        
        # Project metrics
        features['project_duration'] = (
            pd.to_datetime(data['end_date']) - 
            pd.to_datetime(data['start_date'])
        ).dt.days
        features['emission_reduction_rate'] = (
            data['emission_reduction'] / features['project_duration']
        )
        features['project_scale'] = pd.to_numeric(features['project_scale'], errors='coerce')
        features['project_duration'] = pd.to_numeric(features['project_duration'], errors='coerce')
        
        if features[['project_scale', 'project_duration']].isnull().any().any():
            print("Warning: NaN values detected after conversion. Check the data source.")

        features['project_scale'] = data['project_size']
        
        # Historical performance features
        features['historical_success_rate'] = (
            data['successful_projects'] / data['total_projects']
        )
        features['verification_score'] = data['verification_score']
        
        # Risk metrics
        features['risk_score'] = data['risk_assessment']
        features['compliance_rating'] = data['compliance_score']
        
        # Interaction features
        features['success_risk_interaction'] = (
            features['historical_success_rate'] * features['risk_score']
        )
        features['scale_duration_ratio'] = (
            features['project_scale'] / features['project_duration']
        )
        
        # Statistical features
        features['emission_reduction_per_size'] = (
            data['emission_reduction'] / features['project_scale']
        )
        
        # Normalize features
        for col in features.columns:
            if features[col].dtype in ['int64', 'float64']:
                features[col] = self.feature_transformers['robust_scaler'].fit_transform(
                    features[[col]]
                )
                
        return features
        
    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Objective function for hyperparameter optimization using Optuna.
        """
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(
                **param,
                objective='binary:logistic',
                random_state=self.random_state
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            scores.append(score)
            
        return np.mean(scores)
        
    def train(
        self,
        data: pd.DataFrame,
        target: str,
        features: List[str],
        categorical_features: List[str] = None
    ) -> Dict:
        """
        Trains the ensemble credit validation system with optimized models.
        """
        self.logger.info("Starting credit validation system training...")
        
        # Initialize feature transformers
        self.feature_transformers = self._create_feature_transformers()
        
        # Engineer features
        X = self._engineer_features(data[features])
        y = data[target]
        
        # Handle categorical features
        if categorical_features:
            X = pd.get_dummies(X, columns=categorical_features)
        
        # Optimize XGBoost hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials
        )
        
        # Train ensemble models
        self.models['xgboost'] = xgb.XGBClassifier(
            **study.best_params,
            objective='binary:logistic',
            random_state=self.random_state
        )
        
        self.models['lightgbm'] = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            random_state=self.random_state
        )
        
        self.models['logistic'] = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state
        )
        
        # Create voting classifier
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgb', self.models['xgboost']),
                ('lgb', self.models['lightgbm']),
                ('log', self.models['logistic'])
            ],
            voting='soft',
            weights=[0.5, 0.3, 0.2]
        )
        
        # Train ensemble
        self.ensemble.fit(X, y)
        
        # Calculate feature importance
        self._calculate_feature_importance(X)
        
        # Generate SHAP values for explainability
        self._generate_shap_values(X)
        
        self.logger.info("Training completed successfully")
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'feature_importance': self.feature_importance
        }
        
    def _calculate_feature_importance(self, X: pd.DataFrame) -> None:
        """Calculates feature importance using multiple methods."""
        # XGBoost feature importance
        xgb_importance = dict(zip(
            X.columns,
            self.models['xgboost'].feature_importances_
        ))
        
        # LightGBM feature importance
        lgb_importance = dict(zip(
            X.columns,
            self.models['lightgbm'].feature_importances_
        ))
        
        # Logistic regression coefficients
        log_importance = dict(zip(
            X.columns,
            np.abs(self.models['logistic'].coef_[0])
        ))
        
        # Combine importance scores
        self.feature_importance = {
            'xgboost': xgb_importance,
            'lightgbm': lgb_importance,
            'logistic': log_importance
        }
        
    def _generate_shap_values(self, X: pd.DataFrame) -> None:
        """Generates SHAP values for model explainability."""
        explainer = shap.TreeExplainer(self.models['xgboost'])
        self.shap_values = explainer.shap_values(X)
        
    def validate_credit(
        self,
        project_data: pd.DataFrame,
        return_details: bool = False
    ) -> Union[Dict[str, float], float]:
        """
        Validates carbon credit claims with confidence scores and explanations.
        """
        # Engineer features
        X = self._engineer_features(project_data)
        
        # Get predictions from ensemble
        probabilities = self.ensemble.predict_proba(X)[:, 1]
        
        # Get individual model predictions
        predictions = {
            'xgboost': self.models['xgboost'].predict_proba(X)[:, 1],
            'lightgbm': self.models['lightgbm'].predict_proba(X)[:, 1],
            'logistic': self.models['logistic'].predict_proba(X)[:, 1]
        }
        
        if return_details:
            # Calculate SHAP values for explanation
            explainer = shap.TreeExplainer(self.models['xgboost'])
            shap_values = explainer.shap_values(X)
            
            # Get feature contributions
            feature_contributions = dict(zip(
                X.columns,
                shap_values[0]
            ))
            
            return {
                'validation_score': probabilities[0],
                'model_scores': predictions,
                'feature_contributions': feature_contributions,
                'confidence_score': 1 - np.std(list(predictions.values())),
                'feature_importance': self.feature_importance
            }
            
        return probabilities[0]
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation with multiple metrics.
        """
        # Get predictions
        y_pred = self.ensemble.predict(X_test)
        y_pred_proba = self.ensemble.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """Saves all components of the credit validation system."""
        # Save models
        joblib.dump(self.ensemble, path + '_ensemble.pkl')
        joblib.dump(self.feature_transformers, path + '_transformers.pkl')
        joblib.dump(self.feature_importance, path + '_feature_importance.pkl')
        
        if self.shap_values is not None:
            np.save(path + '_shap_values.npy', self.shap_values)
        
        self.logger.info(f"Models saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Loads all components of the credit validation system."""
        self.ensemble = joblib.load(path + '_ensemble.pkl')
        self.feature_transformers = joblib.load(path + '_transformers.pkl')
        self.feature_importance = joblib.load(path + '_feature_importance.pkl')
        
        try:
            self.shap_values = np.load(path + '_shap_values.npy', allow_pickle=True)
        except FileNotFoundError:
            self.logger.warning("SHAP values file not found")
        
        self.logger.info(f"Models loaded from {path}")