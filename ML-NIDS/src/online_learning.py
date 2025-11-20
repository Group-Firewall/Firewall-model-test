"""
Online Learning Module
Implements incremental learning for continuous model updates
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
import joblib
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from river import tree, ensemble, linear_model
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("River library not available. Only SGD will be used.")


class OnlineLearner:
    """
    Online learning for incremental model updates
    Supports multiple algorithms: SGD, Hoeffding Tree, Adaptive Random Forest
    """
    
    def __init__(self, algorithm: str = 'sgd', learning_rate: float = 0.01):
        """
        Initialize online learner
        
        Args:
            algorithm: 'sgd', 'hoeffding_tree', or 'adaptive_rf'
            learning_rate: Learning rate for SGD
        """
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.model = None
        self.is_fitted = False
        self.n_samples_seen = 0
        self.performance_history = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the online learning model"""
        if self.algorithm == 'sgd':
            self.model = SGDClassifier(
                loss='log_loss',  # Logistic regression
                learning_rate='adaptive',
                eta0=self.learning_rate,
                random_state=42,
                class_weight='balanced'
            )
        elif self.algorithm == 'hoeffding_tree' and RIVER_AVAILABLE:
            try:
                self.model = tree.HoeffdingTreeClassifier()
            except:
                print("River library not available. Using SGD as fallback.")
                self.algorithm = 'sgd'
                self.model = SGDClassifier(
                    loss='log_loss',
                    learning_rate='adaptive',
                    eta0=self.learning_rate,
                    random_state=42,
                    class_weight='balanced'
                )
        elif self.algorithm == 'adaptive_rf' and RIVER_AVAILABLE:
            try:
                self.model = ensemble.AdaptiveRandomForestClassifier(n_models=10)
            except:
                print("River library not available. Using SGD as fallback.")
                self.algorithm = 'sgd'
                self.model = SGDClassifier(
                    loss='log_loss',
                    learning_rate='adaptive',
                    eta0=self.learning_rate,
                    random_state=42,
                    class_weight='balanced'
                )
        else:
            if not RIVER_AVAILABLE:
                print("River library not available. Using SGD.")
                self.algorithm = 'sgd'
                self.model = SGDClassifier(
                    loss='log_loss',
                    learning_rate='adaptive',
                    eta0=self.learning_rate,
                    random_state=42,
                    class_weight='balanced'
                )
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, 
                   classes: Optional[np.ndarray] = None) -> 'OnlineLearner':
        """
        Incrementally update the model with new data
        
        Args:
            X: Feature matrix
            y: Target labels
            classes: Unique class labels (required for first fit)
        """
        if self.algorithm == 'sgd':
            if not self.is_fitted:
                if classes is None:
                    classes = np.unique(y)
                self.model.partial_fit(X, y, classes=classes)
                self.is_fitted = True
            else:
                self.model.partial_fit(X, y)
        elif self.algorithm in ['hoeffding_tree', 'adaptive_rf'] and RIVER_AVAILABLE:
            # River models process one sample at a time
            for i in range(len(X)):
                x_dict = {f'feature_{j}': float(X[i, j]) for j in range(X.shape[1])}
                if not self.is_fitted:
                    self.model.learn_one(x_dict, int(y[i]))
                else:
                    self.model.learn_one(x_dict, int(y[i]))
                self.is_fitted = True
        
        self.n_samples_seen += len(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.algorithm == 'sgd':
            return self.model.predict(X)
        else:
            # River models
            predictions = []
            for i in range(len(X)):
                x_dict = {f'feature_{j}': float(X[i, j]) for j in range(X.shape[1])}
                pred = self.model.predict_one(x_dict)
                predictions.append(pred if pred is not None else 0)
            return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.algorithm == 'sgd':
            return self.model.predict_proba(X)
        else:
            # River models - approximate probabilities
            predictions = []
            for i in range(len(X)):
                x_dict = {f'feature_{j}': float(X[i, j]) for j in range(X.shape[1])}
                try:
                    proba = self.model.predict_proba_one(x_dict)
                    if proba:
                        pred = [proba.get(0, 0.0), proba.get(1, 0.0)]
                    else:
                        pred = [0.5, 0.5]
                except:
                    pred = [0.5, 0.5]
                predictions.append(pred)
            return np.array(predictions)
    
    def update_and_evaluate(self, X_new: np.ndarray, y_new: np.ndarray,
                           X_test: Optional[np.ndarray] = None,
                           y_test: Optional[np.ndarray] = None) -> dict:
        """
        Update model with new data and evaluate performance
        
        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Update model
        classes = np.unique(y_new) if not self.is_fitted else None
        self.partial_fit(X_new, y_new, classes=classes)
        
        # Evaluate if test set provided
        metrics = {}
        if X_test is not None and y_test is not None:
            y_pred = self.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'n_samples_seen': self.n_samples_seen
            }
            self.performance_history.append(metrics)
        
        return metrics
    
    def get_performance_history(self) -> pd.DataFrame:
        """Get performance history as DataFrame"""
        if not self.performance_history:
            return pd.DataFrame()
        return pd.DataFrame(self.performance_history)
    
    def save(self, filepath: str):
        """Save online learner"""
        joblib.dump({
            'algorithm': self.algorithm,
            'learning_rate': self.learning_rate,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'n_samples_seen': self.n_samples_seen,
            'performance_history': self.performance_history
        }, filepath)
    
    def load(self, filepath: str):
        """Load online learner"""
        data = joblib.load(filepath)
        self.algorithm = data['algorithm']
        self.learning_rate = data['learning_rate']
        self.model = data['model']
        self.is_fitted = data['is_fitted']
        self.n_samples_seen = data['n_samples_seen']
        self.performance_history = data.get('performance_history', [])
        return self


class IncrementalModelUpdater:
    """
    Manages incremental updates to the hybrid detection system
    """
    
    def __init__(self, hybrid_detector, preprocessor):
        self.hybrid_detector = hybrid_detector
        self.preprocessor = preprocessor
        self.online_learner = OnlineLearner(algorithm='sgd')
        self.update_buffer = []
        self.buffer_size = 100
    
    def add_sample(self, row: pd.Series, label: int, X_processed: Optional[np.ndarray] = None):
        """Add a new labeled sample to the update buffer"""
        self.update_buffer.append({
            'row': row,
            'label': label,
            'X_processed': X_processed
        })
        
        # Update when buffer is full
        if len(self.update_buffer) >= self.buffer_size:
            self.update_models()
    
    def update_models(self):
        """Update online learning model with buffered samples"""
        if not self.update_buffer:
            return
        
        # Prepare data
        X_batch = []
        y_batch = []
        
        for sample in self.update_buffer:
            if sample['X_processed'] is not None:
                X_batch.append(sample['X_processed'])
                y_batch.append(sample['label'])
        
        if X_batch:
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            
            # Update online learner
            self.online_learner.partial_fit(X_batch, y_batch)
        
        # Clear buffer
        self.update_buffer = []
    
    def get_online_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from online learner"""
        if not self.online_learner.is_fitted:
            return np.zeros(len(X))
        return self.online_learner.predict(X)
