"""
Main Training Script for NIDS
Trains supervised, unsupervised, and hybrid models
Adapted for ML-Network Intrusion Detection System folder
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_path)

try:
    from data_preprocessing import DataPreprocessor
    from supervised_models import SupervisedModelTrainer
    from unsupervised_models import UnsupervisedModelTrainer
    from hybrid_detection import HybridDetector
except ImportError:
    # Fallback if running from different location
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from data_preprocessing import DataPreprocessor
    from supervised_models import SupervisedModelTrainer
    from unsupervised_models import UnsupervisedModelTrainer
    from hybrid_detection import HybridDetector

import joblib


def load_and_split_data(data_path: str, test_size: float = 0.2, val_size: float = 0.2):
    """Load and split data into train, validation, and test sets"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Split into train and test
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['Intrusion'])
    
    # Split train into train and validation
    df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=42, stratify=df_train['Intrusion'])
    
    print(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    print(f"Train intrusion rate: {df_train['Intrusion'].mean():.4f}")
    print(f"Val intrusion rate: {df_val['Intrusion'].mean():.4f}")
    print(f"Test intrusion rate: {df_test['Intrusion'].mean():.4f}")
    
    return df_train, df_val, df_test


def train_supervised_models(X_train, y_train, X_val, y_val):
    """Train supervised models"""
    print("\n" + "="*50)
    print("TRAINING SUPERVISED MODELS")
    print("="*50)
    
    trainer = SupervisedModelTrainer()
    models, results = trainer.train_all(X_train, y_train, X_val, y_val)
    
    # Print comparison
    print("\n=== Supervised Models Comparison ===")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 80)
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['roc_auc']:<12.4f}")
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    print(f"\nBest supervised model: {best_name}")
    
    return trainer, best_name, best_model


def train_unsupervised_models(X_train, y_train, X_val, y_val):
    """Train unsupervised models"""
    print("\n" + "="*50)
    print("TRAINING UNSUPERVISED MODELS")
    print("="*50)
    
    # Use only normal data for training unsupervised models
    # Convert to numpy arrays for proper indexing
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    normal_mask = (y_train == 0)
    X_train_normal = X_train[normal_mask]
    print(f"Training on {len(X_train_normal)} normal samples")
    
    trainer = UnsupervisedModelTrainer()
    models, results = trainer.train_all(X_train_normal, X_val, y_val)
    
    # Print comparison
    print("\n=== Unsupervised Models Comparison ===")
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)
    for model_name, metrics in results.items():
        if 'accuracy' in metrics:
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    if best_name:
        print(f"\nBest unsupervised model: {best_name}")
    else:
        best_name = 'isolation_forest'
        best_model = trainer.models[best_name]
        print(f"\nUsing default unsupervised model: {best_name}")
    
    return trainer, best_name, best_model


def main():
    """Main training pipeline"""
    # Configuration - adjust paths for this folder structure
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                            'Time-Series_Network_logs.csv')
    DATA_PATH = os.path.abspath(DATA_PATH)
    
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load and split data
    df_train, df_val, df_test = load_and_split_data(DATA_PATH)
    
    # Preprocess data
    print("\n" + "="*50)
    print("PREPROCESSING DATA")
    print("="*50)
    
    preprocessor = DataPreprocessor(feature_selection_k=25)
    X_train, y_train = preprocessor.fit_transform(df_train, target_col='Intrusion')
    X_val = preprocessor.transform(df_val)
    y_val = df_val['Intrusion'].values
    X_test = preprocessor.transform(df_test)
    y_test = df_test['Intrusion'].values
    
    print(f"Processed features: {X_train.shape[1]}")
    if preprocessor.selected_features:
        print(f"Selected features: {preprocessor.selected_features[:10]}...")  # Show first 10
    
    # Save preprocessor
    preprocessor.save(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    print(f"\nPreprocessor saved to {MODEL_DIR}/preprocessor.pkl")
    
    # Train supervised models
    sup_trainer, sup_best_name, sup_best_model = train_supervised_models(X_train, y_train, X_val, y_val)
    
    # Save supervised models
    for model_name in sup_trainer.models.keys():
        sup_trainer.save_model(model_name, os.path.join(MODEL_DIR, f'supervised_{model_name}.pkl'))
    print(f"\nSupervised models saved to {MODEL_DIR}/")
    
    # Train unsupervised models
    unsup_trainer, unsup_best_name, unsup_best_model = train_unsupervised_models(X_train, y_train, X_val, y_val)
    
    # Save unsupervised models
    for model_name in unsup_trainer.models.keys():
        unsup_trainer.save_model(model_name, os.path.join(MODEL_DIR, f'unsupervised_{model_name}.pkl'))
    print(f"\nUnsupervised models saved to {MODEL_DIR}/")
    
    # Create hybrid detector
    print("\n" + "="*50)
    print("SETTING UP HYBRID DETECTOR")
    print("="*50)
    
    hybrid_detector = HybridDetector(
        supervised_model_name=sup_best_name,
        unsupervised_model_name=unsup_best_name
    )
    
    # Wrap models for hybrid detector
    class ModelWrapper:
        def __init__(self, model, trainer):
            self.model = model
            self.trainer = trainer
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
    
    sup_wrapper = ModelWrapper(sup_best_model, sup_trainer)
    unsup_wrapper = unsup_trainer
    
    hybrid_detector.set_models(sup_wrapper, unsup_wrapper, preprocessor)
    
    # Save hybrid detector
    hybrid_detector.save(os.path.join(MODEL_DIR, 'hybrid_detector.pkl'))
    print(f"Hybrid detector saved to {MODEL_DIR}/hybrid_detector.pkl")
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATING ON TEST SET")
    print("="*50)
    
    test_results = hybrid_detector.detect_batch(df_test)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_true = df_test['Intrusion'].values
    y_pred = test_results['is_attack'].values.astype(int)
    
    print(f"\nTest Set Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print(f"\nDetection Statistics:")
    stats = hybrid_detector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"\nAll models saved to {MODEL_DIR}/")
    print("You can now use the trained models for detection.")


if __name__ == '__main__':
    main()

