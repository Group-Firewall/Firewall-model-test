"""
Quick Start Script
Demonstrates basic usage of the NIDS system
"""

import pandas as pd
import os
import sys

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from data_preprocessing import DataPreprocessor
from hybrid_detection import HybridDetector
from supervised_models import SupervisedModelTrainer
from unsupervised_models import UnsupervisedModelTrainer
import joblib


def quick_demo():
    """Quick demonstration of NIDS capabilities"""
    print("="*60)
    print("NIDS Quick Start Demo")
    print("="*60)
    
    # Check if models exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    MODEL_DIR = os.path.join(parent_dir, 'models')
    
    if not os.path.exists(os.path.join(MODEL_DIR, 'preprocessor.pkl')):
        print("\n⚠️  Models not found. Please run scripts/train_models.py first.")
        print("   This will train all models and save them to models/ directory.")
        return
    
    # Load a sample of data
    print("\n1. Loading sample data...")
    DATA_PATH = os.path.join(parent_dir, '..', 'Time-Series_Network_logs.csv')
    DATA_PATH = os.path.abspath(DATA_PATH)
    
    if not os.path.exists(DATA_PATH):
        print(f"   ⚠️  Dataset not found at {DATA_PATH}")
        print("   Please ensure Time-Series_Network_logs.csv is in the parent directory.")
        return
    
    df = pd.read_csv(DATA_PATH)
    df_sample = df.sample(n=100, random_state=42)
    print(f"   Loaded {len(df_sample)} samples")
    
    # Load models
    print("\n2. Loading trained models...")
    try:
        preprocessor = DataPreprocessor()
        preprocessor.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
        
        # Load supervised model
        sup_trainer = SupervisedModelTrainer()
        sup_trainer.load_model('random_forest', os.path.join(MODEL_DIR, 'supervised_random_forest.pkl'))
        sup_model = sup_trainer.models['random_forest']
        
        # Load unsupervised model
        unsup_trainer = UnsupervisedModelTrainer()
        unsup_trainer.load_model('isolation_forest', os.path.join(MODEL_DIR, 'unsupervised_isolation_forest.pkl'))
        
        # Create wrapper
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
            def predict(self, X):
                return self.model.predict(X)
            def predict_proba(self, X):
                return self.model.predict_proba(X)
        
        sup_wrapper = ModelWrapper(sup_model)
        
        # Create hybrid detector
        hybrid_detector = HybridDetector()
        hybrid_detector.set_models(sup_wrapper, unsup_trainer, preprocessor)
        
        print("   ✓ Models loaded successfully")
        
    except Exception as e:
        print(f"   ✗ Error loading models: {e}")
        print("   Please ensure models are trained first (run scripts/train_models.py)")
        return
    
    # Run detection
    print("\n3. Running intrusion detection...")
    results = hybrid_detector.detect_batch(df_sample)
    
    # Display results
    print("\n4. Detection Results:")
    print(f"   Total samples: {len(df_sample)}")
    print(f"   Attacks detected: {results['is_attack'].sum()}")
    print(f"   Normal traffic: {(results['is_attack'] == False).sum()}")
    print(f"   Signature detections: {results['signature_detected'].sum()}")
    print(f"   ML detections: {results['ml_detected'].sum()}")
    print(f"   Average confidence: {results['confidence'].mean():.2%}")
    
    # Show sample detections
    attacks = results[results['is_attack'] == True]
    if len(attacks) > 0:
        print("\n5. Sample Detected Attacks:")
        for idx, attack in attacks.head(5).iterrows():
            row = df_sample.iloc[idx]
            print(f"\n   Attack #{idx}:")
            print(f"   - Source: {row.get('Source_IP', 'Unknown')}")
            print(f"   - Destination: {row.get('Destination_IP', 'Unknown')}")
            print(f"   - Port: {row.get('Port', 'Unknown')}")
            print(f"   - Type: {attack.get('signature_name', 'ML Detected')}")
            print(f"   - Confidence: {attack['confidence']:.2%}")
    
    # Statistics
    stats = hybrid_detector.get_statistics()
    print("\n6. Detection Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  - Run 'python scripts/evaluate_system.py' for comprehensive evaluation")
    print("  - Run 'streamlit run dashboard/app.py' for interactive dashboard")
    print("  - Check README.md for detailed documentation")


if __name__ == '__main__':
    quick_demo()

