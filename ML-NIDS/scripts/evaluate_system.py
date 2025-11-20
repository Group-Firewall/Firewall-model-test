"""
Evaluation Script
Runs comprehensive evaluation on all three scenarios
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
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from data_preprocessing import DataPreprocessor
from supervised_models import SupervisedModelTrainer
from unsupervised_models import UnsupervisedModelTrainer
from hybrid_detection import HybridDetector
from evaluation import NIDSEvaluator
import joblib


def prepare_scenarios(df: pd.DataFrame):
    """Prepare data for three evaluation scenarios"""
    # Separate normal and attack data
    df_normal = df[df['Intrusion'] == 0].copy()
    df_attacks = df[df['Intrusion'] == 1].copy()
    
    # Split attacks into known and novel (simulate zero-day)
    # Known attacks: BotAttack and PortScan (from training data)
    # Novel attacks: Random subset treated as zero-day
    known_attack_types = ['BotAttack', 'PortScan']
    df_known_attacks = df_attacks[df_attacks['Scan_Type'].isin(known_attack_types)].copy()
    
    # For novel attacks, we'll use a subset and mark as "unknown" type
    # In real scenario, these would be completely new attack patterns
    df_novel_attacks = df_attacks[~df_attacks['Scan_Type'].isin(known_attack_types)].copy()
    
    # If no novel attacks in data, create synthetic by sampling known attacks
    # and modifying features slightly
    if len(df_novel_attacks) < 50:
        print("Creating synthetic novel attacks...")
        df_novel_attacks = df_known_attacks.sample(
            min(200, len(df_known_attacks)), 
            random_state=42
        ).copy()
        # Modify to simulate novel attack
        df_novel_attacks['Scan_Type'] = 'NovelAttack'
        df_novel_attacks['Payload_Size'] = df_novel_attacks['Payload_Size'] * 1.5
        df_novel_attacks['Port'] = df_novel_attacks['Port'] + 1000
    
    print(f"\nData Preparation:")
    print(f"  Normal samples: {len(df_normal)}")
    print(f"  Known attacks: {len(df_known_attacks)}")
    print(f"  Novel attacks: {len(df_novel_attacks)}")
    
    return df_normal, df_known_attacks, df_novel_attacks


def load_models():
    """Load trained models"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    MODEL_DIR = os.path.join(parent_dir, 'models')
    
    # Load preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    
    # Load supervised model
    sup_trainer = SupervisedModelTrainer()
    # Try to load best model (random_forest by default)
    try:
        sup_trainer.load_model('random_forest', os.path.join(MODEL_DIR, 'supervised_random_forest.pkl'))
        sup_model = sup_trainer.models['random_forest']
        sup_model_name = 'random_forest'
    except:
        # Load any available supervised model
        for model_file in os.listdir(MODEL_DIR):
            if model_file.startswith('supervised_'):
                model_name = model_file.replace('supervised_', '').replace('.pkl', '')
                sup_trainer.load_model(model_name, os.path.join(MODEL_DIR, model_file))
                sup_model = sup_trainer.models[model_name]
                sup_model_name = model_name
                break
    
    # Load unsupervised model
    unsup_trainer = UnsupervisedModelTrainer()
    try:
        unsup_trainer.load_model('isolation_forest', os.path.join(MODEL_DIR, 'unsupervised_isolation_forest.pkl'))
        unsup_model = unsup_trainer
        unsup_model_name = 'isolation_forest'
    except:
        # Load any available unsupervised model
        for model_file in os.listdir(MODEL_DIR):
            if model_file.startswith('unsupervised_'):
                model_name = model_file.replace('unsupervised_', '').replace('.pkl', '')
                unsup_trainer.load_model(model_name, os.path.join(MODEL_DIR, model_file))
                unsup_model = unsup_trainer
                unsup_model_name = model_name
                break
    
    # Create model wrappers
    class ModelWrapper:
        def __init__(self, model):
            self.model = model
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
    
    sup_wrapper = ModelWrapper(sup_model)
    
    # Create hybrid detector
    hybrid_detector = HybridDetector(
        supervised_model_name=sup_model_name,
        unsupervised_model_name=unsup_model_name
    )
    hybrid_detector.set_models(sup_wrapper, unsup_model, preprocessor)
    
    return hybrid_detector, preprocessor


def main():
    """Main evaluation pipeline"""
    print("="*60)
    print("NIDS COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Load data - adjust path for this folder structure
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    DATA_PATH = os.path.join(parent_dir, 'Time-Series_Network_logs.csv')
    DATA_PATH = os.path.abspath(DATA_PATH)
    
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape: {df.shape}")
    
    # Prepare scenarios
    df_normal, df_known_attacks, df_novel_attacks = prepare_scenarios(df)
    
    # Load models
    print("\nLoading trained models...")
    try:
        hybrid_detector, preprocessor = load_models()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run scripts/train_models.py first to train the models.")
        return
    
    # Create evaluator
    evaluator = NIDSEvaluator(hybrid_detector, preprocessor)
    
    # Sample data for evaluation (to keep it manageable)
    sample_size_normal = min(1000, len(df_normal))
    sample_size_known = min(200, len(df_known_attacks))
    sample_size_novel = min(200, len(df_novel_attacks))
    
    df_normal_sample = df_normal.sample(n=sample_size_normal, random_state=42)
    df_known_sample = df_known_attacks.sample(n=sample_size_known, random_state=42)
    df_novel_sample = df_novel_attacks.sample(n=sample_size_novel, random_state=42)
    
    # Evaluate Scenario 1: Normal + Known Attack
    print("\n" + "="*60)
    evaluator.evaluate_scenario_1(
        df_normal_sample.copy(),
        df_known_sample.copy()
    )
    
    # Evaluate Scenario 2: Normal + Novel Attack
    print("\n" + "="*60)
    evaluator.evaluate_scenario_2(
        df_normal_sample.copy(),
        df_novel_sample.copy()
    )
    
    # Evaluate Scenario 3: Normal + Known + Novel Attack
    print("\n" + "="*60)
    evaluator.evaluate_scenario_3(
        df_normal_sample.copy(),
        df_known_sample.copy(),
        df_novel_sample.copy()
    )
    
    # Evaluate different loads
    print("\n" + "="*60)
    print("Evaluating Different Network Loads")
    print("="*60)
    
    # Combine all data for load testing
    df_all = pd.concat([df_normal_sample, df_known_sample, df_novel_sample], ignore_index=True)
    load_sizes = [100, 500, 1000, 1500]
    load_results = evaluator.evaluate_different_loads(df_all, load_sizes)
    
    # Generate comparison report
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)
    comparison_df = evaluator.generate_comparison_report()
    print(comparison_df.to_string(index=False))
    
    # Save results
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    RESULTS_DIR = os.path.join(parent_dir, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_report.csv'), index=False)
    evaluator.save_results(os.path.join(RESULTS_DIR, 'evaluation_results.json'))
    
    # Generate plots
    print("\nGenerating performance plots...")
    evaluator.plot_performance_metrics(os.path.join(RESULTS_DIR, 'scenario_comparison.png'))
    evaluator.plot_load_performance(load_results, os.path.join(RESULTS_DIR, 'load_performance.png'))
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print(f"  - {RESULTS_DIR}/comparison_report.csv")
    print(f"  - {RESULTS_DIR}/evaluation_results.json")
    print(f"  - {RESULTS_DIR}/scenario_comparison.png")
    print(f"  - {RESULTS_DIR}/load_performance.png")


if __name__ == '__main__':
    main()

