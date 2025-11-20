"""
Evaluation Framework
Implements comprehensive evaluation for NIDS with multiple scenarios
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class NIDSEvaluator:
    """Comprehensive evaluation framework for NIDS"""
    
    def __init__(self, hybrid_detector, preprocessor):
        self.hybrid_detector = hybrid_detector
        self.preprocessor = preprocessor
        self.results = {}
        
    def evaluate_scenario(self, scenario_name: str, df: pd.DataFrame, 
                         known_attacks: bool = True, novel_attacks: bool = False) -> Dict:
        """
        Evaluate NIDS on a specific scenario
        
        Args:
            scenario_name: Name of the scenario
            df: Test dataframe
            known_attacks: Whether dataset contains known attacks
            novel_attacks: Whether dataset contains novel/zero-day attacks
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Scenario: {scenario_name}")
        print(f"{'='*60}")
        
        # Preprocess
        start_time = time.time()
        X_processed = self.preprocessor.transform(df)
        preprocessing_time = time.time() - start_time
        
        # Detect
        start_time = time.time()
        results = self.hybrid_detector.detect_batch(df)
        detection_time = time.time() - start_time
        
        # Calculate metrics
        y_true = df['Intrusion'].values
        y_pred = results['is_attack'].values.astype(int)
        
        # Performance metrics
        metrics = {
            'scenario': scenario_name,
            'n_samples': len(df),
            'n_attacks': int(y_true.sum()),
            'n_normal': int((y_true == 0).sum()),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'false_positive_rate': self._calculate_fpr(y_true, y_pred),
            'false_negative_rate': self._calculate_fnr(y_true, y_pred),
            'latency_per_sample_ms': (detection_time / len(df)) * 1000,
            'throughput_samples_per_sec': len(df) / detection_time,
            'preprocessing_time': preprocessing_time,
            'detection_time': detection_time,
            'total_time': preprocessing_time + detection_time
        }
        
        # Resource usage
        process = psutil.Process(os.getpid())
        metrics['cpu_percent'] = process.cpu_percent()
        metrics['memory_mb'] = process.memory_info().rss / 1024 / 1024
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        # Detection breakdown
        metrics['signature_detections'] = int(results['signature_detected'].sum())
        metrics['ml_detections'] = int(results['ml_detected'].sum())
        metrics['hybrid_detections'] = int(results['is_attack'].sum())
        metrics['conflicts'] = int(results['conflict'].sum())
        
        # Confidence statistics
        metrics['avg_confidence'] = float(results['confidence'].mean())
        metrics['avg_signature_confidence'] = float(results['signature_confidence'].mean())
        metrics['avg_ml_confidence'] = float(results['ml_confidence'].mean())
        
        self.results[scenario_name] = {
            'metrics': metrics,
            'detailed_results': results,
            'data': df
        }
        
        # Print summary
        self._print_scenario_summary(metrics)
        
        return metrics
    
    def evaluate_scenario_1(self, df_normal: pd.DataFrame, df_known_attack: pd.DataFrame) -> Dict:
        """Scenario 1: Normal + Known Attack"""
        df = pd.concat([df_normal, df_known_attack], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return self.evaluate_scenario("Scenario 1: Normal + Known Attack", df, 
                                     known_attacks=True, novel_attacks=False)
    
    def evaluate_scenario_2(self, df_normal: pd.DataFrame, df_novel_attack: pd.DataFrame) -> Dict:
        """Scenario 2: Normal + Novel Attack (Zero-day)"""
        df = pd.concat([df_normal, df_novel_attack], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return self.evaluate_scenario("Scenario 2: Normal + Novel Attack", df, 
                                     known_attacks=False, novel_attacks=True)
    
    def evaluate_scenario_3(self, df_normal: pd.DataFrame, df_known_attack: pd.DataFrame,
                           df_novel_attack: pd.DataFrame) -> Dict:
        """Scenario 3: Normal + Known + Novel Attack"""
        df = pd.concat([df_normal, df_known_attack, df_novel_attack], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return self.evaluate_scenario("Scenario 3: Normal + Known + Novel Attack", df, 
                                     known_attacks=True, novel_attacks=True)
    
    def evaluate_different_loads(self, df: pd.DataFrame, load_sizes: List[int]) -> Dict:
        """Evaluate system under different network loads"""
        print(f"\n{'='*60}")
        print("Evaluating Different Network Loads")
        print(f"{'='*60}")
        
        load_results = {}
        
        for load_size in load_sizes:
            if load_size > len(df):
                load_size = len(df)
            
            df_sample = df.sample(n=load_size, random_state=42)
            
            start_time = time.time()
            X_processed = self.preprocessor.transform(df_sample)
            preprocessing_time = time.time() - start_time
            
            start_time = time.time()
            results = self.hybrid_detector.detect_batch(df_sample)
            detection_time = time.time() - start_time
            
            y_true = df_sample['Intrusion'].values
            y_pred = results['is_attack'].values.astype(int)
            
            load_results[load_size] = {
                'n_samples': load_size,
                'latency_ms': (detection_time / load_size) * 1000,
                'throughput_samples_per_sec': load_size / detection_time,
                'accuracy': accuracy_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'false_positive_rate': self._calculate_fpr(y_true, y_pred),
                'total_time': preprocessing_time + detection_time
            }
            
            print(f"Load {load_size}: Latency={load_results[load_size]['latency_ms']:.2f}ms, "
                  f"Throughput={load_results[load_size]['throughput_samples_per_sec']:.2f} samples/sec")
        
        return load_results
    
    def _calculate_fpr(self, y_true, y_pred):
        """Calculate false positive rate"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def _calculate_fnr(self, y_true, y_pred):
        """Calculate false negative rate"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    def _print_scenario_summary(self, metrics: Dict):
        """Print scenario evaluation summary"""
        print(f"\nResults for {metrics['scenario']}:")
        print(f"  Samples: {metrics['n_samples']} (Attacks: {metrics['n_attacks']}, Normal: {metrics['n_normal']})")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        print(f"  Latency: {metrics['latency_per_sample_ms']:.2f} ms/sample")
        print(f"  Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"  Memory: {metrics['memory_mb']:.2f} MB")
        print(f"  Signature Detections: {metrics['signature_detections']}")
        print(f"  ML Detections: {metrics['ml_detections']}")
        print(f"  Conflicts: {metrics['conflicts']}")
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comparison report across all scenarios"""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for scenario_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Scenario': scenario_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'FPR': metrics['false_positive_rate'],
                'FNR': metrics['false_negative_rate'],
                'Latency (ms)': metrics['latency_per_sample_ms'],
                'Throughput (samples/sec)': metrics['throughput_samples_per_sec'],
                'Memory (MB)': metrics['memory_mb']
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_performance_metrics(self, save_path: str = None):
        """Plot performance metrics across scenarios"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NIDS Performance Metrics Across Scenarios', fontsize=16)
        
        scenarios = list(self.results.keys())
        metrics_to_plot = [
            ('accuracy', 'Accuracy', axes[0, 0]),
            ('precision', 'Precision', axes[0, 1]),
            ('recall', 'Recall', axes[0, 2]),
            ('f1', 'F1 Score', axes[1, 0]),
            ('false_positive_rate', 'False Positive Rate', axes[1, 1]),
            ('latency_per_sample_ms', 'Latency (ms)', axes[1, 2])
        ]
        
        for metric_key, metric_name, ax in metrics_to_plot:
            values = [self.results[s]['metrics'][metric_key] for s in scenarios]
            ax.bar(scenarios, values, color='steelblue', alpha=0.7)
            ax.set_title(metric_name)
            ax.set_ylabel(metric_name)
            ax.set_xticklabels(scenarios, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_load_performance(self, load_results: Dict, save_path: str = None):
        """Plot performance under different loads"""
        loads = sorted(load_results.keys())
        latencies = [load_results[l]['latency_ms'] for l in loads]
        throughputs = [load_results[l]['throughput_samples_per_sec'] for l in loads]
        accuracies = [load_results[l]['accuracy'] for l in loads]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Performance Under Different Network Loads', fontsize=16)
        
        axes[0].plot(loads, latencies, marker='o', color='red')
        axes[0].set_xlabel('Load (samples)')
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Latency vs Load')
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(loads, throughputs, marker='o', color='green')
        axes[1].set_xlabel('Load (samples)')
        axes[1].set_ylabel('Throughput (samples/sec)')
        axes[1].set_title('Throughput vs Load')
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(loads, accuracies, marker='o', color='blue')
        axes[2].set_xlabel('Load (samples)')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Accuracy vs Load')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, filepath: str):
        """Save evaluation results to file"""
        import json
        
        # Convert to JSON-serializable format
        results_serializable = {}
        for scenario, result in self.results.items():
            results_serializable[scenario] = {
                'metrics': result['metrics'],
                'confusion_matrix': result['metrics']['confusion_matrix']
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {filepath}")

