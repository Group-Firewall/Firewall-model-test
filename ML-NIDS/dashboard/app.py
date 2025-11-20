"""
Real-time NIDS Dashboard
Interactive dashboard for monitoring network intrusions
Adapted for ML-Network Intrusion Detection System folder
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
import joblib
from datetime import datetime, timedelta

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

try:
    from hybrid_detection import HybridDetector
    from data_preprocessing import DataPreprocessor
    from supervised_models import SupervisedModelTrainer
    from unsupervised_models import UnsupervisedModelTrainer
except ImportError:
    from src.hybrid_detection import HybridDetector
    from src.data_preprocessing import DataPreprocessor
    from src.supervised_models import SupervisedModelTrainer
    from src.unsupervised_models import UnsupervisedModelTrainer


@st.cache_resource
def load_models():
    """Load trained models (cached)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    MODEL_DIR = os.path.join(parent_dir, 'models')
    
    try:
        # Load preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
        
        # Load supervised model
        sup_trainer = SupervisedModelTrainer()
        try:
            sup_trainer.load_model('random_forest', os.path.join(MODEL_DIR, 'supervised_random_forest.pkl'))
            sup_model = sup_trainer.models['random_forest']
        except:
            # Try any available model
            for f in os.listdir(MODEL_DIR):
                if f.startswith('supervised_'):
                    model_name = f.replace('supervised_', '').replace('.pkl', '')
                    sup_trainer.load_model(model_name, os.path.join(MODEL_DIR, f))
                    sup_model = sup_trainer.models[model_name]
                    break
        
        # Load unsupervised model
        unsup_trainer = UnsupervisedModelTrainer()
        try:
            unsup_trainer.load_model('isolation_forest', os.path.join(MODEL_DIR, 'unsupervised_isolation_forest.pkl'))
        except:
            for f in os.listdir(MODEL_DIR):
                if f.startswith('unsupervised_'):
                    model_name = f.replace('unsupervised_', '').replace('.pkl', '')
                    unsup_trainer.load_model(model_name, os.path.join(MODEL_DIR, f))
                    break
        
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
        
        return hybrid_detector, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


def process_live_data(df, hybrid_detector, preprocessor):
    """Process live data through NIDS"""
    results = hybrid_detector.detect_batch(df)
    return results


def main():
    st.set_page_config(
        page_title="NIDS Dashboard",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Network Intrusion Detection System Dashboard")
    st.markdown("Real-time monitoring and alert management for network security")
    
    # Load models
    with st.spinner("Loading NIDS models..."):
        hybrid_detector, preprocessor = load_models()
    
    if hybrid_detector is None:
        st.error("Failed to load models. Please ensure models are trained first.")
        st.info("Run: `python scripts/train_models.py` to train the models.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Operation Mode",
        ["Live Monitoring", "Batch Analysis", "Historical Data"]
    )
    
    # Initialize session state
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'total_processed': 0,
            'attacks_detected': 0,
            'normal_traffic': 0,
            'signature_detections': 0,
            'ml_detections': 0
        }
    
    # Main content area
    if mode == "Live Monitoring":
        st.header("📊 Live Network Monitoring")
        
        # Upload live data
        uploaded_file = st.file_uploader(
            "Upload network log file (CSV)",
            type=['csv'],
            help="Upload a CSV file with network logs to analyze"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} network log entries")
            
            # Process data
            if st.button("🔍 Analyze Network Traffic", type="primary"):
                with st.spinner("Analyzing network traffic..."):
                    results = process_live_data(df, hybrid_detector, preprocessor)
                    
                    # Update statistics
                    st.session_state.stats['total_processed'] += len(df)
                    st.session_state.stats['attacks_detected'] += results['is_attack'].sum()
                    st.session_state.stats['normal_traffic'] += (results['is_attack'] == False).sum()
                    st.session_state.stats['signature_detections'] += results['signature_detected'].sum()
                    st.session_state.stats['ml_detections'] += results['ml_detected'].sum()
                    
                    # Create alerts for detected attacks
                    attacks = results[results['is_attack'] == True]
                    for idx, attack in attacks.iterrows():
                        alert = {
                            'timestamp': datetime.now(),
                            'source_ip': df.iloc[idx].get('Source_IP', 'Unknown'),
                            'dest_ip': df.iloc[idx].get('Destination_IP', 'Unknown'),
                            'attack_type': attack.get('signature_name', 'ML Detected'),
                            'confidence': attack['confidence'],
                            'severity': 'High' if attack['confidence'] > 0.8 else 'Medium'
                        }
                        st.session_state.alerts.append(alert)
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Traffic", len(df))
                    with col2:
                        st.metric("Attacks Detected", int(results['is_attack'].sum()), 
                                delta=f"{results['is_attack'].mean()*100:.1f}%")
                    with col3:
                        st.metric("Normal Traffic", int((results['is_attack'] == False).sum()))
                    with col4:
                        st.metric("Avg Confidence", f"{results['confidence'].mean():.2f}")
                    
                    # Visualization
                    st.subheader("Detection Breakdown")
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('Attack Distribution', 'Detection Method', 
                                      'Confidence Distribution', 'Signature Types'),
                        specs=[[{"type": "pie"}, {"type": "bar"}],
                               [{"type": "histogram"}, {"type": "bar"}]]
                    )
                    
                    # Attack distribution
                    attack_counts = results['is_attack'].value_counts()
                    fig.add_trace(
                        go.Pie(labels=['Normal', 'Attack'], values=attack_counts.values, name="Traffic"),
                        row=1, col=1
                    )
                    
                    # Detection method
                    detection_methods = []
                    detection_counts = []
                    if results['signature_detected'].sum() > 0:
                        detection_methods.append('Signature')
                        detection_counts.append(results['signature_detected'].sum())
                    if results['ml_detected'].sum() > 0:
                        detection_methods.append('ML')
                        detection_counts.append(results['ml_detected'].sum())
                    if results['is_attack'].sum() > 0:
                        detection_methods.append('Hybrid')
                        detection_counts.append(results['is_attack'].sum())
                    
                    if detection_methods:
                        fig.add_trace(
                            go.Bar(x=detection_methods, y=detection_counts, name="Detections"),
                            row=1, col=2
                        )
                    
                    # Confidence distribution
                    fig.add_trace(
                        go.Histogram(x=results['confidence'], nbinsx=20, name="Confidence"),
                        row=2, col=1
                    )
                    
                    # Signature types
                    if results['signature_detected'].sum() > 0:
                        sig_counts = results[results['signature_detected']]['signature_name'].value_counts()
                        fig.add_trace(
                            go.Bar(x=sig_counts.index, y=sig_counts.values, name="Signatures"),
                            row=2, col=2
                        )
                    
                    fig.update_layout(height=800, showlegend=False, title_text="Network Traffic Analysis")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results table
                    st.subheader("Detailed Detection Results")
                    display_df = df.copy()
                    display_df['Attack_Detected'] = results['is_attack']
                    display_df['Confidence'] = results['confidence']
                    display_df['Detection_Method'] = results.apply(
                        lambda x: 'Signature' if x['signature_detected'] else 'ML' if x['ml_detected'] else 'None',
                        axis=1
                    )
                    
                    show_only_attacks = st.checkbox("Show only attacks")
                    if show_only_attacks:
                        st.dataframe(display_df[display_df['Attack_Detected'] == True])
                    else:
                        st.dataframe(display_df)
    
    elif mode == "Batch Analysis":
        st.header("📈 Batch Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file for batch analysis", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            if st.button("Analyze Batch"):
                with st.spinner("Processing batch..."):
                    results = process_live_data(df, hybrid_detector, preprocessor)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Attacks", int(results['is_attack'].sum()))
                    with col3:
                        st.metric("Detection Rate", f"{results['is_attack'].mean()*100:.2f}%")
                    
                    # Time series analysis
                    if 'Timestamp' in df.columns:
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                        df['Attack'] = results['is_attack']
                        
                        fig = px.scatter(
                            df,
                            x='Timestamp',
                            y='Payload_Size',
                            color='Attack',
                            title="Network Traffic Over Time",
                            labels={'Attack': 'Attack Detected'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    elif mode == "Historical Data":
        st.header("📊 Historical Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Processed", st.session_state.stats['total_processed'])
        with col2:
            st.metric("Attacks Detected", st.session_state.stats['attacks_detected'])
        with col3:
            st.metric("Signature Detections", st.session_state.stats['signature_detections'])
        with col4:
            st.metric("ML Detections", st.session_state.stats['ml_detections'])
        
        # Alerts
        st.subheader("🚨 Recent Alerts")
        if st.session_state.alerts:
            alerts_df = pd.DataFrame(st.session_state.alerts[-50:])  # Last 50 alerts
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.info("No alerts generated yet")
    
    # Feature Importance (if available)
    st.sidebar.header("Model Information")
    st.sidebar.info("""
    **Hybrid NIDS System**
    - Signature-based detection
    - Supervised ML (Random Forest)
    - Unsupervised ML (Isolation Forest)
    - Fusion algorithm for decision making
    """)


if __name__ == '__main__':
    main()

