# Network Intrusion Detection System (NIDS)

A comprehensive Machine Learning-based Network Intrusion Detection System that combines signature-based detection with supervised and unsupervised ML models to detect both known and novel (zero-day) network intrusions.

## Project Structure

```
ML-Network Intrusion Detection System/
├── src/                          # Core source code
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   ├── supervised_models.py      # Supervised ML models
│   ├── unsupervised_models.py    # Unsupervised anomaly detection
│   ├── signature_detection.py    # Signature-based detection
│   ├── hybrid_detection.py       # Hybrid fusion system
│   ├── online_learning.py        # Online learning module
│   ├── evaluation.py             # Evaluation framework
│   └── alert_system.py           # Alert notification system
├── scripts/                      # Main scripts
│   ├── train_models.py          # Training script
│   └── evaluate_system.py       # Evaluation script
├── dashboard/                    # Streamlit dashboard
│   └── app.py                    # Dashboard application
├── models/                       # Trained models (generated)
├── results/                      # Evaluation results (generated)
├── data/                         # Data directory
│   ├── processed/               # Processed datasets
│   ├── raw/                     # Raw data
│   └── incremental/             # Incremental learning data
├── notebooks/                    # Jupyter notebooks for exploration
├── docs/                         # Documentation
└── README.md                     # This file
```

## Features

- **Hybrid Detection**: Combines signature-based and ML-based detection
- **Known Threat Detection**: Supervised models (Random Forest, Logistic Regression, XGBoost)
- **Novel Threat Detection**: Unsupervised anomaly detection (Isolation Forest, One-Class SVM, Autoencoder)
- **Online Learning**: Continuous model updates from new data
- **Real-time Dashboard**: Interactive Streamlit dashboard for monitoring
- **Comprehensive Evaluation**: Three test scenarios with performance metrics
- **Alert System**: Automated alert generation and notification

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas, numpy, scikit-learn
- xgboost
- matplotlib, seaborn, plotly
- streamlit
- joblib, psutil
- tensorflow (optional, for autoencoder)
- river (optional, for advanced online learning)

2. **Prepare dataset**:
   - Place `Time-Series_Network_logs.csv` in the parent directory (Firewall_Frontend/)
   - The dataset should contain columns: Timestamp, Source_IP, Destination_IP, Port, Request_Type, Protocol, Payload_Size, User_Agent, Status, Intrusion, Scan_Type

## Usage

### 1. Train Models

Train all models (supervised, unsupervised, and hybrid):

```bash
cd "ML-Network Intrusion Detection System"
python scripts/train_models.py
```

This will:
- Load and preprocess the dataset
- Train supervised models (RF, LR, XGBoost)
- Train unsupervised models (Isolation Forest, One-Class SVM, Autoencoder)
- Create and save the hybrid detector
- Save all models to `models/` directory

### 2. Evaluate System

Run comprehensive evaluation on all three scenarios:

```bash
python scripts/evaluate_system.py
```

This will:
- Evaluate Scenario 1: Normal + Known Attack
- Evaluate Scenario 2: Normal + Novel Attack
- Evaluate Scenario 3: Normal + Known + Novel Attack
- Test performance under different network loads
- Generate comparison reports and visualizations
- Save results to `results/` directory

### 3. Launch Dashboard

Start the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- Real-time network monitoring
- Batch analysis
- Historical statistics
- Alert management
- Interactive visualizations

## System Components

### 1. Data Preprocessing
- Temporal feature extraction (hour, day, weekday, etc.)
- IP address encoding
- Categorical variable encoding
- Network-specific feature engineering
- Feature selection

### 2. Supervised Models
- **Random Forest**: Ensemble method for known threat detection
- **Logistic Regression**: Linear model for fast inference
- **XGBoost**: Gradient boosting for high accuracy

### 3. Unsupervised Models
- **Isolation Forest**: Tree-based anomaly detection
- **One-Class SVM**: Support vector machine for novelty detection
- **Autoencoder**: Neural network for reconstruction-based anomaly detection

### 4. Signature Detection
- Rule-based detection for known attack patterns
- Port scanning detection
- Botnet attack detection
- Protocol mismatch detection
- Custom signature support

### 5. Hybrid Detection
- Fusion algorithm combining all detection methods
- Weighted decision making
- Conflict resolution

### 6. Online Learning
- Incremental model updates
- SGD (Stochastic Gradient Descent)
- Hoeffding Tree (if River library available)
- Adaptive Random Forest (if River library available)

## Evaluation Scenarios

### Scenario 1: Normal + Known Attack
Tests detection of known attack patterns (BotAttack, PortScan) mixed with normal traffic.

### Scenario 2: Normal + Novel Attack
Tests detection of zero-day/novel attacks that weren't in training data.

### Scenario 3: Normal + Known + Novel Attack
Tests detection when both known and novel attacks are present simultaneously.

## Performance Metrics

The system evaluates:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: False alarms
- **Latency**: Detection time per sample
- **Throughput**: Samples processed per second
- **Memory Usage**: Resource consumption

## Fusion Algorithm

The hybrid fusion algorithm combines decisions from:
- Signature-based detection (weight: 30%)
- Supervised ML (weight: 50%)
- Unsupervised ML (weight: 20%)

See documentation for detailed pseudocode and decision flow.

## Configuration

### Alert Configuration
Create `config/alert_config.json`:
```json
{
  "email_enabled": true,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "email_from": "your-email@gmail.com",
  "email_to": ["admin@company.com"],
  "email_password": "your-app-password",
  "alert_threshold": 0.7,
  "high_severity_threshold": 0.9
}
```

## Troubleshooting

### Models Not Found
**Solution**: Run `python scripts/train_models.py` first

### TensorFlow Issues
**Solution**: Install TensorFlow or the system will use a fallback MLPRegressor from sklearn.

### River Library
**Solution**: Online learning algorithms (Hoeffding Tree, Adaptive RF) require the River library. If not available, SGD will be used as fallback.

### Import Errors
**Solution**: Ensure you're running scripts from the correct directory. Use:
```bash
cd "ML-Network Intrusion Detection System"
python scripts/train_models.py
```

## Documentation

- System architecture and component details
- Fusion algorithm pseudocode
- Feature importance analysis
- Usage guides and examples

## License

[Your License Here]

## Contact

[Your Contact Information]

