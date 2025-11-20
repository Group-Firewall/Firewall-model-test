# NIDS System Architecture

## Overview

The Network Intrusion Detection System (NIDS) is a hybrid ML-based system that combines signature-based detection with machine learning models to detect both known and novel (zero-day) network intrusions.

## System Components

### 1. Data Capture Module
- **Location**: `src/data_preprocessing.py`
- **Function**: Captures and preprocesses network traffic logs
- **Features**:
  - Temporal feature extraction (hour, day, weekday, etc.)
  - IP address encoding
  - Categorical variable encoding
  - Network-specific feature engineering
  - Feature selection

### 2. Signature-Based Detection
- **Location**: `src/signature_detection.py`
- **Function**: Detects known attack patterns using rule-based signatures
- **Signatures**:
  - Port scanning attacks
  - Botnet attacks
  - Suspicious port activity
  - Protocol mismatches
  - Large payload detection
  - Rapid connection failures

### 3. Supervised ML Models
- **Location**: `src/supervised_models.py`
- **Function**: Detects known threats using labeled historical data
- **Models**:
  - Random Forest Classifier
  - Logistic Regression
  - XGBoost Classifier
- **Training**: Trained on labeled attack data with balanced class weights

### 4. Unsupervised ML Models
- **Location**: `src/unsupervised_models.py`
- **Function**: Detects novel/zero-day threats using anomaly detection
- **Models**:
  - Isolation Forest
  - One-Class SVM
  - Autoencoder (Neural Network)
- **Training**: Trained only on normal traffic data

### 5. Hybrid Detection System
- **Location**: `src/hybrid_detection.py`
- **Function**: Combines signature-based and ML-based detection
- **Fusion Algorithm**: See `docs/FUSION_ALGORITHM.md`

### 6. Online Learning Module
- **Location**: `src/online_learning.py`
- **Function**: Enables continuous learning from new data
- **Algorithms**:
  - Stochastic Gradient Descent (SGD)
  - Hoeffding Tree (if River library available)
  - Adaptive Random Forest (if River library available)

### 7. Evaluation Framework
- **Location**: `src/evaluation.py`
- **Function**: Comprehensive evaluation across multiple scenarios
- **Scenarios**:
  - Scenario 1: Normal + Known Attack
  - Scenario 2: Normal + Novel Attack
  - Scenario 3: Normal + Known + Novel Attack

### 8. Dashboard
- **Location**: `dashboard/app.py`
- **Function**: Real-time monitoring and visualization
- **Features**:
  - Live network monitoring
  - Batch analysis
  - Historical statistics
  - Alert management
  - Interactive visualizations

### 9. Alert System
- **Location**: `src/alert_system.py`
- **Function**: Alert generation and notification
- **Features**:
  - Email notifications
  - Alert prioritization
  - Alert logging
  - Alert statistics

## Data Flow

```
Network Traffic Logs
    ↓
Data Preprocessing
    ↓
    ├──→ Signature Detection ──┐
    │                          │
    ├──→ Supervised ML ────────┤
    │                          ├──→ Hybrid Fusion ──→ Final Decision
    └──→ Unsupervised ML ──────┘
                                ↓
                          Alert System
                                ↓
                          Dashboard
```

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Network Traffic Logs                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing & Feature Engineering        │
│  - Temporal features                                         │
│  - IP encoding                                              │
│  - Categorical encoding                                     │
│  - Network features                                         │
└──────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   Signature Detection    │    │      ML Detection            │
│  - Port scan rules       │    │  ┌────────────────────────┐  │
│  - Bot attack rules      │    │  │ Supervised Models     │  │
│  - Protocol mismatch     │    │  │ - Random Forest       │  │
│  - Large payload         │    │  │ - Logistic Regression │  │
│                          │    │  │ - XGBoost             │  │
└──────────┬───────────────┘    │  └────────────────────────┘  │
           │                     │  ┌────────────────────────┐  │
           │                     │  │ Unsupervised Models   │  │
           │                     │  │ - Isolation Forest    │  │
           │                     │  │ - One-Class SVM       │  │
           │                     │  │ - Autoencoder         │  │
           │                     │  └────────────────────────┘  │
           │                     └──────────────┬───────────────┘
           │                                    │
           └────────────────┬───────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Fusion Algorithm                   │
│  - Weighted combination                                      │
│  - Conflict resolution                                      │
│  - Confidence scoring                                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final Decision & Alerting                │
│  - Attack classification                                    │
│  - Confidence score                                         │
│  - Alert generation                                         │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard & Monitoring                    │
│  - Real-time visualization                                   │
│  - Alert management                                         │
│  - Performance metrics                                      │
└─────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

### Training Phase
1. Load historical network logs
2. Preprocess data
3. Train supervised models on labeled data
4. Train unsupervised models on normal data only
5. Save models to disk

### Inference Phase
1. Capture live network traffic
2. Preprocess in real-time
3. Run through hybrid detector
4. Generate alerts for detected attacks
5. Update dashboard
6. Optionally update online learning model

## Performance Considerations

- **Latency**: Optimized for real-time detection (< 10ms per sample)
- **Throughput**: Capable of processing 1000+ samples/second
- **Memory**: Efficient feature selection to minimize memory footprint
- **Scalability**: Modular design allows horizontal scaling

## Security Considerations

- Models are trained offline and validated before deployment
- Signature rules are regularly updated
- Online learning includes validation to prevent model poisoning
- Alert system includes rate limiting to prevent alert flooding

