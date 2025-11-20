# NIDS Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python train_models.py
```
This will:
- Load and preprocess the dataset
- Train supervised models (RF, LR, XGBoost)
- Train unsupervised models (Isolation Forest, One-Class SVM, Autoencoder)
- Create and save the hybrid detector
- Save all models to `models/` directory

**Expected time**: 5-15 minutes depending on dataset size

### 3. Quick Demo
```bash
python quick_start.py
```
This demonstrates basic detection on a sample of data.

### 4. Comprehensive Evaluation
```bash
python evaluate_system.py
```
This evaluates the system on all three scenarios and generates:
- Performance metrics
- Comparison reports
- Visualization plots
- Results saved to `results/` directory

### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser to the URL shown (typically http://localhost:8501)

## Detailed Workflow

### Training Phase

1. **Data Preparation**
   - Ensure `Time-Series_Network_logs.csv` is in the root directory
   - Dataset should have columns: Timestamp, Source_IP, Destination_IP, Port, Request_Type, Protocol, Payload_Size, User_Agent, Status, Intrusion, Scan_Type

2. **Run Training**
   ```bash
   python train_models.py
   ```

3. **Verify Models**
   - Check that `models/` directory contains:
     - `preprocessor.pkl`
     - `supervised_random_forest.pkl`
     - `supervised_logistic_regression.pkl`
     - `supervised_xgboost.pkl`
     - `unsupervised_isolation_forest.pkl`
     - `unsupervised_one_class_svm.pkl`
     - `unsupervised_autoencoder.pkl`
     - `hybrid_detector.pkl`

### Evaluation Phase

1. **Run Evaluation**
   ```bash
   python evaluate_system.py
   ```

2. **Review Results**
   - Check `results/comparison_report.csv` for metrics
   - View `results/scenario_comparison.png` for visualizations
   - Review `results/evaluation_results.json` for detailed results

3. **Interpret Metrics**
   - **Accuracy**: Overall correctness (aim for >95%)
   - **Precision**: Low false positives (aim for >90%)
   - **Recall**: Low false negatives (aim for >85%)
   - **F1 Score**: Balanced metric (aim for >90%)
   - **Latency**: Detection time per sample (aim for <10ms)
   - **Throughput**: Samples per second (aim for >1000)

### Deployment Phase

1. **Real-time Detection**
   - Use the dashboard for interactive monitoring
   - Or integrate the hybrid detector in your application:
   ```python
   from src.hybrid_detection import HybridDetector
   from src.data_preprocessing import DataPreprocessor
   import joblib
   
   # Load models
   preprocessor = DataPreprocessor()
   preprocessor.load('models/preprocessor.pkl')
   
   hybrid_detector = HybridDetector()
   # ... load models (see train_models.py for example)
   
   # Detect on new data
   results = hybrid_detector.detect_batch(new_dataframe)
   ```

2. **Online Learning**
   - Update models incrementally as new labeled data arrives
   - See `src/online_learning.py` for implementation

3. **Alert Management**
   - Configure alerts in `config/alert_config.json`
   - Alerts are automatically generated for high-confidence detections

## Dashboard Usage

### Live Monitoring Mode
1. Upload a CSV file with network logs
2. Click "Analyze Network Traffic"
3. View:
   - Detection statistics
   - Attack breakdown
   - Confidence distributions
   - Detailed results table

### Batch Analysis Mode
1. Upload a CSV file
2. Click "Analyze Batch"
3. View time-series analysis and batch metrics

### Historical Data Mode
- View cumulative statistics
- Review recent alerts
- Monitor system performance over time

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

### Model Configuration
Modify weights in `src/hybrid_detection.py`:
```python
hybrid_detector = HybridDetector(
    signature_weight=0.3,      # Adjust based on performance
    supervised_weight=0.5,      # Adjust based on performance
    unsupervised_weight=0.2     # Adjust based on performance
)
```

## Troubleshooting

### Issue: "Models not found"
**Solution**: Run `python train_models.py` first

### Issue: "TensorFlow not available"
**Solution**: Install TensorFlow or the system will use a fallback MLPRegressor

### Issue: "Memory error during training"
**Solution**: 
- Reduce dataset size
- Reduce feature selection K in DataPreprocessor
- Use smaller batch sizes

### Issue: "Low detection accuracy"
**Solution**:
- Check data quality
- Retrain with more data
- Adjust fusion algorithm weights
- Tune model hyperparameters

### Issue: "Dashboard not loading"
**Solution**:
- Ensure Streamlit is installed: `pip install streamlit`
- Check that models are trained
- Verify file paths in dashboard/app.py

## Best Practices

1. **Regular Retraining**: Retrain models monthly or when attack patterns change
2. **Feature Monitoring**: Monitor feature importance over time
3. **Threshold Tuning**: Adjust confidence thresholds based on false positive/negative rates
4. **Signature Updates**: Regularly update signature rules based on new attack patterns
5. **Performance Monitoring**: Track latency and throughput to ensure real-time capability
6. **Alert Tuning**: Adjust alert thresholds to reduce false positives while maintaining detection

## Advanced Usage

### Custom Signatures
Add custom signatures in `src/signature_detection.py`:
```python
signature_detector.add_signature(
    name='custom_attack',
    rules=[lambda row: row['Port'] == 1234],
    description='Custom attack pattern',
    severity='high',
    weight=0.9
)
```

### Online Learning
Update models incrementally:
```python
from src.online_learning import OnlineLearner

learner = OnlineLearner(algorithm='sgd')
learner.partial_fit(X_new, y_new)
predictions = learner.predict(X_test)
```

### Custom Evaluation
Create custom evaluation scenarios:
```python
from src.evaluation import NIDSEvaluator

evaluator = NIDSEvaluator(hybrid_detector, preprocessor)
metrics = evaluator.evaluate_scenario("Custom Scenario", df)
```

## Performance Optimization

1. **Feature Selection**: Reduce K in DataPreprocessor for faster processing
2. **Model Selection**: Use faster models (e.g., Logistic Regression) for real-time
3. **Batch Processing**: Process in batches for better throughput
4. **Caching**: Use Streamlit caching for dashboard
5. **Parallel Processing**: Enable n_jobs=-1 in models for parallel training

## Support

For issues or questions:
1. Check documentation in `docs/` directory
2. Review code comments
3. Check evaluation results for model performance
4. Review logs for error messages

