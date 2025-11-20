# Feature Importance Analysis

## Overview

Feature importance analysis helps understand which features contribute most to intrusion detection decisions. This document provides insights into feature importance across different models.

## Feature Categories

### 1. Temporal Features
- **Hour**: Time of day (0-23)
- **Minute**: Minute of hour (0-59)
- **Day**: Day of month (1-31)
- **Weekday**: Day of week (0-6)
- **DayOfYear**: Day of year (1-365)
- **IsWeekend**: Binary indicator (0/1)
- **IsBusinessHours**: Binary indicator (0/1)

**Importance**: Temporal features help identify time-based attack patterns (e.g., attacks during off-hours).

### 2. Network Features
- **Source_IP**: Hashed source IP address
- **Destination_IP**: Hashed destination IP address
- **Port**: Destination port number
- **IsWellKnownPort**: Binary (ports < 1024)
- **IsHighPort**: Binary (ports > 49152)
- **Protocol**: Network protocol (TCP, UDP, ICMP)
- **Request_Type**: Type of request (HTTP, HTTPS, SSH, etc.)

**Importance**: Core network features that directly indicate attack patterns.

### 3. Payload Features
- **Payload_Size**: Size of payload in bytes
- **Payload_Size_Log**: Logarithm of payload size
- **IsLargePayload**: Binary indicator (top 5% of sizes)

**Importance**: Unusually large payloads may indicate data exfiltration or DDoS attacks.

### 4. Behavioral Features
- **SuspiciousCombo**: Protocol/request type mismatches
- **Status**: Request status (Success/Failure)
- **User_Agent**: Client user agent string

**Importance**: Behavioral anomalies often indicate malicious activity.

## Model-Specific Feature Importance

### Random Forest
Random Forest provides feature importance based on Gini impurity reduction:

**Top Features (typically)**:
1. Payload_Size / Payload_Size_Log
2. Port / IsWellKnownPort
3. Protocol type
4. Request_Type
5. Status (Success/Failure)
6. Temporal features (Hour, Weekday)

### Logistic Regression
Logistic Regression provides feature importance through coefficient magnitudes:

**Top Features (typically)**:
1. Payload_Size
2. Port
3. Protocol mismatches
4. Status
5. Request_Type

### XGBoost
XGBoost uses gain-based importance:

**Top Features (typically)**:
1. Payload_Size
2. Port
3. Protocol
4. Request_Type
5. Temporal features

### Isolation Forest (Unsupervised)
Isolation Forest identifies anomalies based on feature isolation:

**Top Features (typically)**:
1. Payload_Size (unusual sizes are isolated quickly)
2. Port (rare ports are isolated)
3. Protocol/Request_Type combinations
4. Temporal anomalies

## Feature Importance Visualization

The system generates feature importance charts showing:
1. **Bar charts**: Top N features by importance
2. **Heatmaps**: Correlation between features and attack types
3. **SHAP values**: Model-agnostic feature importance (if available)

## Interpretation Guidelines

### High Importance Features
- **Payload_Size**: Large payloads may indicate data exfiltration
- **Port**: Unusual ports may indicate port scanning or backdoors
- **Protocol/Request_Type mismatch**: Often indicates evasion attempts

### Medium Importance Features
- **Temporal features**: Help identify time-based attack patterns
- **Status**: High failure rates may indicate brute force attacks
- **User_Agent**: Suspicious user agents (e.g., Nikto, nmap) indicate scanning

### Low Importance Features
- **Source_IP/Destination_IP**: Important for tracking but less for detection
- **Minute**: Too granular for most attack patterns

## Feature Engineering Insights

### Effective Features
1. **Log-transformed payload size**: Better captures outliers
2. **Port categories**: More informative than raw port numbers
3. **Temporal aggregations**: Capture time-based patterns
4. **Protocol combinations**: Detect mismatches

### Less Effective Features
1. **Raw IP addresses**: Better to hash or encode
2. **Exact timestamps**: Better to extract temporal features
3. **High cardinality categoricals**: May cause overfitting

## Recommendations

1. **Monitor feature importance over time**: Attack patterns evolve
2. **Retrain models periodically**: Feature importance may shift
3. **Feature selection**: Use top K features to reduce overfitting
4. **Domain knowledge**: Combine ML importance with security expertise

## Example Feature Importance Output

```
Feature Importance (Random Forest):
1. Payload_Size_Log: 0.245
2. Port: 0.189
3. Protocol_TCP: 0.156
4. Request_Type_SSH: 0.134
5. Status_Success: 0.098
6. Hour: 0.067
7. IsWellKnownPort: 0.054
8. Weekday: 0.031
9. SuspiciousCombo: 0.028
10. User_Agent_nmap: 0.019
...
```

This shows that payload size and port are the most discriminative features for detecting intrusions.

