# Hybrid Fusion Algorithm

## Overview

The fusion algorithm combines decisions from signature-based detection and ML-based detection (both supervised and unsupervised) to make a final intrusion detection decision.

## Pseudocode

```
FUNCTION HybridFusion(signature_result, ml_result):
    INPUT:
        signature_result = {
            is_attack: boolean,
            confidence: float [0, 1],
            signature_name: string
        }
        ml_result = {
            is_attack: boolean,
            confidence: float [0, 1],
            supervised_pred: int,
            unsupervised_pred: int
        }
    
    OUTPUT:
        final_decision = {
            is_attack: boolean,
            confidence: float [0, 1],
            conflict: boolean,
            decision_path: string
        }
    
    BEGIN
        sig_attack = signature_result.is_attack
        sig_conf = signature_result.confidence
        ml_attack = ml_result.is_attack
        ml_conf = ml_result.confidence
        
        // High signature confidence - trust signature
        IF sig_conf > 0.8 THEN
            RETURN {
                is_attack: TRUE,
                confidence: sig_conf,
                conflict: FALSE,
                decision_path: "signature_high_confidence"
            }
        END IF
        
        // Medium signature confidence - use ML for confirmation
        IF sig_conf > 0.5 THEN
            IF ml_conf > 0.6 THEN
                // Both agree on attack
                fused_conf = (signature_weight * sig_conf) + 
                            (ml_weight * ml_conf)
                RETURN {
                    is_attack: TRUE,
                    confidence: fused_conf,
                    conflict: FALSE,
                    decision_path: "signature_ml_agreement"
                }
            ELSE
                // Conflict: signature says attack, ML says normal
                RETURN {
                    is_attack: FALSE,
                    confidence: 1 - sig_conf,
                    conflict: TRUE,
                    decision_path: "signature_ml_conflict"
                }
            END IF
        END IF
        
        // Low signature confidence - rely on ML
        IF ml_conf > 0.7 THEN
            RETURN {
                is_attack: TRUE,
                confidence: ml_conf,
                conflict: FALSE,
                decision_path: "ml_high_confidence"
            }
        ELSE
            RETURN {
                is_attack: FALSE,
                confidence: 1 - ml_conf,
                conflict: FALSE,
                decision_path: "ml_low_confidence"
            }
        END IF
    END FUNCTION
```

## Decision Flow Diagram

```
                    ┌─────────────────┐
                    │  Network Log    │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │  Signature       │      │  ML Detection    │
    │  Detection       │      │                  │
    └────────┬─────────┘      └────────┬─────────┘
             │                          │
             │                          │
    ┌────────┴─────────┐      ┌────────┴─────────┐
    │  Confidence      │      │  Confidence      │
    │  Score           │      │  Score           │
    └────────┬─────────┘      └────────┬─────────┘
             │                          │
             └──────────┬───────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Fusion Algorithm    │
            │                       │
            │  IF sig_conf > 0.8:   │
            │    → Trust Signature  │
            │  ELIF sig_conf > 0.5: │
            │    → Check ML         │
            │      IF ml_conf > 0.6:│
            │        → Attack        │
            │      ELSE:             │
            │        → Conflict      │
            │  ELSE:                 │
            │    → Trust ML          │
            │      IF ml_conf > 0.7:│
            │        → Attack        │
            │      ELSE:             │
            │        → Normal        │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Final Decision      │
            │  - is_attack          │
            │  - confidence         │
            │  - conflict flag      │
            └───────────────────────┘
```

## Weight Configuration

The fusion algorithm uses configurable weights:

- **signature_weight**: 0.3 (30%)
- **supervised_weight**: 0.5 (50%)
- **unsupervised_weight**: 0.2 (20%)

These weights can be adjusted based on:
- Historical performance of each component
- Network environment characteristics
- Security requirements (higher precision vs recall)

## Conflict Resolution

When signature and ML disagree:

1. **High signature confidence (>0.8)**: Always trust signature (fast response to known patterns)
2. **Medium signature confidence (0.5-0.8)**: 
   - If ML agrees (confidence >0.6): Flag as attack
   - If ML disagrees: Flag as normal (ML may detect false positive in signature)
3. **Low signature confidence (<0.5)**: Trust ML decision

## Confidence Calculation

Final confidence is calculated as:

```
IF signature_high_confidence:
    confidence = signature_confidence
ELIF signature_medium_confidence AND ml_agrees:
    confidence = (signature_weight * sig_conf) + (ml_weight * ml_conf)
ELSE:
    confidence = ml_confidence
```

## Example Scenarios

### Scenario 1: High Confidence Signature Match
- Signature: Attack detected, confidence = 0.95
- ML: Normal, confidence = 0.3
- **Decision**: Attack (trust signature)
- **Confidence**: 0.95
- **Path**: signature_high_confidence

### Scenario 2: Medium Signature, ML Agreement
- Signature: Attack detected, confidence = 0.65
- ML: Attack detected, confidence = 0.75
- **Decision**: Attack (both agree)
- **Confidence**: 0.70 (weighted average)
- **Path**: signature_ml_agreement

### Scenario 3: Medium Signature, ML Disagreement
- Signature: Attack detected, confidence = 0.60
- ML: Normal, confidence = 0.40
- **Decision**: Normal (trust ML, signature may be false positive)
- **Confidence**: 0.40
- **Path**: signature_ml_conflict

### Scenario 4: Low Signature, High ML Confidence
- Signature: Normal, confidence = 0.30
- ML: Attack detected, confidence = 0.85
- **Decision**: Attack (trust ML)
- **Confidence**: 0.85
- **Path**: ml_high_confidence

