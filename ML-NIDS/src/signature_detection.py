"""
Signature-Based Detection Module
Detects known attack patterns using rule-based signatures
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class SignatureDetector:
    """Signature-based intrusion detection using rule patterns"""
    
    def __init__(self):
        self.signatures = self._load_signatures()
        
    def _load_signatures(self) -> Dict[str, Dict]:
        """Load attack signatures"""
        return {
            'port_scan': {
                'description': 'Port scanning attack',
                'rules': [
                    lambda row: row['Scan_Type'] == 'PortScan',
                    lambda row: row['Port'] > 49152 and row['Status'] == 'Failure',
                    lambda row: row['Request_Type'] in ['SSH', 'FTP'] and row['Status'] == 'Failure'
                ],
                'severity': 'high',
                'weight': 0.9
            },
            'bot_attack': {
                'description': 'Botnet attack',
                'rules': [
                    lambda row: row['Scan_Type'] == 'BotAttack',
                    lambda row: 'Nikto' in str(row.get('User_Agent', '')),
                    lambda row: row['Payload_Size'] > 4000 and row['Status'] == 'Failure'
                ],
                'severity': 'critical',
                'weight': 1.0
            },
            'suspicious_port': {
                'description': 'Suspicious port activity',
                'rules': [
                    lambda row: row['Port'] in [135, 139, 445, 1433, 3389],  # Common attack ports
                    lambda row: row['Port'] < 1024 and row['Request_Type'] not in ['HTTP', 'HTTPS', 'FTP', 'SSH']
                ],
                'severity': 'medium',
                'weight': 0.7
            },
            'protocol_mismatch': {
                'description': 'Protocol and request type mismatch',
                'rules': [
                    lambda row: (row['Request_Type'] == 'SSH' and row['Protocol'] == 'ICMP'),
                    lambda row: (row['Request_Type'] == 'HTTP' and row['Protocol'] == 'ICMP'),
                    lambda row: (row['Request_Type'] == 'HTTPS' and row['Protocol'] == 'UDP')
                ],
                'severity': 'medium',
                'weight': 0.6
            },
            'large_payload': {
                'description': 'Suspiciously large payload',
                'rules': [
                    lambda row: row['Payload_Size'] > 5000,
                    lambda row: row['Payload_Size'] > 3000 and row['Status'] == 'Failure'
                ],
                'severity': 'low',
                'weight': 0.5
            },
            'rapid_failure': {
                'description': 'Rapid connection failures',
                'rules': [
                    lambda row: row['Status'] == 'Failure' and row['Request_Type'] in ['SSH', 'FTP']
                ],
                'severity': 'medium',
                'weight': 0.6
            }
        }
    
    def detect(self, row: pd.Series) -> Tuple[bool, float, str]:
        """
        Detect if a row matches any signature
        
        Returns:
            (is_attack, confidence, signature_name)
        """
        max_confidence = 0.0
        matched_signature = None
        
        for sig_name, sig_config in self.signatures.items():
            # Check if any rule matches
            matches = sum(1 for rule in sig_config['rules'] if self._safe_apply(rule, row))
            
            if matches > 0:
                # Confidence based on number of matching rules and signature weight
                confidence = min(1.0, (matches / len(sig_config['rules'])) * sig_config['weight'])
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    matched_signature = sig_name
        
        is_attack = max_confidence > 0.5
        return is_attack, max_confidence, matched_signature or 'none'
    
    def _safe_apply(self, rule, row):
        """Safely apply a rule function"""
        try:
            return rule(row)
        except (KeyError, TypeError, AttributeError):
            return False
    
    def detect_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect attacks in a batch of data"""
        results = []
        
        for idx, row in df.iterrows():
            is_attack, confidence, signature = self.detect(row)
            results.append({
                'signature_detected': is_attack,
                'signature_confidence': confidence,
                'signature_name': signature
            })
        
        return pd.DataFrame(results)
    
    def add_signature(self, name: str, rules: List, description: str, 
                     severity: str = 'medium', weight: float = 0.7):
        """Add a custom signature"""
        self.signatures[name] = {
            'description': description,
            'rules': rules,
            'severity': severity,
            'weight': weight
        }
    
    def get_signature_info(self, name: str) -> Dict:
        """Get information about a signature"""
        return self.signatures.get(name, {})

