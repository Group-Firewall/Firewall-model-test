"""
Alert Notification System
Handles alert generation, prioritization, and notifications
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import os
from typing import Dict, List, Optional
import logging


class AlertSystem:
    """Alert notification and management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize alert system
        
        Args:
            config_path: Path to configuration file
        """
        self.alerts = []
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration"""
        default_config = {
            'email_enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email_from': '',
            'email_to': [],
            'email_password': '',
            'alert_threshold': 0.7,
            'high_severity_threshold': 0.9,
            'log_file': 'alerts.log'
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('NIDS_Alerts')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.config['log_file'])
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def create_alert(self, detection_result: Dict, network_data: Dict) -> Dict:
        """
        Create an alert from detection result
        
        Args:
            detection_result: Result from hybrid detector
            network_data: Original network log data
        
        Returns:
            Alert dictionary
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.alerts)}",
            'source_ip': network_data.get('Source_IP', 'Unknown'),
            'destination_ip': network_data.get('Destination_IP', 'Unknown'),
            'port': network_data.get('Port', 'Unknown'),
            'protocol': network_data.get('Protocol', 'Unknown'),
            'attack_detected': detection_result['is_attack'],
            'confidence': detection_result['confidence'],
            'signature_name': detection_result.get('signature_name', 'ML Detected'),
            'detection_method': 'Signature' if detection_result['signature_detected'] else 'ML',
            'severity': self._calculate_severity(detection_result),
            'status': 'new'
        }
        
        self.alerts.append(alert)
        self.logger.info(f"Alert created: {alert['alert_id']} - {alert['severity']} severity")
        
        # Send notification if threshold met
        if alert['confidence'] >= self.config['alert_threshold']:
            self._send_notification(alert)
        
        return alert
    
    def _calculate_severity(self, detection_result: Dict) -> str:
        """Calculate alert severity"""
        confidence = detection_result['confidence']
        
        if confidence >= self.config['high_severity_threshold']:
            return 'CRITICAL'
        elif confidence >= 0.8:
            return 'HIGH'
        elif confidence >= 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _send_notification(self, alert: Dict):
        """Send alert notification via email"""
        if not self.config['email_enabled']:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email_from']
            msg['To'] = ', '.join(self.config['email_to'])
            msg['Subject'] = f"NIDS Alert: {alert['severity']} - {alert['alert_id']}"
            
            body = f"""
Network Intrusion Detection System Alert

Alert ID: {alert['alert_id']}
Timestamp: {alert['timestamp']}
Severity: {alert['severity']}

Attack Details:
- Source IP: {alert['source_ip']}
- Destination IP: {alert['destination_ip']}
- Port: {alert['port']}
- Protocol: {alert['protocol']}
- Detection Method: {alert['detection_method']}
- Signature: {alert['signature_name']}
- Confidence: {alert['confidence']:.2%}

Please investigate this potential security threat immediately.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['email_from'], self.config['email_password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert['alert_id']}")
        
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    def get_alerts(self, severity: Optional[str] = None, 
                   status: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict]:
        """Get alerts with optional filtering"""
        filtered = self.alerts
        
        if severity:
            filtered = [a for a in filtered if a['severity'] == severity]
        
        if status:
            filtered = [a for a in filtered if a['status'] == status]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def update_alert_status(self, alert_id: str, status: str):
        """Update alert status"""
        for alert in self.alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = status
                self.logger.info(f"Alert {alert_id} status updated to {status}")
                return
        
        self.logger.warning(f"Alert {alert_id} not found")
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        if not self.alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_status': {},
                'by_detection_method': {}
            }
        
        stats = {
            'total_alerts': len(self.alerts),
            'by_severity': {},
            'by_status': {},
            'by_detection_method': {}
        }
        
        for alert in self.alerts:
            # Severity
            severity = alert['severity']
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
            
            # Status
            status = alert['status']
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            # Detection method
            method = alert['detection_method']
            stats['by_detection_method'][method] = stats['by_detection_method'].get(method, 0) + 1
        
        return stats
    
    def export_alerts(self, filepath: str, format: str = 'json'):
        """Export alerts to file"""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.alerts, f, indent=2)
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(self.alerts)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Alerts exported to {filepath}")
    
    def clear_alerts(self, older_than_days: Optional[int] = None):
        """Clear alerts, optionally only old ones"""
        if older_than_days:
            cutoff = datetime.now().timestamp() - (older_than_days * 24 * 3600)
            self.alerts = [
                a for a in self.alerts 
                if datetime.fromisoformat(a['timestamp']).timestamp() > cutoff
            ]
        else:
            self.alerts = []
        
        self.logger.info(f"Alerts cleared (older_than_days={older_than_days})")

