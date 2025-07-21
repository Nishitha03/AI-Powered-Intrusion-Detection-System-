# config.py - Configuration management for AI-IDS

import os
import json
from typing import Dict, Any

class IDSConfig:
    """Configuration management for IDS system"""
    
    def __init__(self, config_file: str = "ids_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return self.create_default_config()
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            "system": {
                "log_level": "INFO",
                "log_file": "ids.log",
                "data_dir": "data/",
                "models_dir": "models/",
                "alerts_dir": "alerts/"
            },
            "network": {
                "interface": None,  # Auto-detect
                "capture_filter": "",
                "packet_buffer_size": 1000,
                "monitoring_timeout": 1
            },
            "ml_models": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "random_state": 42
                },
                "xgboost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "random_state": 42
                },
                "autoencoder": {
                    "encoding_dim_ratio": 0.5,
                    "epochs": 50,
                    "batch_size": 32,
                    "validation_split": 0.2
                }
            },
            "alerts": {
                "max_alerts": 1000,
                "alert_threshold": 0.7,
                "email_notifications": False,
                "email_settings": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                }
            },
            "dashboard": {
                "port": 8501,
                "host": "localhost",
                "auto_refresh_interval": 2,
                "max_display_alerts": 50
            },
            "datasets": {
                "nsl_kdd": {
                    "train_file": "data/KDDTrain+.txt",
                    "test_file": "data/KDDTest+.txt"
                },
                "cic_ids2017": {
                    "data_dir": "data/cicids2017/"
                },
                "synthetic": {
                    "n_samples": 10000,
                    "attack_ratio": 0.2
                }
            }
        }
        
        # Save default config
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Dict[str, Any] = None):
        """Save configuration to file"""
        if config is None:
            config = self.config
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()

# deployment.py - Deployment utilities

import subprocess
import sys
import platform
from pathlib import Path

class IDSDeployer:
    """Deployment utilities for IDS system"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.config = IDSConfig()
    
    def check_dependencies(self) -> bool:
        """Check if all dependencies are installed"""
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'tensorflow',
            'scapy', 'streamlit', 'plotly', 'xgboost'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing packages: {', '.join(missing_packages)}")
            print("Please install using: pip install " + " ".join(missing_packages))
            return False
        
        return True
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.get('system.data_dir'),
            self.config.get('system.models_dir'),
            self.config.get('system.alerts_dir'),
            'logs/'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def check_network_permissions(self) -> bool:
        """Check if we have network capture permissions"""
        try:
            from scapy.all import get_if_list
            interfaces = get_if_list()
            
            if not interfaces:
                print("No network interfaces found!")
                return False
            
            print(f"Available interfaces: {', '.join(interfaces)}")
            
            # Test packet capture (requires privileges)
            if self.system == 'linux':
                result = subprocess.run(['id', '-nG'], capture_output=True, text=True)
                if 'wireshark' not in result.stdout and os.geteuid() != 0:
                    print("Warning: May need sudo privileges for packet capture")
                    print("Consider adding user to wireshark group or running with sudo")
            
            return True
            
        except Exception as e:
            print(f"Network permission check failed: {e}")
            return False
    
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        if self.system == 'linux':
            print("Installing Linux dependencies...")
            subprocess.run([
                'sudo', 'apt-get', 'update'
            ], check=False)
            subprocess.run([
                'sudo', 'apt-get', 'install', '-y',
                'python3-dev', 'libpcap-dev', 'tcpdump'
            ], check=False)
            
        elif self.system == 'darwin':  # macOS
            print("Installing macOS dependencies...")
            subprocess.run([
                'brew', 'install', 'libpcap'
            ], check=False)
            
        elif self.system == 'windows':
            print("For Windows, please install:")
            print("1. Npcap from: https://nmap.org/npcap/")
            print("2. Microsoft C++ Build Tools")
    
    def deploy(self, mode='development'):
        """Deploy the IDS system"""
        print(f"Deploying AI-IDS in {mode} mode...")
        
        # Check dependencies
        if not self.check_dependencies():
            print("Dependency check failed. Please install missing packages.")
            return False
        
        # Setup directories
        self.setup_directories()
        
        # Check network permissions
        self.check_network_permissions()
        
        # Create systemd service (Linux production mode)
        if mode == 'production' and self.system == 'linux':
            self.create_systemd_service()
        
        print("Deployment completed successfully!")
        print("\nNext steps:")
        print("1. Train models: python ids_system.py train")
        print("2. Start dashboard: streamlit run ids_system.py")
        print("3. Or start monitoring: python ids_system.py monitor")
        
        return True
    
    def create_systemd_service(self):
        """Create systemd service for production deployment"""
        service_content = f"""[Unit]
Description=AI-Powered Intrusion Detection System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={os.getcwd()}
ExecStart={sys.executable} ids_system.py monitor
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = '/etc/systemd/system/ai-ids.service'
        
        try:
            with open(service_file, 'w') as f:
                f.write(service_content)
            
            subprocess.run(['sudo', 'systemctl', 'daemon-reload'])
            subprocess.run(['sudo', 'systemctl', 'enable', 'ai-ids'])
            
            print(f"Systemd service created: {service_file}")
            print("Start with: sudo systemctl start ai-ids")
            print("Check status: sudo systemctl status ai-ids")
            
        except PermissionError:
            print("Need sudo privileges to create systemd service")

# monitoring.py - Enhanced monitoring capabilities

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import sqlite3
from typing import List, Dict

class AlertManager:
    """Enhanced alert management with database storage and notifications"""
    
    def __init__(self, config: IDSConfig):
        self.config = config
        self.db_path = "alerts.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for alert storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                src_ip TEXT,
                dst_ip TEXT,
                protocol TEXT,
                threat_level TEXT,
                packet_size INTEGER,
                alert_type TEXT,
                confidence REAL,
                raw_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert: Dict):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts 
            (timestamp, src_ip, dst_ip, protocol, threat_level, 
             packet_size, alert_type, confidence, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.get('timestamp'),
            alert.get('src_ip'),
            alert.get('dst_ip'),
            alert.get('protocol'),
            alert.get('threat_level'),
            alert.get('packet_size'),
            alert.get('alert_type', 'ML_DETECTION'),
            alert.get('confidence', 0.0),
            str(alert.get('raw_data', ''))
        ))
        
        conn.commit()
        conn.close()
        
        # Send notification if enabled
        if self.config.get('alerts.email_notifications'):
            self.send_email_alert(alert)
    
    def send_email_alert(self, alert: Dict):
        """Send email notification for high-priority alerts"""
        try:
            email_config = self.config.get('alerts.email_settings')
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['Subject'] = f"[AI-IDS] Security Alert - {alert['threat_level']}"
            
            body = f"""
            Security Alert Detected
            
            Time: {alert['timestamp']}
            Source IP: {alert['src_ip']}
            Destination IP: {alert['dst_ip']}
            Protocol: {alert['protocol']}
            Threat Level: {alert['threat_level']}
            Confidence: {alert.get('confidence', 'N/A')}
            
            Please investigate immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            for recipient in email_config['recipients']:
                msg['To'] = recipient
                server.send_message(msg)
            
            server.quit()
            
        except Exception as e:
            print(f"Failed to send email alert: {e}")
    
    def get_alerts(self, hours=24, threat_level=None) -> List[Dict]:
        """Retrieve alerts from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM alerts 
            WHERE timestamp > datetime('now', '-{} hours')
        '''.format(hours)
        
        params = []
        if threat_level:
            query += " AND threat_level = ?"
            params.append(threat_level)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        conn.close()
        
        # Convert to dictionaries
        columns = ['id', 'timestamp', 'src_ip', 'dst_ip', 'protocol', 
                  'threat_level', 'packet_size', 'alert_type', 'confidence', 'raw_data']
        
        return [dict(zip(columns, row)) for row in results]

# docker_deployment.py - Docker containerization

class DockerDeployer:
    """Docker deployment utilities"""
    
    def create_dockerfile(self):
        """Create Dockerfile for containerization"""
        dockerfile_content = '''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libpcap-dev \\
    tcpdump \\
    gcc \\
    python3-dev \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models alerts logs

# Expose port for Streamlit
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "ids_system.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''
        
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
    
    def create_docker_compose(self):
        """Create docker-compose.yml"""
        compose_content = '''version: '3.8'

services:
  ai-ids:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./alerts:/app/alerts
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    network_mode: host  # Required for packet capture
    cap_add:
      - NET_ADMIN
      - NET_RAW
    privileged: true
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana-storage:
'''
        
        with open('docker-compose.yml', 'w') as f:
            f.write(compose_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-IDS Deployment Utility")
    parser.add_argument('action', choices=['deploy', 'check', 'docker'], 
                       help='Deployment action')
    parser.add_argument('--mode', choices=['development', 'production'], 
                       default='development', help='Deployment mode')
    
    args = parser.parse_args()
    
    if args.action == 'deploy':
        deployer = IDSDeployer()
        deployer.deploy(args.mode)
        
    elif args.action == 'check':
        deployer = IDSDeployer()
        print("Checking dependencies...")
        deps_ok = deployer.check_dependencies()
        print("Checking network permissions...")
        net_ok = deployer.check_network_permissions()
        
        if deps_ok and net_ok:
            print("✓ System ready for deployment!")
        else:
            print("✗ Please resolve issues before deployment")
            
    elif args.action == 'docker':
        docker_deployer = DockerDeployer()
        print("Creating Docker files...")
        docker_deployer.create_dockerfile()
        docker_deployer.create_docker_compose()
        print("Docker files created! Build with: docker-compose up --build")