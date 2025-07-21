# AI-Powered Intrusion Detection System (IDS)
# Complete implementation with ML models, real-time monitoring, and visualization

import pandas as pd
import numpy as np
import pickle
import logging
import threading
import time
import json
from datetime import datetime
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Network Analysis
from scapy.all import *
import socket
import psutil

# Visualization & Dashboard
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing for IDS datasets"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def load_nsl_kdd(self, train_path='KDDTrain+.txt', test_path='KDDTest+.txt'):
        """Load and preprocess NSL-KDD dataset"""
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
        ]
        
        try:
            train_df = pd.read_csv(train_path, names=columns)
            test_df = pd.read_csv(test_path, names=columns)
            
            # Combine datasets
            df = pd.concat([train_df, test_df], ignore_index=True)
            
            # Binary classification: Normal vs Attack
            df['label'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
            
            return df
        except FileNotFoundError:
            logger.error("NSL-KDD files not found. Generating synthetic data...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic network traffic data for demonstration"""
        np.random.seed(42)
        
        data = {
            'duration': np.random.exponential(1, n_samples),
            'src_bytes': np.random.exponential(1000, n_samples),
            'dst_bytes': np.random.exponential(1000, n_samples),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(5, n_samples),
            'same_srv_rate': np.random.uniform(0, 1, n_samples),
            'diff_srv_rate': np.random.uniform(0, 1, n_samples),
            'dst_host_count': np.random.poisson(20, n_samples),
            'dst_host_srv_count': np.random.poisson(10, n_samples),
            'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'telnet'], n_samples),
            'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTR'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create labels (20% attacks)
        df['label'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        
        # Make attacks have different characteristics
        attack_mask = df['label'] == 1
        df.loc[attack_mask, 'src_bytes'] *= np.random.uniform(5, 50, attack_mask.sum())
        df.loc[attack_mask, 'count'] *= np.random.uniform(2, 10, attack_mask.sum())
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the dataset for ML models"""
        df = df.copy()
        
        # Handle categorical variables
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = set(df[col].astype(str).unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unseen_values = unique_values - known_values
                    
                    if unseen_values:
                        # Add unseen values to encoder
                        new_classes = np.append(self.label_encoders[col].classes_, list(unseen_values))
                        self.label_encoders[col].classes_ = new_classes
                    
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Select numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numerical_cols:
            numerical_cols.remove('label')
        if 'difficulty' in numerical_cols:
            numerical_cols.remove('difficulty')
        if 'attack_type' in df.columns:
            numerical_cols.remove('attack_type') if 'attack_type' in numerical_cols else None
        
        self.feature_columns = numerical_cols
        
        # Handle missing values
        df[numerical_cols] = df[numerical_cols].fillna(0)
        
        X = df[numerical_cols]
        y = df['label'] if 'label' in df.columns else None
        
        return X, y

class MLModels:
    """Machine Learning models for intrusion detection"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor()
    
    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None):
        """Train Random Forest classifier"""
        logger.info("Training Random Forest model...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        if X_test is not None and y_test is not None:
            y_pred = rf_model.predict(X_test)
            logger.info(f"Random Forest Test Accuracy: {rf_model.score(X_test, y_test):.4f}")
            print(classification_report(y_test, y_pred))
        
        return rf_model
    
    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None):
        """Train XGBoost classifier"""
        logger.info("Training XGBoost model...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        if X_test is not None and y_test is not None:
            y_pred = xgb_model.predict(X_test)
            logger.info(f"XGBoost Test Accuracy: {xgb_model.score(X_test, y_test):.4f}")
            print(classification_report(y_test, y_pred))
        
        return xgb_model
    
    def train_autoencoder(self, X_train, X_test=None, contamination=0.1):
        """Train Deep Autoencoder for anomaly detection"""
        logger.info("Training Deep Autoencoder...")
        
        input_dim = X_train.shape[1]
        encoding_dim = input_dim // 2
        
        # Build autoencoder
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation='relu')(input_layer)
        encoder = Dense(encoding_dim // 2, activation='relu')(encoder)
        encoder = Dense(encoding_dim // 4, activation='relu')(encoder)
        
        decoder = Dense(encoding_dim // 2, activation='relu')(encoder)
        decoder = Dense(encoding_dim, activation='relu')(decoder)
        decoder = Dense(input_dim, activation='sigmoid')(decoder)
        
        autoencoder = Model(input_layer, decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train on normal data only
        normal_data = X_train  # Assume most data is normal
        
        history = autoencoder.fit(
            normal_data, normal_data,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(patience=5)],
            verbose=0
        )
        
        self.models['autoencoder'] = autoencoder
        
        # Calculate threshold for anomaly detection
        train_pred = autoencoder.predict(normal_data)
        train_loss = np.mean(np.square(normal_data - train_pred), axis=1)
        threshold = np.percentile(train_loss, 95)
        self.models['autoencoder_threshold'] = threshold
        
        logger.info(f"Autoencoder trained. Anomaly threshold: {threshold:.4f}")
        
        return autoencoder
    
    def predict(self, X, model_name='random_forest'):
        """Make predictions using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if model_name == 'autoencoder':
            # Anomaly detection using reconstruction error
            pred = model.predict(X)
            mse = np.mean(np.square(X - pred), axis=1)
            threshold = self.models['autoencoder_threshold']
            return (mse > threshold).astype(int)
        else:
            return model.predict(X)
    
    def save_models(self, path='models/'):
        """Save trained models"""
        import os
        import joblib
        os.makedirs(path, exist_ok=True)
        
        try:
            for name, model in self.models.items():
                if name == 'autoencoder':
                    model.save(f'{path}{name}.h5')
                elif name != 'autoencoder_threshold':
                    # Use joblib instead of pickle for better compatibility
                    joblib.dump(model, f'{path}{name}.pkl')
            
            # Save preprocessor components separately to avoid pickle issues
            if hasattr(self.preprocessor, 'label_encoders'):
                joblib.dump(self.preprocessor.label_encoders, f'{path}label_encoders.pkl')
            if hasattr(self.preprocessor, 'scaler'):
                joblib.dump(self.preprocessor.scaler, f'{path}scaler.pkl')
            if hasattr(self.preprocessor, 'feature_columns'):
                joblib.dump(self.preprocessor.feature_columns, f'{path}feature_columns.pkl')
            
            if 'autoencoder_threshold' in self.models:
                joblib.dump(self.models['autoencoder_threshold'], f'{path}autoencoder_threshold.pkl')
                
            logger.info("Models saved successfully!")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            # Continue without saving if there's an error
            pass

class NetworkMonitor:
    """Real-time network packet monitoring using Scapy"""
    
    def __init__(self, ml_models, interface=None):
        self.ml_models = ml_models
        self.interface = interface
        self.packet_buffer = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.is_monitoring = False
        self.stats = defaultdict(int)
    
    def packet_handler(self, packet):
        """Process captured packets"""
        try:
            features = self.extract_features(packet)
            if features:
                # Convert to format expected by ML model
                feature_vector = self.features_to_vector(features)
                
                # Predict using ML model
                prediction = self.ml_models.predict(
                    np.array([feature_vector]), 
                    model_name='random_forest'
                )[0]
                
                # Generate alert if anomalous
                if prediction == 1:
                    alert = {
                        'timestamp': datetime.now(),
                        'src_ip': features.get('src_ip', 'Unknown'),
                        'dst_ip': features.get('dst_ip', 'Unknown'),
                        'protocol': features.get('protocol', 'Unknown'),
                        'threat_level': 'HIGH',
                        'packet_size': features.get('packet_size', 0)
                    }
                    self.alerts.append(alert)
                    logger.warning(f"ALERT: Suspicious activity detected from {features.get('src_ip')}")
                
                # Update statistics
                self.stats['total_packets'] += 1
                self.stats['alerts'] = len(self.alerts)
                
                # Store packet info
                self.packet_buffer.append({
                    'timestamp': datetime.now(),
                    'features': features,
                    'prediction': prediction
                })
                
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    def extract_features(self, packet):
        """Extract features from network packet"""
        features = {}
        
        try:
            # Basic packet info
            features['packet_size'] = len(packet)
            features['timestamp'] = time.time()
            
            # IP layer
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                features['src_ip'] = ip_layer.src
                features['dst_ip'] = ip_layer.dst
                features['protocol'] = ip_layer.proto
                features['ttl'] = ip_layer.ttl
                features['ip_len'] = ip_layer.len
            
            # TCP layer
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                features['src_port'] = tcp_layer.sport
                features['dst_port'] = tcp_layer.dport
                features['tcp_flags'] = tcp_layer.flags
                features['tcp_window'] = tcp_layer.window
                features['protocol_type'] = 'tcp'
            
            # UDP layer
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                features['src_port'] = udp_layer.sport
                features['dst_port'] = udp_layer.dport
                features['protocol_type'] = 'udp'
            
            # ICMP layer
            elif packet.haslayer(ICMP):
                features['protocol_type'] = 'icmp'
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def features_to_vector(self, features):
        """Convert packet features to ML model input vector"""
        # This should match the feature format used in training
        vector = [
            features.get('packet_size', 0),
            features.get('src_port', 0),
            features.get('dst_port', 0),
            features.get('tcp_window', 0),
            features.get('ip_len', 0),
            features.get('ttl', 64),
            1 if features.get('protocol_type') == 'tcp' else 0,
            1 if features.get('protocol_type') == 'udp' else 0,
            1 if features.get('protocol_type') == 'icmp' else 0,
            features.get('tcp_flags', 0) if features.get('tcp_flags') else 0
        ]
        
        # Pad or truncate to match training features
        expected_length = len(self.ml_models.preprocessor.feature_columns)
        if len(vector) < expected_length:
            vector.extend([0] * (expected_length - len(vector)))
        elif len(vector) > expected_length:
            vector = vector[:expected_length]
        
        return vector
    
    def start_monitoring(self):
        """Start real-time packet capture"""
        self.is_monitoring = True
        logger.info("Starting network monitoring...")
        
        try:
            sniff(
                iface=self.interface,
                prn=self.packet_handler,
                store=False,
                stop_filter=lambda x: not self.is_monitoring
            )
        except Exception as e:
            logger.error(f"Error in packet capture: {e}")
    
    def stop_monitoring(self):
        """Stop packet capture"""
        self.is_monitoring = False
        logger.info("Stopped network monitoring")
    
    def get_recent_alerts(self, n=10):
        """Get recent security alerts"""
        return list(self.alerts)[-n:]
    
    def get_stats(self):
        """Get monitoring statistics"""
        return dict(self.stats)

class IDSVisualization:
    """Streamlit-based dashboard for IDS visualization"""
    
    def __init__(self, network_monitor):
        self.monitor = network_monitor
    
    def create_dashboard(self):
        """Create Streamlit dashboard"""
        st.set_page_config(page_title="AI-Powered IDS Dashboard", layout="wide")
        
        st.title("🛡️ AI-Powered Intrusion Detection System")
        st.markdown("Real-time network security monitoring with machine learning")
        
        # Sidebar controls
        st.sidebar.title("Controls")
        
        if st.sidebar.button("Start Monitoring"):
            if not self.monitor.is_monitoring:
                # Start monitoring in separate thread
                monitor_thread = threading.Thread(target=self.monitor.start_monitoring)
                monitor_thread.daemon = True
                monitor_thread.start()
                st.sidebar.success("Monitoring started!")
        
        if st.sidebar.button("Stop Monitoring"):
            self.monitor.stop_monitoring()
            st.sidebar.info("Monitoring stopped")
        
        # Main dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        stats = self.monitor.get_stats()
        
        with col1:
            st.metric("Total Packets", stats.get('total_packets', 0))
        
        with col2:
            st.metric("Active Alerts", stats.get('alerts', 0))
        
        with col3:
            st.metric("Monitoring Status", 
                     "ACTIVE" if self.monitor.is_monitoring else "STOPPED")
        
        with col4:
            st.metric("Threat Level", "HIGH" if stats.get('alerts', 0) > 5 else "LOW")
        
        # Alerts section
        st.header("🚨 Recent Security Alerts")
        alerts = self.monitor.get_recent_alerts()
        
        if alerts:
            alert_df = pd.DataFrame(alerts)
            st.dataframe(alert_df, use_container_width=True)
            
            # Alert timeline
            if len(alerts) > 1:
                fig = px.scatter(alert_df, x='timestamp', y='src_ip', 
                               size='packet_size', color='threat_level',
                               title="Alert Timeline")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alerts detected")
        
        # Network traffic visualization
        st.header("📊 Network Traffic Analysis")
        
        if len(self.monitor.packet_buffer) > 0:
            # Convert packet buffer to DataFrame
            packet_data = []
            for packet_info in list(self.monitor.packet_buffer)[-100:]:  # Last 100 packets
                packet_data.append({
                    'timestamp': packet_info['timestamp'],
                    'src_ip': packet_info['features'].get('src_ip', 'Unknown'),
                    'dst_ip': packet_info['features'].get('dst_ip', 'Unknown'),
                    'protocol': packet_info['features'].get('protocol_type', 'Unknown'),
                    'size': packet_info['features'].get('packet_size', 0),
                    'prediction': 'Attack' if packet_info['prediction'] == 1 else 'Normal'
                })
            
            if packet_data:
                df = pd.DataFrame(packet_data)
                
                # Traffic over time
                fig = px.line(df, x='timestamp', y='size', color='prediction',
                             title="Network Traffic Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Protocol distribution
                protocol_counts = df['protocol'].value_counts()
                fig = px.pie(values=protocol_counts.values, names=protocol_counts.index,
                           title="Protocol Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No network traffic data available. Start monitoring to see real-time data.")
        
        # Auto-refresh
        time.sleep(2)
        st.rerun()

def main():
    """Main IDS application"""
    st.set_page_config(page_title="AI-Powered IDS", layout="wide")
    
    # Initialize components
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = MLModels()
        st.session_state.data_loaded = False
        st.session_state.models_trained = False
    
    if 'network_monitor' not in st.session_state:
        st.session_state.network_monitor = NetworkMonitor(st.session_state.ml_models)
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Navigation", 
                               ["Dashboard", "Model Training", "Dataset Analysis"])
    
    if page == "Model Training":
        st.title("🤖 Machine Learning Model Training")
        
        if st.button("Load and Prepare Data"):
            with st.spinner("Loading dataset..."):
                # Load and preprocess data
                preprocessor = DataPreprocessor()
                df = preprocessor.load_nsl_kdd()
                X, y = preprocessor.preprocess_data(df)
                
                # Store in session state
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.ml_models.preprocessor = preprocessor
                st.session_state.data_loaded = True
                
                st.success(f"Data loaded successfully! Shape: {X.shape}")
                st.write("Dataset preview:")
                st.dataframe(df.head())
        
        if st.session_state.data_loaded:
            if st.button("Train All Models"):
                with st.spinner("Training models..."):
                    X = st.session_state.X
                    y = st.session_state.y
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    st.session_state.ml_models.preprocessor.scaler = scaler
                    
                    # Train models
                    st.info("Training Random Forest...")
                    st.session_state.ml_models.train_random_forest(
                        X_train_scaled, y_train, X_test_scaled, y_test
                    )
                    
                    st.info("Training XGBoost...")
                    st.session_state.ml_models.train_xgboost(
                        X_train_scaled, y_train, X_test_scaled, y_test
                    )
                    
                    st.info("Training Autoencoder...")
                    st.session_state.ml_models.train_autoencoder(
                        X_train_scaled, X_test_scaled
                    )
                    
                    st.session_state.models_trained = True
                    st.success("All models trained successfully!")
                    
                    # Save models (with error handling)
                    try:
                        st.session_state.ml_models.save_models()
                        st.info("Models saved to disk")
                    except Exception as e:
                        st.warning(f"Models trained but couldn't save to disk: {e}")
                        st.info("You can continue using the system - models are loaded in memory")
    
    elif page == "Dataset Analysis":
        st.title("📊 Dataset Analysis")
        
        if st.session_state.data_loaded:
            df_sample = pd.DataFrame(st.session_state.X).head(1000)  # Sample for visualization
            df_sample['label'] = st.session_state.y[:1000]
            
            # Dataset statistics
            st.subheader("Dataset Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Samples", len(st.session_state.X))
                st.metric("Features", st.session_state.X.shape[1])
            
            with col2:
                normal_count = (st.session_state.y == 0).sum()
                attack_count = (st.session_state.y == 1).sum()
                st.metric("Normal Traffic", normal_count)
                st.metric("Attack Traffic", attack_count)
            
            # Class distribution
            labels = ['Normal', 'Attack']
            values = [normal_count, attack_count]
            fig = px.pie(values=values, names=labels, title="Class Distribution")
            st.plotly_chart(fig)
            
            # Feature correlations
            st.subheader("Feature Correlations")
            corr_matrix = df_sample.corr()
            fig = px.imshow(corr_matrix, title="Feature Correlation Matrix")
            st.plotly_chart(fig)
        
        else:
            st.info("Please load data first in the Model Training section")
    
    else:  # Dashboard
        if st.session_state.models_trained:
            dashboard = IDSVisualization(st.session_state.network_monitor)
            dashboard.create_dashboard()
        else:
            st.title("🛡️ AI-Powered Intrusion Detection System")
            st.info("Please train models first before using the dashboard")
            
            st.markdown("""
            ## Quick Start Guide:
            1. Go to **Model Training** section
            2. Click **Load and Prepare Data** to load the dataset
            3. Click **Train All Models** to train ML models
            4. Return to **Dashboard** to start real-time monitoring
            
            ## Features:
            - **Multiple ML Models**: Random Forest, XGBoost, Deep Autoencoder
            - **Real-time Monitoring**: Live packet capture and analysis
            - **Interactive Dashboard**: Real-time visualizations and alerts
            - **Threat Detection**: Automatic anomaly detection and alerting
            """)

if __name__ == "__main__":
    # For standalone script execution
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        print("Training AI-IDS models...")
        
        # Initialize components
        ml_models = MLModels()
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        df = preprocessor.load_nsl_kdd()
        X, y = preprocessor.preprocess_data(df)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ml_models.preprocessor = preprocessor
        ml_models.preprocessor.scaler = scaler
        
        # Train models
        ml_models.train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
        ml_models.train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)
        ml_models.train_autoencoder(X_train_scaled, X_test_scaled)
        
        # Save models
        ml_models.save_models()
        print("Models trained and saved successfully!")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "monitor":
        # Monitoring mode (requires trained models)
        print("Starting network monitoring...")
        
        # Load trained models
        ml_models = MLModels()
        # Load your trained models here
        
        monitor = NetworkMonitor(ml_models)
        
        try:
            monitor.start_monitoring()
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("Monitoring stopped")
    
    else:
        # Streamlit dashboard mode
        main()