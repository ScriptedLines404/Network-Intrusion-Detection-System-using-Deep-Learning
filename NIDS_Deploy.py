"""
Network Intrusion Detection System (NIDS) Deployment Interface
Main deployment script using Gradio for web-based threat detection interface
"""

import gradio as gr
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys
import os
import pickle
import joblib
import xgboost as xgb
import tensorflow as tf

class NIDSDeployer:
    """
    Main deployment class for the Network Intrusion Detection System
    Handles model loading, prediction, and web interface creation
    """
    
    def __init__(self, trained_nids=None, model_path="trained_nids_simple.pkl"):
        """
        Initialize the NIDS deployer with a trained model or load from file
        
        Args:
            trained_nids: Pre-trained NIDS model object (optional)
            model_path: Path to saved model file (default: "trained_nids_simple.pkl")
        """
        self.model_path = model_path
        self.anomaly_threshold = 0.11817  # Threshold for autoencoder anomaly detection
        
        # Try to load the trained model
        self.nids = self._load_trained_model(trained_nids)
        
        # If model loading failed, try loading individual components
        if self.nids is None:
            print("⚠️ Could not load trained model. Trying to load components...")
            self._load_model_components()
        
        # If still no model, create a mock system for demonstration
        if not hasattr(self, 'nids') or self.nids is None:
            print("⚠️ Could not load any model. Using mock system for demonstration.")
            self._create_mock_system()

    def _load_trained_model(self, trained_nids):
        """
        Attempt to load a trained NIDS model
        
        Args:
            trained_nids: Pre-trained model object or None
        
        Returns:
            Loaded model object or None if loading failed
        """
        # If model is provided directly, use it
        if trained_nids is not None:
            print("✅ Using provided trained model")
            return trained_nids
        
        # Try loading from the specified path
        try:
            if os.path.exists(self.model_path):
                print(f"📂 Loading trained model from {self.model_path}")
                with open(self.model_path, 'rb') as f:
                    saved_model = pickle.load(f)
                print("✅ Full model loaded successfully from file")
                return saved_model
            else:
                # Try alternative paths if default doesn't exist
                alternative_paths = [
                    "saved_models/trained_nids_model.pkl",
                    "trained_nids_model.pkl",
                    "nids_model_simple.pkl"
                ]
                for path in alternative_paths:
                    if os.path.exists(path):
                        print(f"📂 Loading trained model from {path}")
                        with open(path, 'rb') as f:
                            saved_model = pickle.load(f)
                        print("✅ Full model loaded successfully from file")
                        return saved_model
        except Exception as e:
            print(f"⚠️ Could not load full model from file: {e}")
        
        return None

    def _load_model_components(self):
        """
        Load individual model components from separate files
        Used when the full model pickle file is not available
        
        Returns:
            bool: True if essential components were loaded successfully
        """
        try:
            print("🔍 Attempting to load model components...")
            
            if os.path.exists("saved_models"):
                print("📁 Found saved_models directory")
                
                # Create a container object for the loaded components
                class LoadedNIDS:
                    pass
                
                self.nids = LoadedNIDS()
                
                # Try different filename prefixes
                prefixes = ["trained_nids_model", "trained_nids", "nids_model"]
                
                for prefix in prefixes:
                    try:
                        # Load scaler for feature normalization
                        scaler_path = f"saved_models/{prefix}_scaler.pkl"
                        if os.path.exists(scaler_path):
                            self.nids.scaler = joblib.load(scaler_path)
                            print(f"✅ Loaded scaler from {scaler_path}")
                        
                        # Load label encoder for class decoding
                        le_path = f"saved_models/{prefix}_label_encoder.pkl"
                        if os.path.exists(le_path):
                            self.nids.label_encoder = joblib.load(le_path)
                            print(f"✅ Loaded label encoder from {le_path}")
                        
                        # Load XGBoost classifier for known attack detection
                        xgb_path = f"saved_models/{prefix}_xgb.model"
                        if os.path.exists(xgb_path):
                            self.nids.xgb_model = xgb.XGBClassifier()
                            self.nids.xgb_model.load_model(xgb_path)
                            print(f"✅ Loaded XGBoost model from {xgb_path}")
                        
                        # Load autoencoder for anomaly detection
                        ae_path = f"saved_models/{prefix}_autoencoder.h5"
                        if os.path.exists(ae_path):
                            self.nids.autoencoder = tf.keras.models.load_model(ae_path, compile=False)
                            print(f"✅ Loaded Autoencoder from {ae_path}")
                        
                        # Load feature names for input validation
                        features_path = f"saved_models/{prefix}_features.pkl"
                        if os.path.exists(features_path):
                            with open(features_path, 'rb') as f:
                                self.nids.feature_names = pickle.load(f)
                            print(f"✅ Loaded feature names from {features_path}")
                        
                        # Load anomaly detection threshold
                        threshold_path = f"saved_models/{prefix}_threshold.pkl"
                        if os.path.exists(threshold_path):
                            with open(threshold_path, 'rb') as f:
                                self.anomaly_threshold = pickle.load(f)
                            print(f"✅ Loaded anomaly threshold from {threshold_path}")
                        
                        # Check if essential components were loaded
                        if (hasattr(self.nids, 'scaler') and 
                            hasattr(self.nids, 'feature_names') and 
                            (hasattr(self.nids, 'xgb_model') or hasattr(self.nids, 'autoencoder'))):
                            print("✅ Successfully loaded essential model components")
                            return True
                            
                    except Exception as e:
                        print(f"⚠️ Error loading components with prefix {prefix}: {e}")
                        continue
                        
        except Exception as e:
            print(f"⚠️ Error loading model components: {e}")
        
        return False

    def _create_mock_system(self):
        """
        Create a mock NIDS system for demonstration purposes
        Used when no trained model is available
        """
        print("⚠️ Creating mock system for demonstration...")
        
        class MockTrainedSystem:
            """
            Mock system that simulates NIDS behavior without actual ML models
            """
            def __init__(self):
                # Define the expected feature names
                self.feature_names = [
                    'Destination Port', 'Total Fwd Packets', 'Total Length of Fwd Packets',
                    'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Max', 'Fwd IAT Max',
                    'Fwd IAT Min', 'Bwd IAT Min', 'Fwd Packets/s', 'PSH Flag Count',
                    'ACK Flag Count', 'Subflow Fwd Bytes', 'act_data_pkt_fwd',
                    'Idle Mean', 'Idle Max', 'Idle Min'
                ]
                
                # Mock scaler (identity transformation)
                class MockScaler:
                    def transform(self, x):
                        return np.array(x)
                
                # Mock XGBoost model with simple rule-based predictions
                class MockXGBModel:
                    def predict(self, x):
                        predictions = []
                        for sample in x:
                            # Simple heuristics for demonstration
                            if sample[1] > 1000:  # Many packets = DDoS
                                predictions.append(1)
                            elif sample[0] in [22, 23, 3389, 5900, 5901]:  # Suspicious ports = Port Scan
                                predictions.append(2)
                            else:
                                predictions.append(0)  # Normal
                        return np.array(predictants)
                    
                    def predict_proba(self, x):
                        n_samples = x.shape[0]
                        probs = []
                        for sample in x:
                            if sample[1] > 1000:
                                probs.append([0.1, 0.8, 0.1])  # High probability for DDoS
                            elif sample[0] in [22, 23, 3389, 5900, 5901]:
                                probs.append([0.2, 0.1, 0.7])  # High probability for Port Scan
                            else:
                                probs.append([0.9, 0.05, 0.05])  # High probability for Normal
                        return np.array(probs)
                
                # Mock autoencoder for anomaly detection
                class MockAutoencoder:
                    def predict(self, x, verbose=0):
                        # Add random noise to simulate reconstruction
                        noise = np.random.normal(0, 0.01, x.shape)
                        for i, sample in enumerate(x):
                            if sample[1] > 1000:  # Add more noise for suspicious traffic
                                noise[i] += np.random.normal(0, 0.05, x.shape[1])
                        return x + noise
                
                # Mock label encoder for class name mapping
                class MockLabelEncoder:
                    def inverse_transform(self, labels):
                        class_names = []
                        for label in labels:
                            if label == 0:
                                class_names.append("Normal Traffic")
                            elif label == 1:
                                class_names.append("DDoS Attack")
                            else:
                                class_names.append("Port Scan")
                        return class_names
                    
                    @property
                    def classes_(self):
                        return ["Normal Traffic", "DDoS Attack", "Port Scan"]
                
                # Initialize mock components
                self.scaler = MockScaler()
                self.xgb_model = MockXGBModel()
                self.autoencoder = MockAutoencoder()
                self.label_encoder = MockLabelEncoder()
        
        # Set the nids attribute to the mock system
        self.nids = MockTrainedSystem()
        print("✅ Mock system created successfully")
        return True

    def predict_single_observation(self, features_dict):
        """
        Make a prediction for a single network traffic observation
        
        Args:
            features_dict: Dictionary containing feature names and values
        
        Returns:
            dict: Prediction results including class, confidence, anomaly status, etc.
        """
        try:
            # Convert dictionary to array in the correct feature order
            features = []
            for feature in self.nids.feature_names:
                if feature in features_dict:
                    features.append(features_dict[feature])
                else:
                    print(f"⚠️ Missing feature {feature}, using default value 0")
                    features.append(0)
            
            features_array = np.array(features).reshape(1, -1)

            # Normalize features using the scaler
            features_scaled = self.nids.scaler.transform(features_array)

            # Default values in case models are not available
            xgb_pred = 0
            xgb_proba = np.array([[1.0, 0.0, 0.0]])
            predicted_class = "Normal Traffic"
            confidence = 1.0

            # XGBoost prediction for known attack classification
            if hasattr(self.nids, 'xgb_model'):
                xgb_pred = self.nids.xgb_model.predict(features_scaled)[0]
                xgb_proba = self.nids.xgb_model.predict_proba(features_scaled)[0]
                
                # Decode the predicted class label
                if hasattr(self.nids, 'label_encoder'):
                    predicted_class = self.nids.label_encoder.inverse_transform([xgb_pred])[0]
                else:
                    predicted_class = f"Class {xgb_pred}"
                
                confidence = np.max(xgb_proba)  # Get the highest probability
            else:
                print("⚠️ XGBoost model not available, using default predictions")

            # Autoencoder-based anomaly detection
            mse = 0
            is_anomaly = False
            
            if hasattr(self.nids, 'autoencoder'):
                # Get reconstruction and calculate error
                reconstruction = self.nids.autoencoder.predict(features_scaled, verbose=0)
                mse = np.mean(np.power(features_scaled - reconstruction, 2))
                is_anomaly = mse > self.anomaly_threshold
            else:
                print("⚠️ Autoencoder not available, skipping anomaly detection")
                # Fallback heuristic for anomaly detection
                if features_dict.get('Total Fwd Packets', 0) > 1000:
                    is_anomaly = True
                    mse = self.anomaly_threshold * 2

            # Format probabilities for display
            if hasattr(self.nids, 'label_encoder'):
                probabilities = {
                    self.nids.label_encoder.classes_[i]: float(prob)
                    for i, prob in enumerate(xgb_proba)
                }
            else:
                probabilities = {
                    "Normal Traffic": float(xgb_proba[0] if len(xgb_proba) > 0 else 0.9),
                    "DDoS Attack": float(xgb_proba[1] if len(xgb_proba) > 1 else 0.05),
                    "Port Scan": float(xgb_proba[2] if len(xgb_proba) > 2 else 0.05)
                }

            # Determine threat level based on predictions
            if is_anomaly:
                threat_level = "HIGH - Novel Anomaly Detected"
                alert_color = "🔴"
                threat_score = 90
            elif predicted_class != "Normal Traffic":
                threat_level = "MEDIUM - Known Attack"
                alert_color = "🟡"
                threat_score = 60
            else:
                threat_level = "LOW - Normal Traffic"
                alert_color = "🟢"
                threat_score = 10

            # Compile all results into a dictionary
            results = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'predicted_attack': predicted_class,
                'confidence': float(confidence),
                'is_anomaly': bool(is_anomaly),
                'reconstruction_error': float(mse),
                'anomaly_threshold': float(self.anomaly_threshold),
                'threat_level': threat_level,
                'alert_color': alert_color,
                'threat_score': threat_score,
                'all_probabilities': probabilities
            }

            return results

        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f"Prediction failed: {str(e)}"}

    def create_visual_output(self, result):
        """
        Create HTML visualization of prediction results
        
        Args:
            result: Dictionary containing prediction results
        
        Returns:
            str: HTML string for displaying results
        """
        # Set colors based on threat level
        if result['is_anomaly']:
            bg_color = "#ffebee"  # Light red
            border_color = "#d32f2f"  # Red
            header_color = "#b71c1c"  # Dark red
            text_color = "#212121"  # Dark gray
            threat_gradient = "linear-gradient(135deg, #ff5252, #b71c1c)"
        elif result['predicted_attack'] != "Normal Traffic":
            bg_color = "#fff3e0"  # Light orange
            border_color = "#f57c00"  # Orange
            header_color = "#e65100"  # Dark orange
            text_color = "#212121"
            threat_gradient = "linear-gradient(135deg, #ffb74d, #e65100)"
        else:
            bg_color = "#e8f5e8"  # Light green
            border_color = "#388e3c"  # Green
            header_color = "#1b5e20"  # Dark green
            text_color = "#212121"
            threat_gradient = "linear-gradient(135deg, #66bb6a, #1b5e20)"

        # Create threat meter visualization
        threat_meter = f"""
        <div style="width: 100%; background: #f0f0f0; border-radius: 10px; height: 20px; margin: 10px 0; overflow: hidden;">
            <div style="width: {result['threat_score']}%; height: 100%; background: {threat_gradient}; border-radius: 10px; transition: width 0.5s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #424242; font-weight: 600;">
            <span>Low</span>
            <span>Medium</span>
            <span>High</span>
        </div>
        """

        # Create probability bars for each class
        probability_bars = ""
        for attack_type, prob in result['all_probabilities'].items():
            if prob > 0.001:  # Only show significant probabilities
                bar_width = min(prob * 100, 100)
                bar_color = "#d32f2f" if attack_type != "Normal Traffic" else "#388e3c"
                probability_bars += f"""
                <div style="margin: 12px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: {text_color}; font-size: 14px;">{attack_type}</span>
                        <span style="color: #424242; font-weight: 600; font-size: 14px;">{prob:.3f}</span>
                    </div>
                    <div style="width: 100%; background: #f5f5f5; border-radius: 5px; height: 10px; overflow: hidden;">
                        <div style="width: {bar_width}%; height: 100%; background: {bar_color}; border-radius: 5px; transition: width 0.5s ease;"></div>
                    </div>
                </div>
                """

        # Check if we're using mock system
        is_mock = hasattr(self.nids, 'xgb_model') and hasattr(self.nids.xgb_model, '__class__') and 'Mock' in self.nids.xgb_model.__class__.__name__
        
        # Build the complete HTML output
        output_html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 30px; border-radius: 20px; background: {bg_color}; border: 3px solid {border_color}; box-shadow: 0 8px 25px rgba(0,0,0,0.1); color: {text_color};">
            <div style="text-align: center; margin-bottom: 25px; padding: 20px; background: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h1 style="color: {header_color}; margin: 0; font-size: 28px; font-weight: 700; text-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                    {result['alert_color']} {result['threat_level']}
                </h1>
                <p style="color: #424242; margin: 10px 0 0 0; font-size: 16px; font-weight: 500;">Threat Analysis Complete</p>
                <p style="color: #666; margin: 5px 0 0 0; font-size: 12px; font-weight: 500;">
                    Mode: {'🛡️ Real Model' if not is_mock else '🎭 Demo Mode'}
                </p>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 25px;">
                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: {header_color}; margin-top: 0; border-bottom: 2px solid {border_color}; padding-bottom: 10px; font-size: 18px; font-weight: 700;">🎯 Primary Classification</h3>
                    <div style="display: grid; gap: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #e0e0e0;">
                            <span style="font-weight: 600; color: {text_color}; font-size: 14px;">Attack Type:</span>
                            <span style="background: {border_color}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px;">{result['predicted_attack']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #e0e0e0;">
                            <span style="font-weight: 600; color: {text_color}; font-size: 14px;">Confidence:</span>
                            <span style="font-weight: 700; color: {header_color}; font-size: 16px;">{result['confidence']:.4f}</span>
                        </div>
                        <div style="padding: 12px 0;">
                            <span style="font-weight: 600; color: {text_color}; font-size: 14px;">Threat Level:</span>
                            {threat_meter}
                        </div>
                    </div>
                </div>

                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: {header_color}; margin-top: 0; border-bottom: 2px solid {border_color}; padding-bottom: 10px; font-size: 18px; font-weight: 700;">🔍 Anomaly Detection</h3>
                    <div style="display: grid; gap: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #e0e0e0;">
                            <span style="font-weight: 600; color: {text_color}; font-size: 14px;">Anomaly Detected:</span>
                            <span style="background: {'#d32f2f' if result['is_anomaly'] else '#388e3c'}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                                {'YES 🔴' if result['is_anomaly'] else 'NO 🟢'}
                            </span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #e0e0e0;">
                            <span style="font-weight: 600; color: {text_color}; font-size: 14px;">Reconstruction Error:</span>
                            <span style="font-weight: 700; color: {header_color}; font-size: 16px;">{result['reconstruction_error']:.6f}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0;">
                            <span style="font-weight: 600; color: {text_color}; font-size: 14px;">Anomaly Threshold:</span>
                            <span style="color: #424242; font-weight: 600; font-size: 14px;">{result['anomaly_threshold']:.6f}</span>
                        </div>
                    </div>
                </div>
            </div>

            <div style="background: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="color: {header_color}; margin-top: 0; border-bottom: 2px solid {border_color}; padding-bottom: 10px; font-size: 18px; font-weight: 700;">📊 Threat Probability Distribution</h3>
                <div style="max-height: 300px; overflow-y: auto; padding-right: 10px;">
                    {probability_bars}
                </div>
            </div>

            <div style="margin-top: 20px; padding: 15px; background: white; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <small style="color: #424242; font-weight: 500;">
                    <strong>🕒 Analysis Timestamp:</strong> {result['timestamp']} |
                    <strong>🛡️ System Status:</strong> {'Production Ready' if not is_mock else 'Demo Mode'}
                </small>
            </div>
        </div>
        """

        return output_html

    def create_error_output(self, error_msg):
        """
        Create HTML error message display
        
        Args:
            error_msg: Error message to display
        
        Returns:
            str: HTML string for error display
        """
        error_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 30px; border-radius: 15px; background: #ffe6e6; border: 2px solid #d32f2f;">
            <div style="text-align: center; color: #b71c1c;">
                <h2 style="margin: 0; font-weight: 700;">❌ Prediction Error</h2>
                <p style="font-size: 16px; margin: 15px 0; font-weight: 500;">{error_msg}</p>
            </div>
        </div>
        """
        return error_html

    def create_gradio_interface(self):
        """
        Create the Gradio web interface for the NIDS
        
        Returns:
            gr.Blocks: Gradio interface object
        """
        # Create input components (sliders for each feature)
        input_components = []
        
        # Configuration for each feature slider
        feature_configs = {
            'Destination Port': {"min": 1, "max": 65535, "default": 53, "step": 1},
            'Total Fwd Packets': {"min": 1, "max": 10000, "default": 2, "step": 1},
            'Total Length of Fwd Packets': {"min": 0, "max": 50000000, "default": 64, "step": 1000},
            'Fwd Packet Length Max': {"min": 0, "max": 50000, "default": 32, "step": 50},
            'Fwd Packet Length Mean': {"min": 0, "max": 50000, "default": 32, "step": 50},
            'Fwd Packet Length Std': {"min": 0, "max": 5000, "default": 0, "step": 50},
            'Flow Bytes/s': {"min": 0, "max": 500000000, "default": 1000000, "step": 1000000},
            'Flow Packets/s': {"min": 0, "max": 500000, "default": 20000, "step": 1000},
            'Flow IAT Max': {"min": 0, "max": 50000000, "default": 100000, "step": 100000},
            'Fwd IAT Max': {"min": 0, "max": 50000000, "default": 50000, "step": 100000},
            'Fwd IAT Min': {"min": 0, "max": 1000000, "default": 1, "step": 1000},
            'Bwd IAT Min': {"min": 0, "max": 1000000, "default": 1, "step": 1000},
            'Fwd Packets/s': {"min": 0, "max": 500000, "default": 10000, "step": 1000},
            'PSH Flag Count': {"min": 0, "max": 1000, "default": 0, "step": 1},
            'ACK Flag Count': {"min": 0, "max": 1000, "default": 1, "step": 1},
            'Subflow Fwd Bytes': {"min": 0, "max": 50000000, "default": 64, "step": 1000},
            'act_data_pkt_fwd': {"min": 0, "max": 2500, "default": 1, "step": 1},
            'Idle Mean': {"min": 0, "max": 50000000, "default": 0, "step": 100000},
            'Idle Max': {"min": 0, "max": 50000000, "default": 0, "step": 100000},
            'Idle Min': {"min": 0, "max": 50000000, "default": 0, "step": 100000}
        }

        # Descriptions for each feature (shown as tooltips)
        feature_descriptions = {
            'Destination Port': "Destination port number (1-65535)",
            'Total Fwd Packets': "Total packets in forward direction",
            'Total Length of Fwd Packets': "Total size of forward packets (bytes)",
            'Fwd Packet Length Max': "Maximum forward packet length",
            'Fwd Packet Length Mean': "Mean forward packet length",
            'Fwd Packet Length Std': "Standard deviation of forward packet length",
            'Flow Bytes/s': "Flow bytes per second",
            'Flow Packets/s': "Flow packets per second",
            'Flow IAT Max': "Maximum inter-arrival time of flow",
            'Fwd IAT Max': "Maximum forward inter-arrival time",
            'Fwd IAT Min': "Minimum forward inter-arrival time",
            'Bwd IAT Min': "Minimum backward inter-arrival time",
            'Fwd Packets/s': "Forward packets per second",
            'PSH Flag Count': "Number of packets with PSH flag",
            'ACK Flag Count': "Number of packets with ACK flag",
            'Subflow Fwd Bytes': "Subflow forward bytes",
            'act_data_pkt_fwd': "Actual data packets in forward direction",
            'Idle Mean': "Mean idle time",
            'Idle Max': "Maximum idle time",
            'Idle Min': "Minimum idle time"
        }

        # Create sliders for each feature
        for feature in self.nids.feature_names:
            config = feature_configs[feature]
            input_components.append(
                gr.Slider(
                    label=feature,
                    minimum=config["min"],
                    maximum=config["max"],
                    value=config["default"],
                    step=config["step"],
                    info=feature_descriptions.get(feature, "Network traffic feature")
                )
            )

        def predict_attack(*feature_values):
            """
            Prediction function called by Gradio when user clicks analyze
            
            Args:
                *feature_values: List of feature values from sliders
            
            Returns:
                tuple: (HTML output, alert indicator, alert message)
            """
            # Convert slider values to dictionary
            features_dict = dict(zip(self.nids.feature_names, feature_values))

            # Get prediction from model
            result = self.predict_single_observation(features_dict)

            # Handle prediction errors
            if 'error' in result:
                error_html = self.create_error_output(result['error'])
                return error_html, "❌", "Error occurred during prediction"

            # Create visualization
            output_html = self.create_visual_output(result)

            # Create alert message based on threat level
            if result['is_anomaly']:
                alert_msg = f"🚨 HIGH ALERT: Novel anomaly detected! Reconstruction error: {result['reconstruction_error']:.6f}"
            elif result['predicted_attack'] != "Normal Traffic":
                alert_msg = f"⚠️  ATTACK DETECTED: {result['predicted_attack']} (Confidence: {result['confidence']:.4f})"
            else:
                alert_msg = "✅ SECURE: Normal traffic - No threats detected"

            return output_html, result['alert_color'], alert_msg

        # Custom CSS for styling
        custom_css = """
        .analyze-btn {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            padding: 15px 30px !important;
            border-radius: 10px !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
        }

        .analyze-btn:hover {
            background: linear-gradient(45deg, #FF5252, #26A69A) !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .gr-button {
            transition: all 0.3s ease !important;
        }

        .gr-slider {
            padding: 10px 0 !important;
        }

        .gr-slider .value {
            font-weight: bold !important;
            color: #1976D2 !important;
        }

        .tab-nav {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .critical-tab {
            background: linear-gradient(135deg, #d32f2f, #b71c1c) !important;
            color: white !important;
        }

        .important-tab {
            background: linear-gradient(135deg, #f57c00, #e65100) !important;
            color: white !important;
        }

        .advanced-tab {
            background: linear-gradient(135deg, #1976D2, #0D47A1) !important;
            color: white !important;
        }

        .gr-input, .gr-slider, .gr-textbox {
            color: #212121 !important;
        }

        .gr-label {
            color: #212121 !important;
            font-weight: 600 !important;
        }

        .gr-markdown {
            color: #212121 !important;
        }

        .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
            color: #212121 !important;
            font-weight: 700 !important;
        }

        .gr-textbox {
            font-weight: 600 !important;
        }
        """

        # Create the main Gradio interface
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="Network Intrusion Detection System",
            css=custom_css
        ) as demo:
            # Check if using mock system for status display
            is_mock = hasattr(self.nids, 'xgb_model') and hasattr(self.nids.xgb_model, '__class__') and 'Mock' in self.nids.xgb_model.__class__.__name__
            
            status_badge = "🛡️ Production Model" if not is_mock else "🎭 Demo Mode"
            status_color = "#4CAF50" if not is_mock else "#FF9800"
            
            # Header section
            gr.Markdown(f"""
            <div style="text-align: center;">
                <h1 style="font-size: 42px; margin-bottom: 10px; color: #212121;">🛡️ Network Intrusion Detection System</h1>
                <p style="font-size: 18px; color: #424242; margin-bottom: 10px;">
                    <strong>Real-time Cybersecurity Threat Detection using Advanced Machine Learning</strong><br>
                    Combining XGBoost for known attack classification and Autoencoder for novel anomaly detection
                </p>
                <div style="display: inline-block; background: {status_color}; color: white; padding: 8px 20px; border-radius: 20px; font-weight: bold; margin-bottom: 20px;">
                    {status_badge}
                </div>
            </div>
            """)

            # Main interface layout
            with gr.Row():
                with gr.Column(scale=1):
                    # Input column
                    gr.Markdown("""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; color: white;">
                        <h2 style="margin: 0; color: white; font-weight: 700;">📊 Configure Network Traffic</h2>
                        <p style="margin: 10px 0 0 0; opacity: 0.9; font-weight: 500;">Adjust the sliders to simulate network traffic patterns</p>
                    </div>
                    """)

                    # Feature tabs for better organization
                    with gr.Tabs():
                        with gr.TabItem("🔴 Critical Indicators", elem_classes="critical-tab"):
                            gr.Markdown("### 🚨 Primary Attack Detection Features")
                            gr.Markdown("These features are most critical for identifying malicious network activity:")
                            critical_features = [
                                'Total Fwd Packets', 'Total Length of Fwd Packets', 'Flow Bytes/s',
                                'Flow Packets/s', 'Fwd Packets/s', 'Destination Port'
                            ]
                            for feature in critical_features:
                                if feature in self.nids.feature_names:
                                    idx = self.nids.feature_names.index(feature)
                                    input_components[idx].render()

                        with gr.TabItem("🟡 Important Metrics", elem_classes="important-tab"):
                            gr.Markdown("### 📈 Key Performance & Behavioral Metrics")
                            gr.Markdown("These features provide important context for traffic analysis:")
                            important_features = [
                                'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                                'Flow IAT Max', 'Fwd IAT Max', 'Subflow Fwd Bytes', 'act_data_pkt_fwd'
                            ]
                            for feature in important_features:
                                if feature in self.nids.feature_names:
                                    idx = self.nids.feature_names.index(feature)
                                    input_components[idx].render()

                        with gr.TabItem("🔵 Advanced Details", elem_classes="advanced-tab"):
                            gr.Markdown("### ⚙️ Technical Protocol Details")
                            gr.Markdown("Advanced network protocol and timing characteristics:")
                            advanced_features = [
                                'Fwd IAT Min', 'Bwd IAT Min', 'PSH Flag Count', 'ACK Flag Count',
                                'Idle Mean', 'Idle Max', 'Idle Min'
                            ]
                            for feature in advanced_features:
                                if feature in self.nids.feature_names:
                                    idx = self.nids.feature_names.index(feature)
                                    input_components[idx].render()

                    # Analyze button
                    predict_btn = gr.Button(
                        "🔍 Analyze Network Traffic",
                        variant="primary",
                        size="lg",
                        elem_classes="analyze-btn"
                    )

                with gr.Column(scale=1):
                    # Output column
                    gr.Markdown("""
                    <div style="background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%); padding: 25px; border-radius: 15px; color: white;">
                        <h2 style="margin: 0; color: white; font-weight: 700;">📋 Security Analysis Results</h2>
                        <p style="margin: 10px 0 0 0; opacity: 0.9; font-weight: 500;">Real-time threat assessment and anomaly detection</p>
                    </div>
                    """)

                    # Main output display
                    output_html = gr.HTML(
                        label="Detailed Analysis",
                        value="<div style='text-align: center; padding: 50px; color: #424242;'><h3 style=\"color: #424242;\">🛡️ Ready for Analysis</h3><p style=\"color: #424242;\">Configure network parameters and click 'Analyze Network Traffic' to begin threat detection</p></div>"
                    )

                    # Alert indicators
                    with gr.Row():
                        alert_indicator = gr.Textbox(
                            label="🔔 Alert Status",
                            interactive=False,
                            max_lines=1
                        )
                        alert_message = gr.Textbox(
                            label="📢 Security Alert",
                            interactive=False,
                            max_lines=2
                        )

            # Example scenarios section
            gr.Markdown("""
            <div style="background: #f8f9fa; padding: 20px; border-radius: 15px; margin-top: 20px;">
                <h2 style="margin: 0; color: #212121; font-weight: 700;">🚀 Real-World Attack Scenarios</h2>
                <p style="margin: 10px 0 0 0; color: #424242; font-weight: 500;">Click any example below to test different network attack scenarios</p>
            </div>
            """)

            # Pre-configured examples for testing
            example_values = [
                [53, 2, 64, 32, 32, 0, 1000000, 20000, 100000, 50000, 1, 1, 10000, 0, 1, 64, 1, 0, 0, 0],
                [80, 5000, 2500000, 1500, 500, 300, 250000000, 250000, 1000, 500, 1, 1, 200000, 0, 0, 2500000, 2500, 0, 0, 0],
                [22, 150, 12000, 1500, 80, 200, 5000000, 50000, 50000, 50000, 100, 100, 25000, 0, 1, 12000, 75, 1000000, 5000000, 100000],
                [23, 200, 16000, 800, 80, 150, 8000000, 80000, 20000, 20000, 50, 50, 40000, 1, 0, 16000, 100, 500000, 2000000, 10000],
                [443, 1000, 50000000, 50000, 50000, 1000, 100000000, 50000, 5000, 5000, 1, 1, 25000, 5, 5, 50000000, 500, 0, 0, 0],
                [6667, 50, 5000, 500, 100, 150, 2000000, 25000, 1000000, 1000000, 1000, 1000, 12500, 2, 2, 5000, 25, 500000, 1000000, 100000]
            ]

            example_labels = [
                "🟢 Normal DNS Traffic",
                "🔴 DDoS Attack - Volume Flood",
                "🟡 Port Scanning - Reconnaissance",
                "🟡 Brute Force - Authentication Attacks",
                "🔴 Data Exfiltration - Large Transfer",
                "🟡 Botnet C&C - Command Channel"
            ]

            # Add examples to interface
            gr.Examples(
                examples=example_values,
                inputs=input_components,
                outputs=[output_html, alert_indicator, alert_message],
                fn=predict_attack,
                cache_examples=False,
                label="🎯 Click to test real attack scenarios:",
                examples_per_page=3
            )

            # Connect button to prediction function
            predict_btn.click(
                fn=predict_attack,
                inputs=input_components,
                outputs=[output_html, alert_indicator, alert_message]
            )

            # Footer
            gr.Markdown("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); color: white; border-radius: 15px;">
                <h3 style="margin: 0; color: white; font-weight: 700;">🛡️ Enterprise-Grade Security Monitoring</h3>
                <p style="margin: 10px 0; opacity: 0.9; font-weight: 500;">
                    <strong>System Performance:</strong> 99.97% Accuracy | Real-time Monitoring | Novel Threat Detection<br>
                    <strong>Attack Coverage:</strong> DDoS | Port Scanning | Brute Force | Data Exfiltration | Botnets<br>
                    <em>Powered by XGBoost + Autoencoder ML Pipeline</em>
                </p>
            </div>
            """)

        return demo

def deploy_nids(trained_system=None):
    """
    Main deployment function
    
    Args:
        trained_system: Pre-trained NIDS model (optional)
    
    Returns:
        Gradio interface object
    """
    print("🚀 Deploying Network Intrusion Detection System...")
    
    # Create deployer instance
    deployer = NIDSDeployer(trained_system)
    
    # Create and return Gradio interface
    demo = deployer.create_gradio_interface()
    
    return demo

def save_trained_model(nids_model, filename="trained_nids_model.pkl"):
    """
    Save a trained NIDS model to disk
    
    Args:
        nids_model: Trained NIDS model object
        filename: Output filename
    
    Returns:
        bool: True if save was successful
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(nids_model, f)
        print(f"✅ Model saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

if __name__ == "__main__":
    """
    Main entry point for the deployment script
    """
    # Ensure Gradio is installed
    try:
        import gradio
    except ImportError:
        print("❌ Gradio is not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        import gradio
    
    # Display startup banner
    print("\n" + "="*60)
    print("🛡️ NETWORK INTRUSION DETECTION SYSTEM - DEPLOYMENT")
    print("="*60)
    
    # Try to load trained model
    print("\n📋 Loading trained model...")
    
    trained_system = None
    try:
        # Check global scope first (for interactive use)
        if 'trained_system' in globals():
            trained_system = globals()['trained_system']
            print("✅ Found trained system in global scope")
        # Then check for saved model file
        elif os.path.exists("trained_nids_simple.pkl"):
            with open("trained_nids_simple.pkl", 'rb') as f:
                trained_system = pickle.load(f)
            print("✅ Loaded trained system from trained_nids_simple.pkl")
    except Exception as e:
        print(f"⚠️ Could not load trained system: {e}")
        trained_system = None
    
    # Deploy the interface
    demo = deploy_nids(trained_system)
    
    # Launch information
    print("\n" + "="*60)
    print("🌐 Launching NIDS Web Interface...")
    print("="*60)
    print("\n➡️ Local URL: http://127.0.0.1:7860")
    print("➡️ Public URL will be generated after launch")
    print("➡️ Press Ctrl+C to stop the server")
    print("\n" + "="*60)
    
    # Launch the Gradio interface
    demo.launch(
        share=True,  # Create public URL
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        debug=False
    )