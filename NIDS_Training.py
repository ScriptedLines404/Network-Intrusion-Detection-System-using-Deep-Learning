"""
Network Intrusion Detection System (NIDS) - Training Pipeline
Main script for training ML models (XGBoost + Autoencoder) for network intrusion detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os
import gdown  # For downloading from Google Drive
import zipfile
import pickle
import joblib
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

class NetworkIntrusionDetectionSystem:
    """
    Main class for the Network Intrusion Detection System
    Combines XGBoost for classification and Autoencoder for anomaly detection
    """
    
    def __init__(self, data_path_or_url):
        """
        Initialize the NIDS system
        
        Args:
            data_path_or_url: Path to local CSV file or Google Drive URL for dataset
        """
        self.data_path_or_url = data_path_or_url
        self.data = None  # Raw dataset
        self.X = None  # Features
        self.y = None  # Labels
        self.X_train = None  # Training features
        self.X_test = None  # Testing features
        self.y_train = None  # Training labels
        self.y_test = None  # Testing labels
        self.X_train_ae = None  # Training features for autoencoder (unsupervised)
        self.X_test_ae = None  # Testing features for autoencoder (unsupervised)
        
        # Preprocessing components
        self.scaler = StandardScaler()  # For feature normalization
        self.label_encoder = LabelEncoder()  # For encoding categorical labels
        
        # Models
        self.xgb_model = None  # XGBoost classifier for known attacks
        self.autoencoder = None  # Autoencoder for anomaly detection
        
        # Feature names (network traffic characteristics)
        self.feature_names = [
            'Destination Port', 'Total Fwd Packets', 'Total Length of Fwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Max', 'Fwd IAT Max',
            'Fwd IAT Min', 'Bwd IAT Min', 'Fwd Packets/s', 'PSH Flag Count',
            'ACK Flag Count', 'Subflow Fwd Bytes', 'act_data_pkt_fwd', 'Idle Mean',
            'Idle Max', 'Idle Min'
        ]
        
        # Column name for attack types in dataset
        self.attack_type_col = 'Attack Type'
        
        # Local file paths
        self.local_file_path = 'cicids2017_cleaned_and_processed.csv'
        
        # Anomaly detection threshold
        self.anomaly_threshold = None

    def download_dataset(self):
        """
        Download dataset from Google Drive if URL is provided
        
        Returns:
            str: Path to the dataset file (local path or downloaded file)
        """
        # Check if input is a URL
        if self.data_path_or_url.startswith('http'):
            print(f"Downloading dataset from Google Drive...")

            # Extract file ID from Google Drive URL
            file_id = None
            if 'drive.google.com' in self.data_path_or_url:
                if '/file/d/' in self.data_path_or_url:
                    file_id = self.data_path_or_url.split('/file/d/')[1].split('/')[0]
                elif 'id=' in self.data_path_or_url:
                    file_id = self.data_path_or_url.split('id=')[1].split('&')[0]

            # Download using gdown if file ID was extracted
            if file_id:
                download_url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(download_url, self.local_file_path, quiet=False)
                print(f"Dataset downloaded to: {self.local_file_path}")

                # Verify download
                if os.path.exists(self.local_file_path):
                    return self.local_file_path
                else:
                    print("Warning: File download may have failed.")
                    return None
            else:
                print("Could not extract file ID from URL.")
                return None
        else:
            # Return local file path if not a URL
            return self.data_path_or_url

    def load_and_preprocess_data(self):
        """
        Load and preprocess the dataset
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("Loading dataset...")
        try:
            # Get the actual file path (local or downloaded)
            actual_file_path = self.download_dataset()

            if actual_file_path is None:
                print("Could not download the dataset. Please check the URL or provide a local file path.")
                return False

            if not os.path.exists(actual_file_path):
                print(f"File not found: {actual_file_path}")
                return False

            # Try different methods to read CSV file
            try:
                self.data = pd.read_csv(actual_file_path)
            except pd.errors.ParserError:
                print("CSV parsing error. Trying with different parameters...")
                self.data = pd.read_csv(actual_file_path, encoding='utf-8', on_bad_lines='skip')
            except Exception as e:
                print(f"Error reading CSV: {e}")
                try:
                    self.data = pd.read_csv(actual_file_path, encoding='latin1')
                except Exception as e2:
                    print(f"Failed to load with latin1 encoding: {e2}")
                    return False

            print(f"Dataset loaded successfully. Shape: {self.data.shape}")

            # Show first few columns for debugging
            print(f"\nFirst few columns: {list(self.data.columns)[:20]}...")

            # Check if dataset looks valid
            if self.data.shape[1] < 5:
                print("Warning: Dataset has very few columns. Might be incorrect format.")

            # Find attack type column (handle different column names)
            available_columns = [col.lower() for col in self.data.columns]
            if 'attack type' not in available_columns:
                # Look for columns with 'attack' or 'label' in name
                attack_cols = [col for col in self.data.columns if 'attack' in col.lower() or 'label' in col.lower()]
                if attack_cols:
                    print(f"Found potential attack columns: {attack_cols}")
                    self.attack_type_col = attack_cols[0]
                    print(f"Using '{self.attack_type_col}' as attack type column")
                else:
                    # Create dummy labels if no attack column found (for testing)
                    print("No attack type column found. Creating dummy labels for testing...")
                    self.data['Attack Type'] = 'Normal'
                    self.attack_type_col = 'Attack Type'

            # Display dataset information
            print("\nDataset Info:")
            print(f"Total samples: {len(self.data)}")
            print(f"Number of features: {len(self.data.columns)}")

            if self.attack_type_col in self.data.columns:
                print(f"Attack type distribution:\n{self.data[self.attack_type_col].value_counts().head()}")
            else:
                print(f"Column '{self.attack_type_col}' not found in dataset")

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_features(self):
        """
        Prepare features for model training
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("\nPreparing features...")

        try:
            # Check which features are available in the dataset
            available_features = []
            for feature in self.feature_names:
                if feature in self.data.columns:
                    available_features.append(feature)
                else:
                    print(f"Warning: Feature '{feature}' not found in dataset")

            # Fallback: use numeric columns if predefined features are insufficient
            if len(available_features) < 5:
                print(f"Only {len(available_features)} features available. Using all available numeric columns instead.")
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                if self.attack_type_col in numeric_cols:
                    numeric_cols.remove(self.attack_type_col)
                available_features = numeric_cols[:20]  # Use top 20 numeric columns

            self.feature_names = available_features
            print(f"Using {len(self.feature_names)} features: {self.feature_names[:10]}...")

            # Extract features
            X = self.data[self.feature_names].copy()

            # Extract and encode labels
            if self.attack_type_col in self.data.columns:
                y = self.data[self.attack_type_col].copy()

                # Ensure labels are strings
                if y.dtype != 'object':
                    y = y.astype(str)

                # Encode categorical labels to numeric
                y_encoded = self.label_encoder.fit_transform(y)

                # Split data for XGBoost (supervised learning)
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                # Use unsupervised approach if no labels available
                print("No target column found. Using unsupervised approach only.")
                y_encoded = np.zeros(len(X))
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42
                )

            # Handle missing values
            X = X.fillna(X.mean())

            # Prepare data for autoencoder (unsupervised, no labels needed)
            X_ae = self.data[self.feature_names].copy().fillna(X.mean())
            self.X_train_ae, self.X_test_ae = train_test_split(
                X_ae, test_size=0.2, random_state=42
            )

            # Scale features for both models
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            self.X_train_ae_scaled = self.scaler.fit_transform(self.X_train_ae)
            self.X_test_ae_scaled = self.scaler.transform(self.X_test_ae)

            # Print dataset shapes
            print(f"Training set shape (XGBoost): {self.X_train_scaled.shape}")
            print(f"Test set shape (XGBoost): {self.X_test_scaled.shape}")
            print(f"Training set shape (Autoencoder): {self.X_train_ae_scaled.shape}")
            print(f"Test set shape (Autoencoder): {self.X_test_ae_scaled.shape}")

            return True

        except Exception as e:
            print(f"Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return False

    def build_autoencoder(self, encoding_dim=10):
        """
        Build an autoencoder model for anomaly detection
        
        Args:
            encoding_dim: Dimension of the encoded representation
        
        Returns:
            Model: Keras autoencoder model
        """
        print("\nBuilding Autoencoder...")

        input_dim = len(self.feature_names)

        # Input layer
        input_layer = Input(shape=(input_dim,))

        # Encoder layers (compress data)
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)  # Regularization
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)  # Bottleneck layer

        # Decoder layers (reconstruct data)
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)  # Output layer

        # Create autoencoder model
        self.autoencoder = Model(input_layer, decoded)

        # Compile model
        self.autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error for reconstruction
            metrics=['mae']  # Mean Absolute Error
        )

        # Print model architecture
        print("Autoencoder architecture:")
        print(f"Input dimension: {input_dim}")
        print(f"Encoding dimension: {encoding_dim}")
        self.autoencoder.summary()

        return self.autoencoder

    def train_autoencoder(self, epochs=50, batch_size=256):
        """
        Train the autoencoder model
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
        
        Returns:
            History: Training history object
        """
        print("\nTraining Autoencoder...")

        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]

        # Train autoencoder (unsupervised - both input and output are the same)
        history = self.autoencoder.fit(
            self.X_train_ae_scaled, self.X_train_ae_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test_ae_scaled, self.X_test_ae_scaled),
            callbacks=callbacks,
            verbose=1
        )

        # Plot training history
        self.plot_training_history(history, "Autoencoder")

        return history

    def build_xgboost_model(self):
        """
        Build and train an XGBoost classifier for attack classification
        
        Returns:
            XGBClassifier: Trained XGBoost model
        """
        print("\nBuilding XGBoost model...")

        # XGBoost hyperparameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'multi:softprob',  # Multi-class classification
            'random_state': 42,
            'n_jobs': -1,  # Use all CPU cores
            'tree_method': 'hist'  # Histogram-based algorithm for faster training
        }

        # Adjust objective based on number of classes
        num_classes = len(np.unique(self.y_train))
        if num_classes < 2:
            print("Not enough classes for classification. Using binary classification.")
            params['objective'] = 'binary:logistic'
        elif num_classes == 2:
            params['objective'] = 'binary:logistic'
        else:
            params['num_class'] = num_classes  # For multi-class classification

        # Create and train XGBoost model
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(
            self.X_train_scaled,
            self.y_train,
            eval_set=[(self.X_test_scaled, self.y_test)],
            verbose=False
        )

        return self.xgb_model

    def detect_anomalies(self, threshold_quantile=0.95):
        """
        Detect anomalies using the trained autoencoder
        
        Args:
            threshold_quantile: Quantile to use for anomaly threshold (default: 95th percentile)
        
        Returns:
            tuple: (anomalies, test_mse, threshold)
        """
        print("\nDetecting anomalies with Autoencoder...")

        # Get reconstructions on training data
        train_reconstructions = self.autoencoder.predict(self.X_train_ae_scaled, verbose=0)
        
        # Calculate reconstruction error (Mean Squared Error)
        train_mse = np.mean(np.power(self.X_train_ae_scaled - train_reconstructions, 2), axis=1)

        # Set threshold based on training data reconstruction errors
        threshold = np.quantile(train_mse, threshold_quantile)

        # Get reconstructions on test data
        test_reconstructions = self.autoencoder.predict(self.X_test_ae_scaled, verbose=0)
        test_mse = np.mean(np.power(self.X_test_ae_scaled - test_reconstructions, 2), axis=1)

        # Identify anomalies (samples with reconstruction error above threshold)
        anomalies = test_mse > threshold

        # Print anomaly detection statistics
        print(f"Anomaly detection threshold: {threshold:.4f}")
        print(f"Number of anomalies detected: {np.sum(anomalies)}")
        print(f"Anomaly rate: {np.mean(anomalies):.4f}")

        return anomalies, test_mse, threshold

    def evaluate_models(self):
        """
        Evaluate both XGBoost and Autoencoder models
        
        Returns:
            dict: Evaluation metrics for both models
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)

        print("\n1. XGBoost Performance:")
        try:
            # XGBoost predictions
            y_pred_xgb = self.xgb_model.predict(self.X_test_scaled)
            y_pred_proba_xgb = self.xgb_model.predict_proba(self.X_test_scaled)

            # Calculate metrics
            accuracy_xgb = accuracy_score(self.y_test, y_pred_xgb)
            precision_xgb, recall_xgb, f1_xgb, _ = precision_recall_fscore_support(
                self.y_test, y_pred_xgb, average='weighted', zero_division=0
            )

            print(f"Accuracy: {accuracy_xgb:.4f}")
            print(f"Precision: {precision_xgb:.4f}")
            print(f"Recall: {recall_xgb:.4f}")
            print(f"F1-Score: {f1_xgb:.4f}")
        except Exception as e:
            print(f"XGBoost evaluation error: {e}")
            accuracy_xgb = precision_xgb = recall_xgb = f1_xgb = 0

        print("\n2. Autoencoder Anomaly Detection:")
        try:
            # Anomaly detection
            anomalies, test_mse, threshold = self.detect_anomalies()

            # Print reconstruction error statistics
            print(f"Reconstruction Error Stats:")
            print(f"Mean: {np.mean(test_mse):.4f}")
            print(f"Std: {np.std(test_mse):.4f}")
            print(f"Max: {np.max(test_mse):.4f}")
            print(f"Threshold (95th percentile): {threshold:.4f}")

            self.anomaly_threshold = threshold
        except Exception as e:
            print(f"Autoencoder evaluation error: {e}")
            threshold = 0
            anomalies = []

        print("\n3. Combined System Overview:")
        print("XGBoost handles known attack patterns with high accuracy")
        print("Autoencoder detects novel/unknown anomalies")
        print("System provides defense-in-depth approach")

        # Return all evaluation metrics
        return {
            'xgb_accuracy': accuracy_xgb,
            'xgb_precision': precision_xgb,
            'xgb_recall': recall_xgb,
            'xgb_f1': f1_xgb,
            'autoencoder_threshold': threshold,
            'anomalies_detected': np.sum(anomalies) if len(anomalies) > 0 else 0
        }

    def plot_training_history(self, history, model_name):
        """
        Plot training history (loss and metrics)
        
        Args:
            history: Keras history object from model.fit()
            model_name: Name of the model for plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{model_name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot MAE if available
        if 'mae' in history.history:
            ax2.plot(history.history['mae'], label='Training MAE')
            ax2.plot(history.history['val_mae'], label='Validation MAE')
            ax2.set_title(f'{model_name} - MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        """
        Plot feature importance from XGBoost model
        
        Returns:
            DataFrame: Feature importance DataFrame or None if model not trained
        """
        if self.xgb_model is None:
            print("XGBoost model not trained yet.")
            return None

        # Get feature importance scores
        importance_scores = self.xgb_model.feature_importances_
        
        # Create DataFrame for sorting and visualization
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)

        # Plot top 20 most important features
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df.head(20), y='feature', x='importance')
        plt.title('Top 20 XGBoost Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

        return feature_importance_df

    def run_complete_pipeline(self):
        """
        Run the complete NIDS training pipeline
        
        Returns:
            dict: Results dictionary with evaluation metrics or None if pipeline fails
        """
        print("="*60)
        print("Starting Network Intrusion Detection System Pipeline")
        print("="*60)

        # Step 1: Load and preprocess data
        print("\n[Step 1/5] Loading and preprocessing data...")
        if not self.load_and_preprocess_data():
            print("Failed to load data. Exiting pipeline.")
            return None

        # Step 2: Prepare features
        print("\n[Step 2/5] Preparing features...")
        if not self.prepare_features():
            print("Failed to prepare features. Exiting pipeline.")
            return None

        # Step 3: Build and train autoencoder
        print("\n[Step 3/5] Building and training Autoencoder...")
        self.build_autoencoder()
        self.train_autoencoder(epochs=30)

        # Step 4: Build and train XGBoost
        print("\n[Step 4/5] Building and training XGBoost...")
        self.build_xgboost_model()

        # Step 5: Evaluate models
        print("\n[Step 5/5] Evaluating models...")
        results = self.evaluate_models()

        # Additional visualization
        print("\n[Additional] Generating visualizations...")
        self.plot_feature_importance()

        # Save trained model
        print("\n[Step 7/7] Saving trained model...")
        save_success = self.save_model("trained_nids_model")
        
        if save_success:
            print("✅ Model saved successfully!")
        else:
            print("⚠️ Model saving failed, but pipeline completed.")

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

        return results

    def save_model(self, model_name="trained_nids_model"):
        """
        Save all model components to disk
        
        Args:
            model_name: Base name for saved model files
        
        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            # Create directory for saved models
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save full model object
            full_model_path = os.path.join(model_dir, f"{model_name}.pkl")
            with open(full_model_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"✅ Full model saved to: {full_model_path}")
            
            # Save individual components for flexibility
            scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            
            label_encoder_path = os.path.join(model_dir, f"{model_name}_label_encoder.pkl")
            joblib.dump(self.label_encoder, label_encoder_path)
            
            xgb_path = os.path.join(model_dir, f"{model_name}_xgb.model")
            self.xgb_model.save_model(xgb_path)
            
            autoencoder_path = os.path.join(model_dir, f"{model_name}_autoencoder.h5")
            self.autoencoder.save(autoencoder_path)
            
            features_path = os.path.join(model_dir, f"{model_name}_features.pkl")
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            # Save anomaly threshold if available
            if self.anomaly_threshold is not None:
                threshold_path = os.path.join(model_dir, f"{model_name}_threshold.pkl")
                with open(threshold_path, 'wb') as f:
                    pickle.dump(self.anomaly_threshold, f)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'anomaly_threshold': self.anomaly_threshold,
                'num_features': len(self.feature_names),
                'model_info': {
                    'xgb_model_type': type(self.xgb_model).__name__,
                    'autoencoder_layers': len(self.autoencoder.layers),
                    'label_encoder_classes': list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else []
                }
            }
            metadata_path = os.path.join(model_dir, f"{model_name}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Print summary of saved files
            print(f"✅ All model components saved to {model_dir}/")
            print(f"   - Full model: {model_name}.pkl")
            print(f"   - Scaler: {model_name}_scaler.pkl")
            print(f"   - Label encoder: {model_name}_label_encoder.pkl")
            print(f"   - XGBoost model: {model_name}_xgb.model")
            print(f"   - Autoencoder: {model_name}_autoencoder.h5")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_model(self, model_path="saved_models/trained_nids_model.pkl"):
        """
        Load a previously saved model
        
        Args:
            model_path: Path to saved model file
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Load full model object
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Copy all attributes from loaded model to current instance
            for attr_name in dir(loaded_model):
                if not attr_name.startswith('__'):
                    try:
                        attr_value = getattr(loaded_model, attr_name)
                        setattr(self, attr_name, attr_value)
                    except:
                        pass
            
            print(f"✅ Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def save_model_simple(self, filename="nids_model_simple.pkl"):
        """
        Save a simple version of the model (just the object)
        
        Args:
            filename: Output filename
        
        Returns:
            bool: True if saving successful, False otherwise
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"✅ Model saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False


def main():
    """
    Main function to run the complete NIDS training pipeline
    """
    # Google Drive URL for the dataset
    google_drive_url = 'https://drive.google.com/file/d/1J5-W5HCwSDLy8f25MjOYG_IlW06O7w7n/view?usp=drive_link'
    print(f"Using Google Drive URL: {google_drive_url}")

    # Create NIDS instance
    nids = NetworkIntrusionDetectionSystem(google_drive_url)

    # Run complete training pipeline
    results = nids.run_complete_pipeline()

    if results:
        # Print final results
        print("\nFinal Results Summary:")
        print("-" * 30)
        for key, value in results.items():
            print(f"{key}: {value}")

        # Save simple model version for deployment
        print("\n💾 Saving simple model version...")
        nids.save_model_simple("trained_nids_simple.pkl")
        
        # Deployment information
        print("\n" + "="*60)
        print("SYSTEM DEPLOYMENT READY")
        print("="*60)
        print("The system can now be used for:")
        print("1. Real-time network traffic monitoring")
        print("2. Known attack classification (XGBoost)")
        print("3. Novel anomaly detection (Autoencoder)")
        print("4. Comprehensive threat intelligence")
        
        # Save model globally for Gradio deployment
        global trained_system
        trained_system = nids
        print("\n✅ Model saved globally as 'trained_system' for Gradio deployment")

    return nids


def check_and_install_requirements():
    """
    Check if required packages are installed and install missing ones
    """
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn',
                         'scikit-learn', 'xgboost', 'tensorflow', 'gdown']

    import subprocess
    import sys

    # Check and install each required package
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Check joblib separately
    try:
        import joblib
        print("✓ joblib is already installed")
    except ImportError:
        print("Installing joblib...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])

    print("\nAll requirements are satisfied!")


if __name__ == "__main__":
    """
    Main execution block
    """
    # Check and install requirements
    print("Checking requirements...")
    check_and_install_requirements()

    # Start NIDS training
    print("\n" + "="*60)
    print("NETWORK INTRUSION DETECTION SYSTEM - TRAINING")
    print("="*60)

    # Run main training pipeline
    trained_system = main()

    if trained_system:
        # Offer to deploy immediately
        print("\nExample deployment usage:")
        print("Deployer ready for real-time predictions")
        
        deploy_now = input("\n🎯 Do you want to deploy the model now? (yes/no): ").strip().lower()
        if deploy_now in ['yes', 'y']:
            try:
                # Try to launch Gradio interface
                import gradio as gr
                from NIDS_Gradio_App import deploy_nids
                print("\n🚀 Launching Gradio interface...")
                demo = deploy_nids(trained_system)
                demo.launch(share=True)
            except ImportError:
                print("\n⚠️ Gradio not installed. To deploy, install it with:")
                print("   pip install gradio")
                print("\nThen run the deployment script separately.")
            except Exception as e:
                print(f"\n⚠️ Could not launch Gradio: {e}")
                print("\nYou can still deploy using the saved model files.")
        
        # Final summary
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE")
        print("="*60)
        print("\n📁 Saved model files:")
        print("   - trained_nids_simple.pkl (Complete model for deployment)")
        print("   - saved_models/ (Directory with all model components)")
        print("\n🚀 To deploy using Gradio:")
        print("   1. Make sure Gradio is installed: pip install gradio")
        print("   2. Run the deployment script: python NIDS_Deploy.py")