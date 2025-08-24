import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from datetime import datetime
import os

class HoneywellPredictiveModel:
    """Anomaly detection model aligned with Honeywell's reliability standards"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_accuracy = 0.0
        self.model_precision = 0.0
        self.model_recall = 0.0
        self.last_trained = None
        
    def prepare_features(self, df):
        """Prepare features for ML model"""
        features = df[['zone1_temperature', 'zone2_temperature', 'conveyor_speed']].copy()
        
        # Add rolling statistics for temporal patterns
        for col in features.columns:
            features[f'{col}_rolling_mean_10'] = features[col].rolling(window=10).mean()
            features[f'{col}_rolling_std_10'] = features[col].rolling(window=10).std()
        
        # Drop NaN values from rolling calculations
        features = features.dropna()
        
        return features
    
    def train_model(self, training_data):
        """Train the anomaly detection model"""
        print("Training Honeywell PrAna model...")
        
        # Prepare features
        X = self.prepare_features(training_data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest (robust for industrial applications)
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # Expected anomaly rate
            random_state=42,
            verbose=1
        )
        
        self.model.fit(X_scaled)
        
        # Validate model
        predictions = self.model.predict(X_scaled)
        actual = training_data.loc[X.index, 'anomaly_detected'].astype(int)
        actual_binary = np.where(actual == True, -1, 1)  # Convert to IsolationForest format
        
        self.model_accuracy = accuracy_score(actual_binary, predictions)
        self.model_precision = precision_score(actual_binary, predictions, pos_label=-1)
        self.model_recall = recall_score(actual_binary, predictions, pos_label=-1)
        
        self.last_trained = datetime.now()
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {self.model_accuracy:.4f}")
        print(f"Precision: {self.model_precision:.4f}")
        print(f"Recall: {self.model_recall:.4f}")
        
        return self
    
    def predict_anomaly(self, data_point):
        """Predict if a data point is anomalous"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Convert single data point to DataFrame
        df = pd.DataFrame([data_point])
        features = self.prepare_features(df)
        
        if features.empty:
            return False, 0.0  # Not enough data for rolling features
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)
        anomaly_score = self.model.decision_function(features_scaled)
        
        # Convert prediction: -1 = anomaly, 1 = normal
        is_anomaly = (prediction[0] == -1)
        confidence = abs(anomaly_score[0])
        
        return is_anomaly, confidence
    
    def save_model(self, filepath):
        """Save trained model to file"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'metrics': {
                'accuracy': self.model_accuracy,
                'precision': self.model_precision,
                'recall': self.model_recall
            },
            'last_trained': self.last_trained
        }, filepath)
    
    def load_model(self, filepath):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            return False
        
        saved_data = joblib.load(filepath)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.model_accuracy = saved_data['metrics']['accuracy']
        self.model_precision = saved_data['metrics']['precision']
        self.model_recall = saved_data['metrics']['recall']
        self.last_trained = saved_data['last_trained']
        
        print(f"Model loaded successfully (trained: {self.last_trained})")
        return True

# Global model instance
predictive_model = HoneywellPredictiveModel()