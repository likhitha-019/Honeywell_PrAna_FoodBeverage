import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import time

class HoneywellDataSimulator:
    """Simulates data from Honeywell Experion PKS system"""
    
    def __init__(self):
        self.running = False
        self.current_data = None
        
    def generate_normal_data(self, base_value, variability=0.02):
        """Generate normal operating data with slight variability"""
        return base_value * (1 + random.uniform(-variability, variability))
    
    def generate_anomaly(self, base_value, anomaly_type):
        """Generate specific types of process anomalies"""
        if anomaly_type == "gradual_drift":
            return base_value * (1 + random.uniform(0.05, 0.15))  # 5-15% drift
        elif anomaly_type == "sudden_spike":
            return base_value * (1.25)  # 25% spike
        elif anomaly_type == "oscillation":
            return base_value * (1 + 0.1 * np.sin(time.time()))  # 10% oscillation
        else:
            return base_value
    
    def get_live_data(self, anomaly_chance=0.05):
        """Generate live process data with occasional anomalies"""
        timestamp = datetime.now()
        
        # Default to normal operation
        anomaly_type = None
        zone1_temp = self.generate_normal_data(185.0)
        zone2_temp = self.generate_normal_data(195.0)
        conveyor_speed = self.generate_normal_data(2.5)
        
        # Occasionally introduce an anomaly (5% chance)
        if random.random() < anomaly_chance:
            anomaly_type = random.choice(["gradual_drift", "sudden_spike", "oscillation"])
            affected_param = random.choice(["zone1", "zone2", "conveyor"])
            
            if affected_param == "zone1":
                zone1_temp = self.generate_anomaly(185.0, anomaly_type)
            elif affected_param == "zone2":
                zone2_temp = self.generate_anomaly(195.0, anomaly_type)
            else:
                conveyor_speed = self.generate_anomaly(2.5, anomaly_type)
        
        data = {
            'timestamp': timestamp,
            'zone1_temperature': round(zone1_temp, 2),
            'zone2_temperature': round(zone2_temp, 2),
            'conveyor_speed': round(conveyor_speed, 2),
            'anomaly_type': anomaly_type,
            'anomaly_detected': anomaly_type is not None
        }
        
        return data
    
    def generate_training_data(self, samples=10000):
        """Generate comprehensive training dataset"""
        data = []
        for i in range(samples):
            if i % 100 == 0 and i > 0:  # Periodic anomalies
                anomaly_data = self.get_live_data(anomaly_chance=1.0)
            else:
                anomaly_data = self.get_live_data(anomaly_chance=0.0)
            data.append(anomaly_data)
        
        return pd.DataFrame(data)

# Singleton instance
data_simulator = HoneywellDataSimulator()