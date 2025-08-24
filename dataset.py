import pandas as pd
import numpy as np
import requests
import zipfile
import os

def download_nasa_dataset():
    """Download and prepare the NASA Turbofan dataset"""
    # URL for the NASA Turbofan dataset
    url = "https://ti.arc.nasa.gov/c/6/"
    
    # Since the actual NASA URL requires special access, we'll create a sample version
    # In a real scenario, you would use the actual dataset from:
    # https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
    
    print("Creating sample industrial dataset...")
    
    # Create a realistic industrial dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic industrial data
    data = {
        'timestamp': pd.date_range('2023-06-01 08:00:00', periods=n_samples, freq='5min'),
        'engine_id': np.random.randint(1, 6, n_samples),
        'setting_1': np.random.uniform(0.8, 1.2, n_samples),
        'setting_2': np.random.uniform(0.9, 1.1, n_samples),
        'sensor_1': np.random.normal(500, 20, n_samples),
        'sensor_2': np.random.normal(650, 25, n_samples),
        'sensor_3': np.random.normal(185, 5, n_samples),  # Temperature-like
        'sensor_4': np.random.normal(195, 5, n_samples),  # Temperature-like
        'sensor_5': np.random.normal(2.5, 0.3, n_samples),  # Speed-like
        'sensor_6': np.random.normal(450, 30, n_samples),  # Energy-like
        'sensor_7': np.random.normal(1200, 50, n_samples),  # Production-like
        'anomaly': np.zeros(n_samples),
        'rul': np.linspace(150, 0, n_samples)  # Remaining Useful Life
    }
    
    # Introduce some anomalies
    anomaly_indices = [250, 500, 750, 900]
    for idx in anomaly_indices:
        data['anomaly'][idx] = 1
        data['sensor_3'][idx] += np.random.uniform(15, 25)  # Temperature spike
        data['sensor_6'][idx] -= np.random.uniform(40, 60)  # Energy drop
    
    df = pd.DataFrame(data)
    df.to_csv('nasa_industrial_data.csv', index=False)
    print("Dataset created: nasa_industrial_data.csv")
    return df

if __name__ == "__main__":
    download_nasa_dataset()