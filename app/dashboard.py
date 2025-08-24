import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import yaml
import os

# Import our core components
import sys
sys.path.append('../core')
from data_simulator import data_simulator
from predictive_model import predictive_model

# Page configuration
st.set_page_config(
    page_title="Honeywell PrAna - Operational Excellence",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
with open('../core/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class HoneywellDashboard:
    def __init__(self):
        self.data_history = pd.DataFrame()
        self.anomaly_history = []
        self.initialize_session_state()
        
    def initialize_session_state(self):
        if 'data_history' not in st.session_state:
            st.session_state.data_history = pd.DataFrame()
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
            
    def setup_sidebar(self):
        st.sidebar.image("assets/honeywell_logo.png", width=200)
        st.sidebar.title("Honeywell PrAna")
        st.sidebar.markdown("---")
        
        # System status
        st.sidebar.subheader("System Status")
        status = st.sidebar.empty()
        
        # Model metrics
        if st.session_state.model_trained:
            st.sidebar.subheader("Model Performance")
            col1, col2, col3 = st.sidebar.columns(3)
            col1.metric("Accuracy", f"{predictive_model.model_accuracy:.2%}")
            col2.metric("Precision", f"{predictive_model.model_precision:.2%}")
            col3.metric("Recall", f"{predictive_model.model_recall:.2%}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Configuration")
        update_interval = st.sidebar.slider("Update Interval (s)", 1, 10, config['system']['update_interval'])
        anomaly_chance = st.sidebar.slider("Anomaly Frequency (%)", 0, 20, 5)
        
        return status, update_interval, anomaly_chance / 100
    
    def create_metrics_row(self):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_data = self.data_history.iloc[-1] if not self.data_history.empty else None
            if current_data is not None:
                st.metric("Zone 1 Temp", f"{current_data['zone1_temperature']}°C", 
                         delta="Normal" if not current_data['anomaly_detected'] else "Anomaly")
        
        with col2:
            if current_data is not None:
                st.metric("Zone 2 Temp", f"{current_data['zone2_temperature']}°C",
                         delta="Normal" if not current_data['anomaly_detected'] else "Anomaly")
        
        with col3:
            if current_data is not None:
                st.metric("Conveyor Speed", f"{current_data['conveyor_speed']} m/s",
                         delta="Normal" if not current_data['anomaly_detected'] else "Anomaly")
        
        with col4:
            anomalies = len(self.data_history[self.data_history['anomaly_detected'] == True])
            total = len(self.data_history)
            anomaly_rate = (anomalies / total * 100) if total > 0 else 0
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%", 
                     delta=f"{anomalies} detected" if anomalies > 0 else "None")
    
    def create_process_plots(self):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Zone 1 Temperature', 'Zone 2 Temperature', 
                           'Conveyor Speed', 'Anomaly Detection Confidence'),
            vertical_spacing=0.15
        )
        
        if not self.data_history.empty:
            # Zone 1 Temperature
            fig.add_trace(
                go.Scatter(x=self.data_history['timestamp'], y=self.data_history['zone1_temperature'],
                          name='Zone 1 Temp', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Zone 2 Temperature
            fig.add_trace(
                go.Scatter(x=self.data_history['timestamp'], y=self.data_history['zone2_temperature'],
                          name='Zone 2 Temp', line=dict(color='red')),
                row=1, col=2
            )
            
            # Conveyor Speed
            fig.add_trace(
                go.Scatter(x=self.data_history['timestamp'], y=self.data_history['conveyor_speed'],
                          name='Conveyor Speed', line=dict(color='green')),
                row=2, col=1
            )
            
            # Anomaly markers
            anomalies = self.data_history[self.data_history['anomaly_detected'] == True]
            if not anomalies.empty:
                for i, row in anomalies.iterrows():
                    fig.add_vline(x=row['timestamp'], line_dash="dash", line_color="red",
                                 opacity=0.3, row="all", col="all")
        
        fig.update_layout(height=600, showlegend=True, title_text="Honeywell Process Monitoring")
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        st.title("Honeywell PrAna - Predictive Process Monitoring")
        st.markdown("Real-time anomaly detection for food and beverage manufacturing")
        
        # Initialize model
        if not st.session_state.model_trained:
            with st.spinner("Training initial model..."):
                training_data = data_simulator.generate_training_data(5000)
                predictive_model.train_model(training_data)
                predictive_model.save_model('../models/trained_model.pkl')
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
        
        # Setup sidebar
        status, update_interval, anomaly_chance = self.setup_sidebar()
        
        # Main dashboard
        placeholder = st.empty()
        
        while True:
            # Generate new data point
            new_data = data_simulator.get_live_data(anomaly_chance)
            
            # Predict anomaly
            is_anomaly, confidence = predictive_model.predict_anomaly(new_data)
            new_data['ml_anomaly_prediction'] = is_anomaly
            new_data['prediction_confidence'] = confidence
            
            # Update data history
            new_df = pd.DataFrame([new_data])
            if st.session_state.data_history.empty:
                st.session_state.data_history = new_df
            else:
                st.session_state.data_history = pd.concat([st.session_state.data_history, new_df], ignore_index=True)
            
            # Keep only recent data
            if len(st.session_state.data_history) > config['system']['data_window']:
                st.session_state.data_history = st.session_state.data_history.tail(config['system']['data_window'])
            
            # Update dashboard
            with placeholder.container():
                # Update status
                if new_data['anomaly_detected']:
                    status.error("🚨 ANOMALY DETECTED")
                else:
                    status.success("✅ NORMAL OPERATION")
                
                # Metrics
                self.create_metrics_row()
                
                # Process plots
                self.create_process_plots()
                
                # Recent events table
                st.subheader("Recent Events")
                recent_data = st.session_state.data_history.tail(10)[['timestamp', 'zone1_temperature', 
                                                                     'zone2_temperature', 'conveyor_speed',
                                                                     'anomaly_detected', 'ml_anomaly_prediction',
                                                                     'prediction_confidence']]
                st.dataframe(recent_data.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x == True else '', 
                    subset=['anomaly_detected', 'ml_anomaly_prediction']
                ))
            
            time.sleep(update_interval)

# Run the dashboard
if __name__ == "__main__":
    dashboard = HoneywellDashboard()
    dashboard.run()