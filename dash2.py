import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Honeywell PrAna - Real Industrial Data Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #FF7900; font-weight: bold; margin-bottom: 0; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 4px solid #FF7900; margin: 5px 0; }
    .alarm-critical { background-color: #ffcccc; padding: 10px; border-radius: 5px; border-left: 4px solid #ff0000; margin: 5px 0; }
    .alarm-warning { background-color: #fff0cc; padding: 10px; border-radius: 5px; border-left: 4px solid #ffaa00; margin: 5px 0; }
    .logo-container { background-color: #FF7900; padding: 15px; border-radius: 8px; text-align: center; color: white; font-weight: bold; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

# Load external dataset
@st.cache_data
def load_industrial_data():
    """Load the NASA industrial dataset"""
    if os.path.exists('nasa_industrial_data.csv'):
        df = pd.read_csv('nasa_industrial_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        st.error("Dataset not found. Please run load_dataset.py first")
        return pd.DataFrame()

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'alarm_history' not in st.session_state:
    st.session_state.alarm_history = []

def add_alarm(level, title, message):
    """Add new alarm to history"""
    alarm = {
        'timestamp': datetime.now(),
        'level': level,
        'title': title,
        'message': message,
        'acknowledged': False
    }
    st.session_state.alarm_history.insert(0, alarm)

def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.markdown("""
        <div class="logo-container">
            HONEYWELL<br>PRANA
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<h1 class="main-header">NASA Industrial Data Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('**Real Turbofan Engine Monitoring Data**')
    with col3:
        st.metric("Data Point", st.session_state.current_index, "Current Index")
    st.markdown("---")

def render_kpi_cards(current_data):
    """Render KPI cards with real data"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("RUL", f"{current_data['rul']:.0f}", "Remaining Life")
        st.progress(current_data['rul'] / 150)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Zone 1 Temp", f"{current_data['sensor_3']:.1f}°C", "Sensor 3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Zone 2 Temp", f"{current_data['sensor_4']:.1f}°C", "Sensor 4")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Energy", f"{current_data['sensor_6']:.0f} kW", "Sensor 6")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        active_alarms = sum(1 for alarm in st.session_state.alarm_history if not alarm['acknowledged'])
        st.metric("Alarms", active_alarms, "Active")
        st.markdown('</div>', unsafe_allow_html=True)

def render_alarm_panel():
    """Render alarm panel"""
    st.subheader("🚨 Alarm Panel")
    
    unacknowledged = [alarm for alarm in st.session_state.alarm_history if not alarm['acknowledged']]
    
    if not unacknowledged:
        st.success("✅ No active alarms - System normal")
    else:
        for alarm in unacknowledged[:3]:
            alarm_class = "alarm-critical" if alarm['level'] == "CRITICAL" else "alarm-warning"
            st.markdown(f'''
            <div class="{alarm_class}">
                <strong>{alarm['timestamp'].strftime('%H:%M:%S')} - {alarm['title']}</strong><br>
                {alarm['message']}
            </div>
            ''', unsafe_allow_html=True)
            
            if st.button("Acknowledge", key=f"ack_{alarm['timestamp'].timestamp()}"):
                alarm['acknowledged'] = True
                st.session_state.current_index += 0  # Trigger update

def render_visualization(df, current_index):
    """Render data visualizations"""
    st.subheader("📊 Historical Data Trends")
    
    # Show last 100 data points
    start_idx = max(0, current_index - 100)
    recent_data = df.iloc[start_idx:current_index + 1]
    
    tab1, tab2, tab3 = st.tabs(["Temperature Trends", "Energy Consumption", "Degradation"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['sensor_3'], 
            name='Zone 1 Temp', line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['sensor_4'], 
            name='Zone 2 Temp', line=dict(color='red')
        ))
        fig.update_layout(title="Temperature Monitoring", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['sensor_6'], 
            name='Energy Consumption', line=dict(color='orange')
        ))
        fig.update_layout(title="Energy Usage", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'], 
            y=recent_data['rul'], 
            name='Remaining Useful Life', line=dict(color='green')
        ))
        fig.update_layout(title="Engine Degradation", height=400)
        st.plotly_chart(fig, use_container_width=True)

# Main application
df = load_industrial_data()

if not df.empty:
    render_header()
    
    # Get current data point
    current_data = df.iloc[st.session_state.current_index]
    
    # Check for anomalies in the dataset
    if current_data['anomaly'] == 1:
        if current_data['sensor_3'] > 200:
            add_alarm("CRITICAL", "Temperature Spike", 
                     f"Zone 1 temperature critical: {current_data['sensor_3']:.1f}°C")
        elif current_data['sensor_6'] < 400:
            add_alarm("WARNING", "Energy Drop", 
                     f"Energy consumption low: {current_data['sensor_6']:.0f} kW")
    
    render_kpi_cards(current_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_visualization(df, st.session_state.current_index)
    
    with col2:
        render_alarm_panel()
    
    # Navigation controls
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if st.button("⏮️ Previous"):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
    
    with nav_col2:
        st.slider("Data Point Index", 0, len(df)-1, st.session_state.current_index, 
                 key="index_slider", on_change=lambda: setattr(st.session_state, 'current_index', st.session_state.index_slider))
    
    with nav_col3:
        if st.button("Next ⏭️"):
            st.session_state.current_index = min(len(df)-1, st.session_state.current_index + 1)
    
    # Dataset info
    with st.expander("📊 Dataset Information"):
        st.write(f"**NASA Turbofan Engine Degradation Dataset**")
        st.write(f"- Total records: {len(df)}")
        st.write(f"- Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        st.write(f"- Anomalies detected: {df['anomaly'].sum()}")
        st.write(f"- Current RUL: {current_data['rul']:.0f} cycles")
        
        if st.checkbox("Show current data"):
            st.write(current_data)
    
    st.markdown("---")
    st.caption("Honeywell PrAna - NASA Industrial Data Dashboard | Real predictive maintenance data")

else:
    st.error("Please run 'load_dataset.py' first to create the industrial dataset")