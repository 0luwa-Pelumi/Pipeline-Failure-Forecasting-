import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Pipeline Fault Detection System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .normal {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .faulty {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class PipelineFaultDetector:
    def __init__(self):
        self.detection_model = None
        self.classification_model = None
        self.feature_names = None
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load the trained models"""
        try:
            # Load fault detection model
            detection_data = joblib.load('m1_fault_detection.plk')
            self.detection_model = detection_data['model']
            
            # Load fault classification model
            classification_data = joblib.load('m2_fault_type.plk')
            self.classification_model = classification_data['model']
            
            self.feature_names = [
                'pressure', 'flow_rate', 'temperature', 'valve_status',
                'pump_state', 'pump_speed', 'compressor_state', 'energy_consumption'
            ]
            
            # Initialize label encoder for fault types
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['blockage', 'degradation', 'leak', 'surge'])
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return False
    
    def predict(self, input_data):
        """Make predictions on input data"""
        try:
            # Convert input to DataFrame with correct feature names
            input_df = pd.DataFrame([input_data], columns=self.feature_names)
            
            # Fault detection prediction
            fault_prediction = self.detection_model.predict(input_df)[0]
            fault_probability = self.detection_model.predict_proba(input_df)[0]
            
            result = {
                'is_faulty': fault_prediction,
                'fault_probability': fault_probability[1],  # Probability of fault
                'normal_probability': fault_probability[0]  # Probability of normal
            }
            
            # If faulty, predict fault type
            if fault_prediction == 1:
                fault_type_pred = self.classification_model.predict(input_df)[0]
                fault_type_proba = self.classification_model.predict_proba(input_df)[0]
                
                result.update({
                    'fault_type': fault_type_pred,
                    'fault_type_name': self.label_encoder.inverse_transform([fault_type_pred])[0],
                    'fault_type_probabilities': {
                        self.label_encoder.inverse_transform([i])[0]: prob 
                        for i, prob in enumerate(fault_type_proba)
                    }
                })
            
            return result
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

def create_sample_data():
    """Create sample data for demonstration"""
    return {
        'normal_sample': {
            'pressure': 74.8, 'flow_rate': 4.5, 'temperature': 32.1,
            'valve_status': 1, 'pump_state': 1, 'pump_speed': 1336.8,
            'compressor_state': 1, 'energy_consumption': 25.8
        },
        'blockage_sample': {
            'pressure': 98.5, 'flow_rate': 1.6, 'temperature': 30.6,
            'valve_status': 2, 'pump_state': 1, 'pump_speed': 1368.7,
            'compressor_state': 1, 'energy_consumption': 37.1
        },
        'leak_sample': {
            'pressure': 56.7, 'flow_rate': 3.2, 'temperature': 31.2,
            'valve_status': 1, 'pump_state': 1, 'pump_speed': 1572.3,
            'compressor_state': 1, 'energy_consumption': 39.6
        }
    }

def main():
    # Initialize detector
    detector = PipelineFaultDetector()
    
    if not detector.load_models():
        st.error("Failed to load models. Please check if model files exist.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🔧 Pipeline Fault Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["Single Prediction", "Batch Prediction", "System Overview", "Model Information"]
    )
    
    # Sample data
    sample_data = create_sample_data()
    
    if app_mode == "Single Prediction":
        single_prediction_mode(detector, sample_data)
    elif app_mode == "Batch Prediction":
        batch_prediction_mode(detector)
    elif app_mode == "System Overview":
        system_overview_mode()
    elif app_mode == "Model Information":
        model_information_mode(detector)

def single_prediction_mode(detector, sample_data):
    """Single prediction interface"""
    st.header("🔍 Single Pipeline Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pipeline Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pressure = st.slider("Pressure (psi)", 40.0, 120.0, 75.0, 0.1)
                flow_rate = st.slider("Flow Rate (m³/s)", 1.0, 8.0, 4.5, 0.1)
                temperature = st.slider("Temperature (°C)", 25.0, 40.0, 32.0, 0.1)
            
            with col2:
                valve_status = st.selectbox("Valve Status", [0, 1, 2], format_func=lambda x: ["Closed", "Open", "Partially Open"][x])
                pump_state = st.selectbox("Pump State", [0, 1], format_func=lambda x: ["Off", "On"][x])
                pump_speed = st.slider("Pump Speed (RPM)", 0.0, 2000.0, 1000.0, 10.0)
            
            with col3:
                compressor_state = st.selectbox("Compressor State", [0, 1], format_func=lambda x: ["Off", "On"][x])
                energy_consumption = st.slider("Energy Consumption (kW)", 5.0, 60.0, 25.0, 0.1)
            
            # Quick load buttons
            st.subheader("Quick Load Samples")
            sample_col1, sample_col2, sample_col3 = st.columns(3)
            
            with sample_col1:
                if st.button("🚰 Load Normal Sample", use_container_width=True):
                    st.session_state.sample_data = sample_data['normal_sample']
            
            with sample_col2:
                if st.button("🚫 Load Blockage Sample", use_container_width=True):
                    st.session_state.sample_data = sample_data['blockage_sample']
            
            with sample_col3:
                if st.button("💧 Load Leak Sample", use_container_width=True):
                    st.session_state.sample_data = sample_data['leak_sample']
            
            submitted = st.form_submit_button("🔍 Analyze Pipeline", use_container_width=True)
    
    with col2:
        st.subheader("Sample Values")
        for sample_name, sample_values in sample_data.items():
            with st.expander(f"{sample_name.replace('_', ' ').title()}"):
                for key, value in sample_values.items():
                    st.write(f"{key}: {value}")
    
    # Process prediction
    if submitted:
        # Use sample data if loaded, otherwise use form data
        if hasattr(st.session_state, 'sample_data'):
            input_data = list(st.session_state.sample_data.values())
            del st.session_state.sample_data  # Clear after use
        else:
            input_data = [pressure, flow_rate, temperature, valve_status, 
                         pump_state, pump_speed, compressor_state, energy_consumption]
        
        # Make prediction
        result = detector.predict(input_data)
        
        if result:
            display_prediction_results(result, input_data)

def display_prediction_results(result, input_data):
    """Display prediction results with visualizations"""
    st.header("📊 Analysis Results")
    
    # Create columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        # Status card
        if result['is_faulty'] == 0:
            st.markdown(f"""
            <div class="prediction-box normal">
                <h3>✅ NORMAL OPERATION</h3>
                <p><strong>Confidence:</strong> {result['normal_probability']:.2%}</p>
                <p>The pipeline is operating within normal parameters.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box faulty">
                <h3>🚨 FAULT DETECTED</h3>
                <p><strong>Fault Type:</strong> {result['fault_type_name'].upper()}</p>
                <p><strong>Confidence:</strong> {result['fault_probability']:.2%}</p>
                <p>Immediate attention required!</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result['fault_probability'] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Fault Probability"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed probabilities
    st.subheader("Detailed Probabilities")
    
    if result['is_faulty'] == 1:
        # Fault type probabilities
        col1, col2, col3, col4 = st.columns(4)
        fault_probs = result['fault_type_probabilities']
        
        with col1:
            st.metric("Blockage", f"{fault_probs['blockage']:.2%}")
        with col2:
            st.metric("Degradation", f"{fault_probs['degradation']:.2%}")
        with col3:
            st.metric("Leak", f"{fault_probs['leak']:.2%}")
        with col4:
            st.metric("Surge", f"{fault_probs['surge']:.2%}")
        
        # Fault type probability chart
        fig = px.bar(
            x=list(fault_probs.keys()),
            y=list(fault_probs.values()),
            title="Fault Type Probabilities",
            labels={'x': 'Fault Type', 'y': 'Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Input parameters display
    st.subheader("Input Parameters")
    param_df = pd.DataFrame({
        'Parameter': detector.feature_names,
        'Value': input_data
    })
    st.dataframe(param_df, use_container_width=True)

def batch_prediction_mode(detector):
    """Batch prediction interface"""
    st.header("📁 Batch Pipeline Analysis")
    
    st.info("Upload a CSV file with pipeline data for batch analysis")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the data
            batch_data = pd.read_csv(uploaded_file)
            
            # Check if required columns exist
            required_columns = detector.feature_names
            missing_columns = [col for col in required_columns if col not in batch_data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            st.success(f"Successfully loaded {len(batch_data)} records")
            
            # Show data preview
            with st.expander("Data Preview"):
                st.dataframe(batch_data.head())
            
            if st.button("🚀 Run Batch Analysis", use_container_width=True):
                with st.spinner("Analyzing pipeline data..."):
                    # Make predictions
                    predictions = []
                    for _, row in batch_data.iterrows():
                        input_data = row[required_columns].values
                        result = detector.predict(input_data)
                        if result:
                            predictions.append(result)
                    
                    # Create results dataframe
                    results_df = batch_data.copy()
                    results_df['is_faulty'] = [pred['is_faulty'] for pred in predictions]
                    results_df['fault_probability'] = [pred['fault_probability'] for pred in predictions]
                    
                    if any(pred['is_faulty'] == 1 for pred in predictions):
                        results_df['fault_type'] = [
                            pred.get('fault_type_name', 'Normal') for pred in predictions
                        ]
                    
                    # Display results
                    st.subheader("Batch Analysis Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    total_records = len(results_df)
                    faulty_count = results_df['is_faulty'].sum()
                    
                    with col1:
                        st.metric("Total Records", total_records)
                    with col2:
                        st.metric("Faulty Pipelines", faulty_count)
                    with col3:
                        st.metric("Fault Rate", f"{(faulty_count/total_records)*100:.1f}%")
                    with col4:
                        st.metric("Normal Pipelines", total_records - faulty_count)
                    
                    # Fault distribution
                    if faulty_count > 0:
                        st.subheader("Fault Type Distribution")
                        fault_types = results_df[results_df['is_faulty'] == 1]['fault_type'].value_counts()
                        fig = px.pie(
                            values=fault_types.values,
                            names=fault_types.index,
                            title="Distribution of Fault Types"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="pipeline_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def system_overview_mode():
    """System overview and documentation"""
    st.header("📈 System Overview")
    
    st.markdown("""
    ## Pipeline Monitoring Dashboard
    
    This system provides real-time monitoring and fault detection for industrial pipeline systems
    using machine learning models trained on SCADA data.
    """)
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>97.5%</h2>
            <p>Fault Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Performance</h3>
            <h2>95.2%</h2>
            <p>Fault Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <h2>Real-time</h2>
            <p>Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🔧 Coverage</h3>
            <h2>4 Fault Types</h2>
            <p>Detected</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance and system diagram
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monitoring Parameters")
        parameters = [
            "Pressure", "Flow Rate", "Temperature", "Valve Status",
            "Pump State", "Pump Speed", "Compressor State", "Energy Consumption"
        ]
        
        for param in parameters:
            st.markdown(f"✅ {param}")
    
    with col2:
        st.subheader("Detected Fault Types")
        fault_types = [
            ("🚫 Blockage", "Partial or complete flow obstruction"),
            ("📉 Degradation", "Gradual performance deterioration"),
            ("💧 Leak", "Fluid leakage from pipeline"),
            ("⚡ Surge", "Pressure surge events")
        ]
        
        for fault, description in fault_types:
            st.markdown(f"**{fault}**")
            st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
            st.markdown("---")

def model_information_mode(detector):
    """Model information and technical details"""
    st.header("🤖 Model Information")
    
    st.markdown("""
    ## Machine Learning Models
    
    This system uses ensemble machine learning models for pipeline fault detection and classification.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fault Detection Model")
        st.markdown("""
        **Algorithm**: Random Forest Classifier
        - **Ensemble Method**: Combines multiple decision trees
        - **Training Data**: 800 samples (80% of dataset)
        - **Test Accuracy**: 97.5%
        - **ROC AUC**: 96.4%
        
        **Hyperparameters**:
        - n_estimators: 200
        - max_depth: 10
        - min_samples_leaf: 2
        - class_weight: balanced
        """)
    
    with col2:
        st.subheader("Fault Classification Model")
        st.markdown("""
        **Algorithm**: Random Forest Classifier
        - **Classes**: 4 fault types
        - **Training Data**: 306 faulty samples
        - **Test Accuracy**: 95.2%
        
        **Fault Types**:
        1. Blockage (0)
        2. Degradation (1) 
        3. Leak (2)
        4. Surge (3)
        
        **Hyperparameters**:
        - n_estimators: 200
        - max_depth: 15
        - min_samples_leaf: 1
        """)
    
    # Feature importance (if available)
    st.subheader("Feature Importance")
    st.markdown("""
    The models consider the following features with their relative importance:
    
    1. **Pressure** (High correlation with faults)
    2. **Flow Rate** (Decreases during faults)
    3. **Energy Consumption** (Increases during faults)
    4. **Pump Speed** (Abnormal patterns indicate issues)
    5. **Temperature** (Secondary indicator)
    6. **Valve/Compressor States** (Operational context)
    """)
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown("""
        **Fault Detection Performance**:
        - Precision: 98%
        - Recall: 93%
        - F1-Score: 96%
        - ROC AUC: 96.4%
        """)
    
    with metrics_col2:
        st.markdown("""
        **Fault Classification Performance**:
        - Blockage Precision: 100%
        - Degradation Precision: 100%
        - Leak Precision: 93%
        - Surge Precision: 87%
        """)

if __name__ == "__main__":
    main()
