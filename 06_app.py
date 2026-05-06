import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, if not available use mock
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🏦 Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - COLORFUL UI
# ============================================================================
st.markdown("""
<style>
            
    /* ================================================================
       DROPDOWN/SELECT OPTIONS - Make visible
       ================================================================ */
    
    .stSelectbox > div > div > div {
        background-color: #23263A !important;
        color: #FFFFFF !important;
    }
    
    /* Dropdown menu options */
    div[role="option"] {
        background-color: #23263A !important;
        color: #FFFFFF !important;
    }
    
    div[role="option"]:hover {
        background-color: #00C896 !important;
        color: #181A20 !important;
    }
    
    /* Dropdown list container */
    .stSelectbox div[role="listbox"] {
        background-color: #23263A !important;
    }
    
    .stSelectbox div[role="option"] {
        background-color: #23263A !important;
        color: #FFFFFF !important;
    }
    
    .stSelectbox [role="option"] {
        background-color: #23263A !important;
        color: #FFFFFF !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stSelectbox [role="option"]:hover {
        background-color: #00C896 !important;
        color: #181A20 !important;
    }
    
    /* Dropdown arrow */
    .stSelectbox [data-testid="stSelectboxTag"] {
        background-color: #23263A !important;
        color: #FFFFFF !important;
    }
            
            
    /* ================================================================
       MASTER COLOR & VISIBILITY FIXES - ALL TEXT WHITE
       ================================================================ */
    
    /* Global text - Force everything to white */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    html, body, .main, [data-testid="stMainBlockContainer"] {
        background: #181A20 !important;
        color: #FFFFFF !important;
    }
    
    /* ALL LABELS - Master override */
    label, .stLabel {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* ================================================================
       HEADINGS & TEXT
       ================================================================ */
    
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
    }
    
    h1 {
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 900;
    }
    
    h2 {
        border-bottom: 3px solid #00C896;
        padding-bottom: 0.8rem;
        margin-top: 1.5rem;
        font-weight: 800;
    }
    
    h3 {
        color: #00C896 !important;
        font-weight: 700;
    }
    
    p, span, div {
        color: #FFFFFF !important;
    }
    
    /* ================================================================
       STREAMLIT COMPONENTS - TEXT & STYLING
       ================================================================ */
    
    .stMarkdown, .stText, .stDataFrame, .stTable, .stMetric, .stForm, 
    .stSelectbox, .stSlider, .stCheckbox, .stRadio, .stButton, .stTabs, 
    .stInfo, .stWarning, .stError, .stSuccess {
        color: #FFFFFF !important;
    }
    
    /* ================================================================
       SIDEBAR STYLING
       ================================================================ */
    
    [data-testid="stSidebar"] {
        background: #23263A !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }
    
    /* ================================================================
       SIDEBAR NAVIGATION MENU - MAKE TEXT VISIBLE
       ================================================================ */
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label > div {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label > div > span {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 1.05rem !important;
    }
    
    [data-testid="stSidebar"] [role="radiogroup"] {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] [role="radio"] {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio span {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 1rem !important;
    }
    
    /* ================================================================
       RADIO BUTTONS (Navigation & Forms)
       ================================================================ */
    
    .stRadio > label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    .stRadio > label > div {
        color: #FFFFFF !important;
    }
    
    .stRadio > label span {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    
    .stRadio div[role="radiogroup"] label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    
    .stRadio div[role="radiogroup"] span {
        color: #FFFFFF !important;
    }
    
    .stRadio > div, 
    .stRadio div[role="radio"], 
    .stRadio div[role="radiogroup"] {
        color: #FFFFFF !important;
    }
    
    /* ================================================================
       FORM & INPUT LABELS - MAKE ALL VISIBLE
       ================================================================ */
    
    .stForm label, 
    .stForm div[role="radiogroup"] label,
    [data-testid="stWidgetLabel"] > label,
    [data-testid="stWidgetLabel"] > div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    div[data-testid="stWidgetLabel"] {
        color: #FFFFFF !important;
    }
    
    div[data-testid="stWidgetLabel"] label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Specific input labels */
    .stNumberInput label, 
    .stSelectbox label, 
    .stTextInput label, 
    .stSlider label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    .stCheckbox > label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    .stCheckbox label {
        color: #FFFFFF !important;
    }
    
    /* ================================================================
       SLIDER STYLING - MAKE TRACK & HANDLE VISIBLE
       ================================================================ */
    
    .stSlider > div > div > div {
        background: #23263A !important;
    }
    
    .stSlider [role="slider"] {
        color: #FFE066 !important;
    }
    
    .stSlider .rc-slider {
        background-color: #23263A !important;
    }
    
    .stSlider .rc-slider-track {
        background-color: #FFE066 !important;
        height: 8px !important;
    }
    
    .stSlider .rc-slider-rail {
        background-color: #404854 !important;
        height: 8px !important;
    }
    
    .stSlider .rc-slider-handle {
        background-color: #FFE066 !important;
        border-color: #FFD700 !important;
        width: 16px !important;
        height: 16px !important;
        margin-top: -4px !important;
    }
    
    .stSlider .rc-slider-mark-text {
        color: #FFFFFF !important;
    }
    
    /* ================================================================
       INPUT FIELDS
       ================================================================ */
    
    .stSelectbox > div > div, 
    .stTextInput > div > input, 
    .stNumberInput > div > input {
        background-color: #23263A !important;
        border: 1.5px solid #00C896 !important;
        color: #FFFFFF !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
    }
    
    /* ================================================================
       BUTTONS
       ================================================================ */
    
    .stButton > button {
        background: linear-gradient(135deg, #00C896 0%, #FFE066 100%) !important;
        color: #181A20 !important;
        font-weight: 900;
        font-size: 1.1rem;
        border-radius: 0.8rem;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        padding: 1rem 2rem;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FFE066 0%, #00C896 100%) !important;
        color: #181A20 !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        filter: brightness(1.1);
    }
    
    /* ================================================================
       METRIC CARDS
       ================================================================ */
    
    .metric-card {
        background: #23263A;
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        text-align: center;
        border: 2px solid #00C896;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0.5rem 0;
        color: #FFE066;
        text-shadow: 1px 1px 2px #23263A;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #FFFFFF;
        font-weight: 600;
    }
    
    .stMetric {
        background: #23263A !important;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #00C896 !important;
        color: #FFE066 !important;
    }
    
    .stMetric > div > label, 
    .stMetric > div > div {
        color: #FFE066 !important;
        font-weight: 900;
    }
    
    /* ================================================================
       SIDEBAR INFO BOX
       ================================================================ */
    
    .sidebar-info {
        background: #23263A;
        padding: 1.5rem 1.5rem 1.5rem 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #00C896;
        margin-top: 1rem;
        color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    }
    
    .sidebar-info strong {
        color: #FFE066 !important;
    }
    
    /* ================================================================
       FORM STYLING
       ================================================================ */
    
    .stForm {
        background: #23263A;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #00C896;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    }
    
    .stForm h3 {
        color: #FFE066 !important;
    }
    
    /* ================================================================
       ALERT BOXES (Success, Warning, Danger)
       ================================================================ */
    
    .warning-box {
        background: #FFB347;
        border-left: 6px solid #FFE066;
        color: #181A20;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .success-box {
        background: #00C896;
        border-left: 6px solid #FFE066;
        color: #181A20;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .danger-box {
        background: #FF1744;
        border-left: 6px solid #FFE066;
        color: #FFFFFF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .warning-box h3, 
    .success-box h3, 
    .danger-box h3,
    .warning-box strong, 
    .success-box strong, 
    .danger-box strong {
        color: #FFD700 !important;
        font-weight: 900;
    }
    
    /* ================================================================
       STATUS MESSAGES
       ================================================================ */
    
    .stSuccess {
        background-color: #00C89622 !important;
        color: #00C896 !important;
    }
    
    .stError {
        background-color: #FF174422 !important;
        color: #FF1744 !important;
    }
    
    .stWarning {
        background-color: #FFB34722 !important;
        color: #FFB347 !important;
    }
    
    .stInfo {
        background-color: #23263A !important;
        color: #FFE066 !important;
    }
    
    /* ================================================================
       TABS
       ================================================================ */
    
    .stTabs > div > div > button {
        color: #FFFFFF !important;
        border-bottom: 3px solid transparent;
    }
    
    .stTabs > div > div > button[aria-selected="true"] {
        border-bottom-color: #00C896 !important;
        color: #00C896 !important;
    }
    
    /* ================================================================
       SPINNER
       ================================================================ */
    
    .stSpinner > div {
        border-color: #FFD700 !important;
    }
    
    /* ================================================================
       PLOTLY DARK THEME PATCH
       ================================================================ */
    
    .plotly-graph-div {
        background-color: rgba(24, 26, 32, 0.1) !important;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================================
# PLOTLY DARK THEME PATCH (for chart text contrast)
# ============================================================================
import plotly.io as pio
pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"].layout.font.color = "#FFFFFF"
pio.templates["plotly_dark"].layout.legend.font.color = "#FFFFFF"
pio.templates["plotly_dark"].layout.xaxis.color = "#FFFFFF"
pio.templates["plotly_dark"].layout.yaxis.color = "#FFFFFF"


@st.cache_resource
def load_preprocessor():
    """Load the fitted preprocessor from joblib file"""
    try:
        import joblib
        preprocessor = joblib.load('preprocessor.pkl')
        st.sidebar.success("✅ Preprocessor loaded")
        return preprocessor
    except Exception as e:
        st.sidebar.error(f"⚠️ Could not load preprocessor: {e}")
        st.sidebar.info("Using fallback preprocessing")
        return None

@st.cache_resource
def load_model():
    """Load the trained Keras model"""
    if not TF_AVAILABLE:
        st.sidebar.error("❌ TensorFlow not available")
        return None
    
    try:
        model = keras_load_model('churn_model.keras')
        st.sidebar.success("✅ Model loaded")
        return model
    except Exception as e:
        st.sidebar.error(f"⚠️ Could not load model: {e}")
        return None

# ============================================================================
# PREPROCESSING FUNCTION (FIXED)
# ============================================================================
def preprocess_input(data_dict, preprocessor=None):
    """
    Preprocess customer input to match training data format
    CORRECT: 8 numerical + 3 categorical (after one-hot) + 3 binary = 14 features
    BUT model expects 16, so we need to check actual columns
    """
    
    try:
        # Create DataFrame with ALL features (before one-hot encoding)
        features = pd.DataFrame({
            'credit_score': [float(data_dict['credit_score'])],
            'age': [float(data_dict['age'])],
            'tenure': [float(data_dict['tenure'])],
            'balance': [float(data_dict['balance'])],
            'num_of_products': [float(data_dict['num_of_products'])],
            'has_cr_card': [float(data_dict['has_cr_card'])],
            'is_active_member': [float(data_dict['is_active_member'])],
            'estimated_salary': [float(data_dict['estimated_salary'])],
            'geography': [str(data_dict['geography'])],
            'gender': [str(data_dict['gender'])],
        })
        
        # STEP 1: Create engineered features
        features['balance_per_product'] = features['balance'] / (features['num_of_products'] + 1)
        features['engagement_score'] = features['num_of_products'] * features['is_active_member']
        features['zero_balance'] = (features['balance'] == 0).astype(float)
        
        # STEP 2: Create age groups with same binning as training
        features['age_group'] = pd.cut(
            features['age'],
            bins=[18, 30, 45, 60, 100],
            labels=['(18, 30]', '(30, 45]', '(45, 60]', '(60, 100]'],
            include_lowest=True
        ).astype(str)
        
        # STEP 3: Apply preprocessor if available
        if preprocessor is not None:
            X_processed = preprocessor.transform(features)
            X_processed = np.asarray(X_processed, dtype=np.float64)
            return X_processed
        else:
            st.error("❌ Preprocessor not loaded. Cannot proceed.")
            return None
        
    except Exception as e:
        st.error(f"❌ Preprocessing error: {str(e)}")
        raise


def predict_churn(model, features):
    """
    Make prediction using the trained model
    Input: preprocessed features array
    Output: churn probability (0-1) and classification using threshold=0.35
    """
    if model is None:
        st.error("❌ Model not loaded")
        return None
    
    try:
        # Ensure float32 for TensorFlow
        features = np.asarray(features, dtype=np.float32)
        
        # Get the expected shape from the model
        expected_features = model.input_shape[1]
        
        # Validate input shape
        if features.shape[1] != expected_features:
            st.error(f"❌ Input shape error: Expected ({1}, {expected_features}), got {features.shape}")
            return None
        
        # Get raw probability
        prediction = model.predict(features, verbose=0)
        churn_prob = float(prediction[0][0])
        
        return churn_prob
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        raise

def create_risk_gauge(churn_probability):
    """Create an interactive gauge chart for churn risk"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=churn_probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk Score"},
        number={'suffix': "%", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#FFD700"},
            'steps': [
                {'range': [0, 33], 'color': "#00D084"},
                {'range': [33, 66], 'color': "#FFA500"},
                {'range': [66, 100], 'color': "#FF1744"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=400, font={'size': 12}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("# 🏦 Bank Churn Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📍 Navigation",
    ["🏠 Dashboard", "🔮 Predict Churn", "📊 Analytics", "ℹ️ About Model"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="sidebar-info">
    <strong>📌 Quick Info</strong><br/>
    <strong>Model Type:</strong> Artificial Neural Network (ANN)<br/>
    <strong>Input Features:</strong> 17 (after preprocessing)<br/>
    • 8 Numerical (scaled)<br/>
    • 6 Categorical (one-hot encoded)<br/>
    • 3 Binary (passthrough)<br/>
    <strong>Architecture:</strong> Input → Dense(32,relu) → Output(sigmoid)<br/>
    <strong>Best Metric:</strong> Recall = 72% (captures churners)<br/>
    <strong>Class Imbalance:</strong> Handled with class weights
</div>
""", unsafe_allow_html=True)

# Load model and preprocessor
model = load_model()
preprocessor = load_preprocessor()

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    st.title("🏦 Bank Customer Churn Prediction Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Dataset Size</div>
            <div class="metric-value">10,000</div>
            <div class="metric-label">Customers Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Churn Rate</div>
            <div class="metric-value">20%</div>
            <div class="metric-label">2,000 Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">80%</div>
            <div class="metric-label">Optimized for Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Recall (Churners)</div>
            <div class="metric-value">72%</div>
            <div class="metric-label">Best Performing Metric</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Performance Comparison")
        models_data = {
            'Model': ['Baseline', 'Threshold\nTuned', 'Class\nWeighted', 'SMOTE', 'Hyperparameter\nTuned', 'Final\nOptimized'],
            'Accuracy': [0.87, 0.85, 0.80, 0.82, 0.86, 0.80],
            'Churn Recall': [0.48, 0.67, 0.73, 0.68, 0.43, 0.72],
            'AUC-ROC': [0.86, 0.85, 0.82, 0.80, 0.86, 0.84]
        }
        models_df = pd.DataFrame(models_data)
        
        fig = px.bar(
            models_df,
            x='Model',
            y=['Accuracy', 'Churn Recall', 'AUC-ROC'],
            barmode='group',
            title='Model Performance Metrics',
            color_discrete_sequence=['#FFD700', '#00D084', '#FF1744']
        )
        fig.update_layout(
            height=400, hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.1)',
            font_color='#FFFFFF', legend=dict(font=dict(color='#FFFFFF')), xaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF')), yaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Key Features Impacting Churn")
        features = ['Age', 'Engagement\nScore', 'Tenure', 'Number of\nProducts', 'Active\nMember', 'Credit\nScore']
        importance = [0.28, 0.22, 0.18, 0.15, 0.12, 0.05]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Feature Importance in Churn Prediction',
            labels={'x': 'Importance Score', 'y': 'Features'},
            color=importance,
            color_continuous_scale=['#FF1744', '#FFA500', '#FFD700', '#00D084']
        )
        fig.update_layout(
            height=400, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.1)',
            font_color='#FFFFFF', legend=dict(font=dict(color='#FFFFFF')), xaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF')), yaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📊 Feature Engineering Details")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <strong>Balance per Product</strong>
            <br/><br/>
            balance / (num_products + 1)
            <br/><br/>
            Measures customer account value per service
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <strong>Engagement Score</strong>
            <br/><br/>
            num_products × is_active
            <br/><br/>
            Captures customer activity level
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <strong>Age Group Bins</strong>
            <br/><br/>
            18-30, 30-45, 45-60, 60+
            <br/><br/>
            Segmentation for better patterns
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <strong>Zero Balance Flag</strong>
            <br/><br/>
            balance == 0 ? 1 : 0
            <br/><br/>
            Risk indicator for inactive accounts
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE: PREDICT CHURN
# ============================================================================
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 0.8rem; border: 2px solid #FFD700;">
        <strong style="color: #FFD700; font-size: 1.1rem;">📝 Enter customer details below to predict churn probability.</strong>
        <br/><span style="color: #E0E0E0;">The model uses 17 engineered features to provide accurate risk assessment.</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    with st.form("churn_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Information")
            
            age = st.slider("Age", min_value=18, max_value=95, value=45, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            geography = st.selectbox("Country", ["France", "Germany", "Spain"])
            
            st.subheader("💳 Account Details")
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=600, step=1)
            tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5, step=1)
            
        with col2:
            st.subheader("💰 Financial Information")
            
            balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=500000.0, value=100000.0, step=1000.0)
            estimated_salary = st.number_input("Estimated Annual Salary ($)", min_value=11000.0, max_value=200000.0, value=75000.0, step=1000.0)
            
            st.subheader("📱 Product & Membership")
            
            num_of_products = st.selectbox("Number of Products Used", [1, 2, 3, 4])
            has_cr_card = st.checkbox("Has Credit Card", value=True)
            is_active_member = st.checkbox("Is Active Member", value=True)
        
        st.markdown("---")
        
        submit_button = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)
        
        if submit_button:
            if model is None:
                st.error("❌ Model not loaded. Cannot make predictions.")
            else:
                try:
                    # Prepare data
                    input_data = {
                        'age': age,
                        'gender': gender,
                        'geography': geography,
                        'credit_score': credit_score,
                        'tenure': tenure,
                        'balance': balance,
                        'estimated_salary': estimated_salary,
                        'num_of_products': int(num_of_products),
                        'has_cr_card': int(has_cr_card),
                        'is_active_member': int(is_active_member)
                    }
                    
                    # Preprocess
                    with st.spinner("🔄 Processing customer data..."):
                        features = preprocess_input(input_data, preprocessor)
                    
                    # Validate preprocessed features
                    if features is None:
                        st.error("❌ Preprocessing failed")
                    else:
                        st.success(f"✅ Features preprocessed: shape {features.shape}, dtype {features.dtype}")
                        
                        # Make prediction
                        with st.spinner("🤖 Running model prediction..."):
                            churn_prob = predict_churn(model, features)
                        
                        if churn_prob is not None:
                            # Display Results
                            st.markdown("---")
                            st.subheader("📊 Prediction Results")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            # Apply threshold of 0.35
                            CHURN_THRESHOLD = 0.35
                            churn_prediction = 1 if churn_prob > CHURN_THRESHOLD else 0
                            
                            with col1:
                                # Show both probability and binary prediction
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=churn_prob * 100,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Churn Probability"},
                                    number={'suffix': "%", 'font': {'size': 20}},
                                    gauge={
                                        'axis': {'range': [None, 100]},
                                        'bar': {'color': "#FFD700"},
                                        'steps': [
                                            {'range': [0, 35], 'color': "#00D084"},
                                            {'range': [35, 66], 'color': "#FFA500"},
                                            {'range': [66, 100], 'color': "#FF1744"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 35  # Show threshold line at 0.35
                                        }
                                    }
                                ))
                                fig.update_layout(height=400, font={'size': 12}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Show binary prediction prominently
                                st.markdown(f"""
                                <div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 1rem; border: 3px solid #FFD700;">
                                    <div style="font-size: 3.5rem; font-weight: 900; margin: 1rem 0; color: #FFD700;">
                                        {churn_prediction}
                                    </div>
                                    <div style="font-size: 1.3rem; color: #E0E0E0; font-weight: 700;">Prediction</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown("---")
                                
                                if churn_prediction == 1:
                                    st.markdown("""
                                    <div class="danger-box">
                                    <h3>🔴 WILL CHURN</h3>
                                    Prediction: <strong>1</strong>
                                    <br/>
                                    Churn Probability: {:.1f}%
                                    <br/>
                                    This customer is predicted to CHURN.
                                    </div>
                                    """.format(churn_prob * 100), unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="success-box">
                                    <h3>🟢 WILL NOT CHURN</h3>
                                    Prediction: <strong>0</strong>
                                    <br/>
                                    Churn Probability: {:.1f}%
                                    <br/>
                                    This customer is predicted to STAY.
                                    </div>
                                    """.format(churn_prob * 100), unsafe_allow_html=True)
                            
                            st.markdown("---")
                            
                            st.subheader("🔍 Customer Profile Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Age", f"{age} years")
                            with col2:
                                st.metric("Tenure", f"{tenure} years")
                            with col3:
                                st.metric("Balance", f"${balance:,.0f}")
                            with col4:
                                st.metric("Salary", f"${estimated_salary:,.0f}")
                            
                            st.markdown("---")
                            
                            # Key Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Churn Prediction", f"{churn_prediction}", "0=Stay, 1=Churn")
                            with col2:
                                st.metric("Churn Probability", f"{churn_prob*100:.2f}%", f"Threshold: 35%")
                            with col3:
                                st.metric("Risk Level", "🔴 High" if churn_prediction == 1 else "🟢 Low", "")
                            
                            st.markdown("---")
                            
                            st.subheader("💡 Recommendations")
                            
                            recommendations = []
                            
                            if churn_prediction == 1:  # Will Churn
                                recommendations.append("🎁 **URGENT: Offer exclusive loyalty rewards** - Customer predicted to churn")
                                if tenure < 2:
                                    recommendations.append("📞 **Early engagement call** - New customers need more support")
                                if balance == 0:
                                    recommendations.append("💰 **Promote savings products** - Zero balance is a critical risk signal")
                                if is_active_member == False:
                                    recommendations.append("🚀 **Re-engagement campaign** - Inactive members show higher churn")
                            else:  # Will Not Churn
                                recommendations.append("✅ **Maintain relationship** - Continue current engagement strategy")
                                if num_of_products == 1:
                                    recommendations.append("🔗 **Cross-sell opportunities** - Suggest complementary products")
                                recommendations.append("📊 **Regular monitoring** - Keep tracking this customer's behavior")
                            
                            for rec in recommendations:
                                st.write(rec)
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.error("Please check the error details above and try again.")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
elif page == "📊 Analytics":
    st.title("📊 Analytics & Insights")
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 0.8rem; border: 2px solid #FFD700;">
        <strong style="color: #FFD700; font-size: 1.1rem;">📈 Explore patterns and insights from the bank churn dataset.</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Churn Distribution by Age Group")
        
        age_churn = pd.DataFrame({
            'Age Group': ['18-30', '30-45', '45-60', '60+'],
            'Churn Rate': [0.17, 0.18, 0.28, 0.37],
            'Non-Churn Rate': [0.83, 0.82, 0.72, 0.63]
        })
        
        fig = px.bar(
            age_churn,
            x='Age Group',
            y=['Churn Rate', 'Non-Churn Rate'],
            title='Churn by Age Group',
            barmode='stack',
            color_discrete_map={'Churn Rate': '#FF1744', 'Non-Churn Rate': '#00D084'}
        )
        fig.update_layout(
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.1)',
            font_color='#FFFFFF', legend=dict(font=dict(color='#FFFFFF')), xaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF')), yaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Churn Distribution by Country")
        
        country_churn = pd.DataFrame({
            'Country': ['France', 'Germany', 'Spain'],
            'Churn Rate': [0.16, 0.32, 0.17]
        })
        
        fig = px.pie(
            country_churn,
            values='Churn Rate',
            names='Country',
            title='Churn Rate by Country',
            color_discrete_sequence=['#FFD700', '#FFA500', '#FF1744']
        )
        fig.update_layout(
            height=400, paper_bgcolor='rgba(0,0,0,0)',
            font_color='#FFFFFF', legend=dict(font=dict(color='#FFFFFF')), xaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF')), yaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Churn by Gender")
        
        gender_churn = pd.DataFrame({
            'Gender': ['Male', 'Female'],
            'Churn Rate': [0.16, 0.25]
        })
        
        fig = px.bar(
            gender_churn,
            x='Gender',
            y='Churn Rate',
            title='Churn Rate by Gender',
            color='Churn Rate',
            color_continuous_scale=['#00D084', '#FF1744']
        )
        fig.update_layout(
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.1)',
            font_color='#FFFFFF', legend=dict(font=dict(color='#FFFFFF')), xaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF')), yaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏦 Churn by Tenure")
        
        tenure_churn = pd.DataFrame({
            'Tenure (Years)': ['0-1', '2-3', '4-6', '7-10'],
            'Churn Rate': [0.27, 0.21, 0.19, 0.12]
        })
        
        fig = px.line(
            tenure_churn,
            x='Tenure (Years)',
            y='Churn Rate',
            markers=True,
            title='Churn Rate by Customer Tenure',
            line_shape='spline'
        )
        fig.update_traces(line=dict(color='#FFD700', width=3), marker=dict(size=10, color='#FF1744'))
        fig.update_layout(
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.1)',
            font_color='#FFFFFF', legend=dict(font=dict(color='#FFFFFF')), xaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF')), yaxis=dict(title_font=dict(color='#FFFFFF'), tickfont=dict(color='#FFFFFF'))
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: ABOUT MODEL
# ============================================================================
elif page == "ℹ️ About Model":
    st.title("ℹ️ About the Churn Prediction Model")
    
    st.subheader("🎯 Model Architecture")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 0.8rem; border-left: 6px solid #FFD700; color: #E0E0E0;">
        <strong style="color: #FFD700; font-size: 1.1rem;">The model is an Artificial Neural Network (ANN) specifically optimized for customer churn prediction:</strong>
        <ul>
            <li><strong style="color: #FFD700;">Input Layer:</strong> 17 features (preprocessed and engineered)</li>
            <li><strong style="color: #FFD700;">Hidden Layer:</strong> Dense(32, activation='relu')</li>
            <li><strong style="color: #FFD700;">Output Layer:</strong> Dense(1, activation='sigmoid')</li>
            <li><strong style="color: #FFD700;">Loss Function:</strong> Binary Crossentropy</li>
            <li><strong style="color: #FFD700;">Optimizer:</strong> Adam</li>
            <li><strong style="color: #FFD700;">Training Technique:</strong> Class weight balancing to handle 20% churn imbalance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    st.subheader("📊 Dataset Characteristics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Total Records:** 10,000 customers")
    with col2:
        st.info("**Original Features:** 10")
    with col3:
        st.info("**Churn Rate:** 20% (2,000 customers)")
    
    st.subheader("🛠️ Preprocessing Pipeline")

    st.markdown("""

    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 0.8rem; border-left: 6px solid #FFD700; color: #E0E0E0;">
        <strong style="color: #FFD700; font-size: 1.1rem;">Key Techniques Applied:</strong>
        <ul>

    <li><strong style="color: #FFD700;">Numerical Features (8):</strong> Scaled with StandardScaler

    credit_score, age, tenure, balance, num_of_products, estimated_salary, balance_per_product, engagement_score</li>

    <li><strong style="color: #FFD700;">Categorical Features (6):</strong> One-hot encoded with drop_first=True

    geography: Germany, Spain (France dropped as reference) <br>
    gender: Male (Female dropped as reference) <br>
    age_group: (30, 45], (45, 60], (60, 100] (18, 30] dropped)</li>

    <li><strong style="color: #FFD700;">Binary Features (3):</strong> Passthrough (unchanged)

    has_cr_card, is_active_member, zero_balance</li>
    </div>
    """, unsafe_allow_html=True)


    st.subheader("🔧 Feature Engineering Details")

    st.markdown("""

    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 0.8rem; border-left: 6px solid #FFD700; color: #E0E0E0;">
        <strong style="color: #FFD700; font-size: 1.1rem;">Key Techniques Applied:</strong>
        <ul>

    <li><strong style="color: #FFD700;">balance_per_product</strong> = balance / (num_products + 1)<br>
    Measures wealth distribution across services</li>

    <li><strong style="color: #FFD700;">engagement_score</strong> = num_products × is_active_member<br>
    Captures customer engagement level</li>

    <li><strong style="color: #FFD700;">zero_balance</strong> = Binary indicator (balance == 0)<br>
    Risk signal for inactive accounts</li>

    <li><strong style="color: #FFD700;">age_group</strong> = Binned into [18-30], [30-45], [45-60], [60+]<br>
    Creates meaningful demographic segments</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    st.subheader("🚀 Model Training & Optimization")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 0.8rem; border-left: 6px solid #FFD700; color: #E0E0E0;">
        <strong style="color: #FFD700; font-size: 1.1rem;">Key Techniques Applied:</strong>
        <ul>
            <li><strong style="color: #FFD700;">Class Weight Balancing</strong> - Penalized minority class (churn) more heavily</li>
            <li><strong style="color: #FFD700;">Early Stopping</strong> - Monitored val_loss with patience=20 to prevent overfitting</li>
            <li><strong style="color: #FFD700;">Train-Test Split</strong> - 80-20 split with stratification to maintain class distribution</li>
            <li><strong style="color: #FFD700;">Hyperparameter Tuning</strong> - KerasTuner with RandomSearch (10 trials)</li>
            <li><strong style="color: #FFD700;">SMOTE Oversampling</strong> - Synthetic minority oversampling tested</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    st.subheader("📈 Final Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "80%", "Overall correctness")
    with col2:
        st.metric("Recall (Churn)", "72%", "Capture rate")
    with col3:
        st.metric("Precision", "51%", "Prediction accuracy")
    with col4:
        st.metric("AUC-ROC", "0.84", "Discrimination")
    
    st.markdown("---")
    st.subheader("🎯 Why These Metrics Matter?")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 0.8rem; border-left: 6px solid #FFD700; color: #E0E0E0;">
        <strong style="color: #FFD700; font-size: 1.1rem;">For bank churn prediction, Recall is prioritized because:</strong>
        <ul>
            <li>✅ <strong style="color: #FFD700;">Missing a churner (False Negative)</strong> = Revenue loss</li>
            <li>✅ <strong style="color: #FFD700;">Falsely predicting churn (False Positive)</strong> = Unnecessary retention cost</li>
            <li>✅ <strong style="color: #FFD700;">72% recall</strong> = Capturing ~7 out of 10 actual churners</li>
            <li>✅ <strong style="color: #FFD700;">Targeted, cost-effective retention interventions</strong> = Better business outcomes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    st.subheader("⚙️ Technical Stack")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <strong>Data Processing:</strong><br/>
            • Pandas<br/>
            • NumPy<br/>
            • Scikit-learn
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <strong>Deep Learning:</strong><br/>
            • TensorFlow<br/>
            • Keras
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <strong>Deployment:</strong><br/>
            • Streamlit<br/>
            • Plotly
        </div>
        """, unsafe_allow_html=True)