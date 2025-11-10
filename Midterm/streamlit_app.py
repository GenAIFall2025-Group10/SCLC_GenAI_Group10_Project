"""
OncoDetect-AI: Enhanced Streamlit UI for Unified API
Professional medical interface with your existing structure
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
from typing import Dict, Any, Optional

# ============================================
# CONFIGURATION
# ============================================

# Single Unified API
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="OncoDetect-AI | Precision Oncology Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Page Headers */
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .page-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .page-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
        font-size: 1.1rem;
    }
    
    /* Risk Cards Enhanced */
    .risk-card {
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 1.25rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInScale 0.5s ease;
        position: relative;
        overflow: hidden;
    }
    
    .risk-card::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .risk-very-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
        border: 2px solid rgba(201, 42, 42, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff8787 0%, #f03e3e 100%);
        color: white;
        border: 2px solid rgba(240, 62, 62, 0.3);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
        color: #2e2e2e;
        border: 2px solid rgba(250, 176, 5, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
        border: 2px solid rgba(55, 178, 77, 0.3);
    }
    
    /* Clinical Report Enhanced */
    .clinical-report {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        font-family: 'Inter', sans-serif;
        white-space: pre-wrap;
        font-size: 14px;
        line-height: 1.8;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    
    .clinical-report::before {
        content: "📋";
        position: absolute;
        top: -10px;
        left: 20px;
        background: white;
        padding: 0 10px;
        font-size: 1.5rem;
    }
    
    /* Analysis Section Enhanced */
    .analysis-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.05);
        border-left: 3px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .analysis-section:hover {
        box-shadow: 0 5px 25px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }
    
    /* Info Cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.05);
        border-top: 3px solid #667eea;
        height: 100%;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    /* Buttons Enhanced */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Enhancement */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 2px solid #dee2e6;
    }
    
    /* Sidebar Header */
    .sidebar-header {
        text-align: center;
        padding: 1.5rem;
        background: white;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem;
    }
    
    .status-active {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
    }
    
    .status-inactive {
        background: linear-gradient(135deg, #ff8787 0%, #f03e3e 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
        color: #2e2e2e;
    }
    
    /* Expander Enhancement */
    .streamlit-expanderHeader {
        background: white !important;
        border-radius: 10px !important;
        border: 1px solid #e9ecef !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f8f9fa !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    }
    
    /* Tab Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: white;
        padding: 0.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Subtype Cards */
    .subtype-card {
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        animation: fadeInScale 0.5s ease;
    }
    
    .subtype-a {
        background: linear-gradient(135deg, #667eea 0%, #5f3dc4 100%);
    }
    
    .subtype-n {
        background: linear-gradient(135deg, #764ba2 0%, #9c36b5 100%);
    }
    
    .subtype-p {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .subtype-y {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Dataframe Enhancement */
    .dataframe {
        border: none !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def call_api(endpoint: str, data: Optional[Dict] = None, method: str = "GET") -> Optional[Dict]:
    """Call the FastAPI backend"""
    try:
        url = f"{API_URL}{endpoint}"
        
        if method == "POST":
            response = requests.post(url, json=data, timeout=120)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error(f"⚠️ Request timed out. Agentic analysis may take up to 2 minutes.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Cannot connect to API at {API_URL}")
        st.info("Make sure the API server is running:\n```bash\npython3 api_oncodetect_unified.py\n```")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def create_enhanced_risk_gauge(risk_score: float, risk_level: str) -> go.Figure:
    """Create an enhanced gauge chart for risk score"""
    
    # Determine colors based on risk level
    if "VERY HIGH" in risk_level:
        gauge_color = "#c92a2a"
        bg_color = "rgba(255, 107, 107, 0.1)"
    elif "HIGH" in risk_level:
        gauge_color = "#f03e3e"
        bg_color = "rgba(255, 135, 135, 0.1)"
    elif "MODERATE" in risk_level:
        gauge_color = "#fab005"
        bg_color = "rgba(255, 212, 59, 0.1)"
    else:
        gauge_color = "#37b24d"
        bg_color = "rgba(81, 207, 102, 0.1)"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score", 'font': {'size': 28, 'color': '#495057', 'family': 'Inter'}},
        delta = {'reference': 50, 'increasing': {'color': "#c92a2a"}, 'decreasing': {'color': "#37b24d"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#dee2e6"},
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': bg_color,
            'borderwidth': 3,
            'bordercolor': gauge_color,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(81, 207, 102, 0.05)'},
                {'range': [25, 40], 'color': 'rgba(255, 212, 59, 0.05)'},
                {'range': [40, 60], 'color': 'rgba(255, 135, 135, 0.05)'},
                {'range': [60, 100], 'color': 'rgba(255, 107, 107, 0.05)'}
            ],
            'threshold': {
                'line': {'color': gauge_color, 'width': 6},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'size': 14}
    )
    return fig


def create_biomarker_radar(biomarkers: dict) -> go.Figure:
    """Create radar chart for biomarker profile"""
    
    categories = ['ASCL1', 'NEUROD1', 'POU2F3', 'YAP1', 'TP53', 'RB1', 'MYC']
    values = [
        biomarkers.get('ascl1', 0),
        biomarkers.get('neurod1', 0),
        biomarkers.get('pou2f3', 0),
        biomarkers.get('yap1', 0),
        biomarkers.get('tp53', 0),
        biomarkers.get('rb1', 0),
        biomarkers.get('myc', 0)
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        marker=dict(size=8, color='#667eea')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 15],
                gridcolor='#e9ecef',
                gridwidth=1
            ),
            angularaxis=dict(
                gridcolor='#e9ecef',
                gridwidth=1
            )
        ),
        showlegend=False,
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'size': 12}
    )
    
    return fig


def display_clinical_report(report: str):
    """Display enhanced clinical report"""
    st.markdown("### 📋 Clinical Report")
    st.markdown(f'<div class="clinical-report">{report}</div>', unsafe_allow_html=True)


def display_analysis_messages(messages: list):
    """Display analysis steps with enhanced UI"""
    if messages:
        st.markdown("### 🧠 Analysis Workflow")
        
        # Filter unique messages
        shown_types = set()
        key_messages = []
        
        for msg in messages:
            msg_lower = msg.lower()
            
            if "ml model predicted" in msg_lower:
                if "ml_prediction" not in shown_types:
                    key_messages.append(("🎯 ML Prediction", msg, "#667eea"))
                    shown_types.add("ml_prediction")
            
            elif msg.startswith("Analysis:") and "analysis" not in shown_types:
                key_messages.append(("🤖 LLM Analysis", msg, "#764ba2"))
                shown_types.add("analysis")
            
            elif "treatment protocol" in msg_lower and "treatment" not in shown_types:
                key_messages.append(("💊 Treatment Protocol", msg, "#f093fb"))
                shown_types.add("treatment")
        
        # Display as cards
        for i, (title, msg, color) in enumerate(key_messages, 1):
            with st.expander(f"Step {i}: {title}", expanded=False):
                st.markdown(f'<div class="analysis-section" style="border-left-color: {color};">{msg}</div>', 
                          unsafe_allow_html=True)


# ============================================
# SIDEBAR
# ============================================

# Enhanced Sidebar Header
st.sidebar.markdown("""
<div class="sidebar-header">
    <div style="font-size: 3rem; margin-bottom: 0.5rem;">🧬</div>
    <h2 style="color: #667eea; margin: 0;">OncoDetect-AI</h2>
    <p style="color: #6c757d; font-size: 0.9rem; margin-top: 0.5rem;">Precision Oncology Platform</p>
</div>
""", unsafe_allow_html=True)

# Store API health check but don't display it
api_health = call_api("/health")

st.sidebar.markdown("---")

# Navigation with better styling
st.sidebar.markdown("### 📍 Navigation")
page = st.sidebar.radio(
    "",
    ["🎯 Risk Prediction", "🧬 Subtype Classification", "📊 Batch Analysis"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Performance Metrics
st.sidebar.markdown("### 📈 Performance")
st.sidebar.info("""
**Risk Model:** 95.6% Accuracy  
**Subtype Model:** 89% Accuracy  
**Data Points:** 4.8M+  
**Avg Response:** <30s
""")

st.sidebar.markdown("---")
st.sidebar.caption(f"v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ============================================
# PAGE 1: RISK PREDICTION
# ============================================

if page == "🎯 Risk Prediction":
    # Page Header
    st.markdown("""
    <div class="page-header">
        <h1>🎯 SCLC Risk Prediction</h1>
        <p>Advanced survival risk assessment using clinical features and AI analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Patient Information")
        
        # Enhanced preset selector
        with st.expander("🔍 Load Example Patient", expanded=False):
            preset = st.selectbox(
                "Select Test Profile",
                ["Custom Input", 
                 "78-year-old Male (Very High Risk)", 
                 "65-year-old Female (Moderate Risk)", 
                 "55-year-old Male (Low Risk)"],
                help="Pre-configured patient profiles for testing"
            )
        
        # Set defaults based on preset
        if preset == "78-year-old Male (Very High Risk)":
            age, is_male_idx, smoker_idx, mutations, tmb = 78, 0, 0, 580, 21.5
        elif preset == "65-year-old Female (Moderate Risk)":
            age, is_male_idx, smoker_idx, mutations, tmb = 65, 1, 1, 350, 12.5
        elif preset == "55-year-old Male (Low Risk)":
            age, is_male_idx, smoker_idx, mutations, tmb = 55, 0, 0, 150, 5.5
        else:
            age, is_male_idx, smoker_idx, mutations, tmb = 65, 0, 0, 250, 10.0
        
        # Organized input sections
        st.markdown("#### Demographics")
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("Age (years)", 18, 100, int(age), help="Patient age at diagnosis")
        with col_b:
            is_male = st.selectbox("Sex", ["Male", "Female"], index=is_male_idx)
        
        st.markdown("#### Clinical History")
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"], index=smoker_idx,
                              help="Smoking history impacts risk assessment")
        
        st.markdown("#### Genomic Features")
        mutation_count = st.number_input("Mutation Count", 0, 2000, int(mutations),
                                        help="Total number of mutations detected")
        tmb = st.number_input("TMB (mut/Mb)", 0.0, 100.0, float(tmb), 0.1,
                             help="Tumor Mutation Burden per megabase")
        
        # TMB Category Display
        if tmb >= 20:
            st.error("🔴 **High TMB** (≥20 mut/Mb) - Increased risk")
        elif tmb >= 10:
            st.warning("🟡 **Intermediate TMB** (10-20 mut/Mb) - Moderate risk")
        else:
            st.success("🟢 **Low TMB** (<10 mut/Mb) - Lower risk")
        
        # Calculate features
        is_tmb_high = 1 if tmb >= 20 else 0
        is_tmb_intermediate = 1 if 10 <= tmb < 20 else 0
        is_tmb_low = 1 if tmb < 10 else 0
        is_male_val = 1 if is_male == "Male" else 0
        is_former_smoker = 1 if smoking == "Former" else 0
        smoker_x_tmb = tmb if smoking == "Former" else 0.0
        
        patient_data = {
            "age": float(age),
            "mutation_count": int(mutation_count),
            "tmb": float(tmb),
            "is_male": is_male_val,
            "is_former_smoker": is_former_smoker,
            "is_tmb_high": is_tmb_high,
            "is_tmb_intermediate": is_tmb_intermediate,
            "is_tmb_low": is_tmb_low,
            "smoker_x_tmb": float(smoker_x_tmb)
        }
        
        st.markdown("---")
        st.markdown("### 🔬 Analysis Type")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            quick_predict = st.button("⚡ Quick Prediction", 
                                     use_container_width=True,
                                     help="Fast ML prediction (~100ms)")
        
        with col_btn2:
            full_analysis = st.button("🤖 Full AI Analysis", 
                                     use_container_width=True,
                                     help="Comprehensive analysis with LLM reasoning (~30-60s)")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        # Quick Prediction
        if quick_predict:
            with st.spinner("Running ML prediction..."):
                result = call_api("/risk/predict", data=patient_data, method="POST")
                
                if result:
                    risk_score = result['risk_score']
                    risk_level = result['risk_level']
                    confidence = result['confidence']
                    
                    # Display enhanced risk card
                    if "VERY HIGH" in risk_level:
                        st.markdown(f'<div class="risk-card risk-very-high">⚠️ {risk_level} RISK</div>', 
                                  unsafe_allow_html=True)
                    elif "HIGH" in risk_level:
                        st.markdown(f'<div class="risk-card risk-high">⚠️ {risk_level} RISK</div>', 
                                  unsafe_allow_html=True)
                    elif "MODERATE" in risk_level:
                        st.markdown(f'<div class="risk-card risk-moderate">⚡ {risk_level} RISK</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-card risk-low">✅ {risk_level} RISK</div>', 
                                  unsafe_allow_html=True)
                    
                    # Enhanced gauge
                    st.plotly_chart(create_enhanced_risk_gauge(risk_score, risk_level), use_container_width=True)
                    
                    # Metrics
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Model Confidence", f"{confidence:.1%}")
                    with col_m2:
                        st.metric("Risk Score", f"{risk_score:.1f}/100")
                    
                    # Treatment recommendations
                    st.markdown("### 💊 Treatment Recommendation")
                    if "VERY HIGH" in risk_level or "HIGH" in risk_level:
                        st.error("""
                        **Immediate Action Required:**
                        - Aggressive platinum doublet + immunotherapy
                        - Consider clinical trial enrollment
                        - Weekly monitoring recommended
                        """)
                    elif "MODERATE" in risk_level:
                        st.warning("""
                        **Standard Intensive Protocol:**
                        - Platinum + etoposide chemotherapy
                        - Monthly monitoring
                        - Consider prophylactic cranial irradiation
                        """)
                    else:
                        st.success("""
                        **Conservative Management:**
                        - Standard chemotherapy regimen
                        - Quarterly follow-up
                        - Focus on quality of life
                        """)
        
        # Full Agentic Analysis
        if full_analysis:
            if api_health and not api_health.get("risk_agent_ready"):
                st.error("⚠️ **Full AI Analysis Not Available**")
                st.warning("""
                The agentic agent is not initialized. Please check:
                
                1. **LLM API Credentials** are set
                2. **Snowflake Connection** is configured
                3. **Restart API** after setting environment variables
                
                You can still use **Quick Prediction** for ML-only analysis.
                """)
            else:
                progress_bar = st.progress(0)
                with st.spinner("🤖 Running comprehensive AI analysis... (30-60 seconds)"):
                    progress_bar.progress(25)
                    
                    request_data = {
                        "patient_data": patient_data,
                        "include_similar_patients": True,
                        "include_treatment_protocol": True
                    }
                    
                    result = call_api("/risk/analyze", data=request_data, method="POST")
                    progress_bar.progress(100)
                    
                    if result:
                        ml_pred = result['ml_prediction']
                        risk_score = ml_pred['risk_score']
                        risk_level = ml_pred['risk_level']
                        
                        # Display enhanced risk card
                        if "VERY HIGH" in risk_level:
                            st.markdown(f'<div class="risk-card risk-very-high">⚠️ {risk_level} RISK</div>', 
                                      unsafe_allow_html=True)
                        elif "HIGH" in risk_level:
                            st.markdown(f'<div class="risk-card risk-high">⚠️ {risk_level} RISK</div>', 
                                      unsafe_allow_html=True)
                        elif "MODERATE" in risk_level:
                            st.markdown(f'<div class="risk-card risk-moderate">⚡ {risk_level} RISK</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="risk-card risk-low">✅ {risk_level} RISK</div>', 
                                      unsafe_allow_html=True)
                        
                        st.plotly_chart(create_enhanced_risk_gauge(risk_score, risk_level), use_container_width=True)
                        
                        # Metrics row
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Confidence", f"{ml_pred['confidence']:.1%}")
                        with col_m2:
                            st.metric("Risk Score", f"{risk_score:.1f}")
                        with col_m3:
                            st.metric("Analysis Time", f"{result['processing_time_seconds']:.1f}s")
                        
                        st.markdown("---")
                        
                        # Similar Patients
                        if result.get('similar_patients'):
                            st.markdown("### 👥 Similar Patient Analysis")
                            st.info(result['similar_patients'])
                        
                        # Analysis Steps
                        if result.get('messages'):
                            display_analysis_messages(result['messages'])
                        
                        st.markdown("---")
                        
                        # Clinical Report
                        if result.get('clinical_report'):
                            display_clinical_report(result['clinical_report'])
                        
                        st.download_button(
                            "📥 Download Clinical Report",
                            result['clinical_report'],
                            f"risk_report_{result['analysis_id']}.txt",
                            "text/plain",
                            use_container_width=True
                        )


# ============================================
# PAGE 2: SUBTYPE CLASSIFICATION
# ============================================

elif page == "🧬 Subtype Classification":
    # Page Header
    st.markdown("""
    <div class="page-header">
        <h1>🧬 SCLC Molecular Subtype Classification</h1>
        <p>Classify SCLC into molecular subtypes (A/N/P/Y) based on biomarker expression</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🧬 Biomarker Expression Panel")
        
        # Preset selector
        with st.expander("🔍 Load Example Subtype", expanded=False):
            preset = st.selectbox(
                "Select Biomarker Profile",
                ["Custom", "SCLC-A", "SCLC-N", "SCLC-P", "SCLC-Y"],
                help="Load characteristic profiles for each subtype"
            )
        
        defaults = {
            "SCLC-A": [12.5, 5.0, 4.0, 5.5, 11.0, 4.5, 8.0, 9.5, 6.0],
            "SCLC-N": [5.0, 12.0, 4.5, 5.0, 10.5, 4.0, 7.5, 8.0, 6.5],
            "SCLC-P": [4.5, 5.0, 13.0, 5.5, 10.0, 4.5, 7.0, 7.5, 6.0],
            "SCLC-Y": [4.0, 4.5, 5.0, 12.5, 11.0, 9.0, 10.0, 6.5, 5.5],
            "Custom": [8.0, 7.0, 6.5, 7.5, 10.0, 6.0, 8.5, 8.0, 6.5]
        }.get(preset, [8.0]*9)
        
        st.markdown("#### Transcription Factors")
        col_tf1, col_tf2 = st.columns(2)
        with col_tf1:
            ascl1 = st.slider("ASCL1", 0.0, 15.0, defaults[0], 0.1,
                             help="Achaete-scute homolog 1")
            neurod1 = st.slider("NEUROD1", 0.0, 15.0, defaults[1], 0.1,
                               help="Neurogenic differentiation 1")
        with col_tf2:
            pou2f3 = st.slider("POU2F3", 0.0, 15.0, defaults[2], 0.1,
                              help="POU class 2 homeobox 3")
            yap1 = st.slider("YAP1", 0.0, 15.0, defaults[3], 0.1,
                            help="Yes-associated protein 1")
        
        st.markdown("#### Tumor Suppressors")
        col_ts1, col_ts2 = st.columns(2)
        with col_ts1:
            tp53 = st.slider("TP53", 0.0, 15.0, defaults[4], 0.1)
        with col_ts2:
            rb1 = st.slider("RB1", 0.0, 15.0, defaults[5], 0.1)
        
        st.markdown("#### MYC Family")
        col_myc1, col_myc2, col_myc3 = st.columns(3)
        with col_myc1:
            myc = st.slider("MYC", 0.0, 15.0, defaults[6], 0.1)
        with col_myc2:
            mycl = st.slider("MYCL", 0.0, 15.0, defaults[7], 0.1)
        with col_myc3:
            mycn = st.slider("MYCN", 0.0, 15.0, defaults[8], 0.1)
        
        # Additional markers (hidden)
        dll3, bcl2, notch1 = 7.0, 8.0, 6.5
        
        # Calculate features
        all_values = [ascl1, neurod1, pou2f3, yap1, tp53, rb1, myc, mycl, mycn]
        myc_family_score = myc + mycl + mycn
        tp53_rb1_dual_loss = 1 if (tp53 < 6 and rb1 < 6) else 0
        mean_expression = np.mean(all_values)
        stddev_expression = np.std(all_values)
        genes_expressed = sum(1 for v in all_values if v > 5)
        
        biomarker_data = {
            "ascl1": ascl1, "neurod1": neurod1, "pou2f3": pou2f3, "yap1": yap1,
            "tp53": tp53, "rb1": rb1, "myc": myc, "mycl": mycl, "mycn": mycn,
            "dll3": dll3, "bcl2": bcl2, "notch1": notch1,
            "myc_family_score": myc_family_score,
            "tp53_rb1_dual_loss": tp53_rb1_dual_loss,
            "mean_expression": mean_expression,
            "stddev_expression": stddev_expression,
            "genes_expressed": genes_expressed
        }
        
        st.markdown("---")
        st.markdown("### 🔬 Analysis Type")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            predict_btn = st.button("⚡ Classify Subtype", 
                                   use_container_width=True,
                                   help="Fast ML classification (~100ms)")
        
        with col_btn2:
            investigate_btn = st.button("🔍 Analyze Subtype", 
                                      use_container_width=True,
                                      help="Full biomarker analysis with differential diagnosis (~30-60s)")
    
    with col2:
        st.markdown("### 📊 Classification Results")
        
        # Show biomarker radar
        fig_radar = create_biomarker_radar(biomarker_data)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Quick Classification
        if predict_btn:
            with st.spinner("Classifying subtype..."):
                result = call_api("/subtype/predict", data=biomarker_data, method="POST")
                
                if result:
                    subtype = result['predicted_subtype']
                    confidence = result['confidence']
                    probs = result['probabilities']
                    
                    # Display subtype card
                    subtype_class = {
                        "SCLC-A": "subtype-a",
                        "SCLC-N": "subtype-n",
                        "SCLC-P": "subtype-p",
                        "SCLC-Y": "subtype-y"
                    }.get(subtype, "subtype-a")
                    
                    st.markdown(f"""
                    <div class="subtype-card {subtype_class}">
                        <h2 style="margin: 0;">Molecular Subtype: {subtype}</h2>
                        <p style="margin: 0.5rem 0 0 0; opacity: 0.95;">Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability chart
                    st.markdown("#### Subtype Probabilities")
                    prob_df = pd.DataFrame({
                        'Subtype': list(probs.keys()),
                        'Probability': [v * 100 for v in probs.values()]
                    })
                    
                    fig = px.bar(prob_df, x='Subtype', y='Probability', 
                                color='Probability',
                                color_continuous_scale='Viridis')
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Treatment info
                    st.markdown("### 💊 Therapeutic Implications")
                    
                    treatments = {
                        "SCLC-A": ("Classic NE", "ASCL1-high", "Standard platinum chemo, DLL3-targeted therapy"),
                        "SCLC-N": ("Alternative NE", "NEUROD1-high", "Platinum chemo, Aurora kinase inhibitors"),
                        "SCLC-P": ("Tuft-like", "POU2F3-high", "PARP inhibitors, immunotherapy combinations"),
                        "SCLC-Y": ("Non-NE", "YAP1-high", "Immunotherapy combinations, targeted therapies")
                    }
                    
                    if subtype in treatments:
                        info = treatments[subtype]
                        st.info(f"**{info[0]}** ({info[1]})\n\n**Treatment:** {info[2]}")
    
    # Full Investigation (full width)
    if investigate_btn:
        st.markdown("---")
        
        if api_health and not api_health.get("risk_agent_ready"):
            st.warning("⚠️ Agentic analysis requires LLM credentials. Using quick classification instead.")
        else:
            progress_bar = st.progress(0)
            with st.spinner("🔍 Investigating biomarker pattern... (30-60 seconds)"):
                progress_bar.progress(25)
                
                result = call_api("/subtype/analyze", 
                                data={"biomarker_data": biomarker_data}, 
                                method="POST")
                progress_bar.progress(100)
                
                if result:
                    ml_pred = result['ml_prediction']
                    subtype = ml_pred['predicted_subtype']
                    confidence = ml_pred['confidence']
                    
                    # Results header
                    col_r1, col_r2, col_r3 = st.columns([2, 1, 1])
                    
                    with col_r1:
                        subtype_class = {
                            "SCLC-A": "subtype-a",
                            "SCLC-N": "subtype-n",
                            "SCLC-P": "subtype-p",
                            "SCLC-Y": "subtype-y"
                        }.get(subtype, "subtype-a")
                        
                        st.markdown(f"""
                        <div class="subtype-card {subtype_class}">
                            <h3 style="margin: 0;">Diagnosed: {subtype}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_r2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col_r3:
                        st.metric("Analysis Time", f"{result['processing_time_seconds']:.1f}s")
                    
                    # Biomarker Analysis
                    if result.get('biomarker_analysis'):
                        st.markdown("### 🔬 Biomarker Analysis")
                        
                        bio_analysis = result['biomarker_analysis']
                        
                        col_b1, col_b2 = st.columns(2)
                        
                        with col_b1:
                            st.markdown(f"""
                            <div class="info-card">
                                <h4>Dominant Transcription Factor</h4>
                                <p style="font-size: 1.5rem; font-weight: bold; color: #667eea;">
                                    {bio_analysis.get('dominant_tf', 'Unknown')} ({bio_analysis.get('dominant_value', 0):.2f})
                                </p>
                                <p>Dominance Ratio: {bio_analysis.get('dominance_ratio', 0):.2f}x</p>
                                <p>Classification: {bio_analysis.get('classification', 'Unknown')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b2:
                            col_ne1, col_ne2 = st.columns(2)
                            with col_ne1:
                                st.metric("NE Score", f"{bio_analysis.get('ne_score', 0):.2f}")
                            with col_ne2:
                                st.metric("Non-NE Score", f"{bio_analysis.get('non_ne_score', 0):.2f}")
                        
                        # Similar profiles
                        if bio_analysis.get('similar_profiles'):
                            st.markdown("#### 👥 Database Validation")
                            st.success(bio_analysis['similar_profiles'])
                    
                    # Hypothesis Testing
                    if result.get('hypothesis_testing'):
                        st.markdown("### 🧪 Differential Diagnosis")
                        
                        compat = result['hypothesis_testing']
                        compat_df = pd.DataFrame({
                            'Subtype': list(compat.keys()),
                            'Compatibility': list(compat.values())
                        })
                        
                        fig = px.bar(compat_df, x='Subtype', y='Compatibility',
                                    color='Compatibility',
                                    color_continuous_scale='RdYlGn',
                                    title="Subtype Compatibility Analysis")
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Investigation Steps
                    if result.get('messages'):
                        with st.expander("🧠 View Detailed Analysis", expanded=False):
                            for msg in result['messages']:
                                if isinstance(msg, dict) and 'content' in msg:
                                    msg = msg['content']
                                
                                # Only process differential diagnosis messages
                                if "Differential Diagnosis" in msg:
                                    # Apply CSS to make all text smaller
                                    st.markdown("""
                                    <style>
                                    .stMarkdown {
                                        font-size: 14px !important;
                                    }
                                    .stMarkdown h1 {
                                        font-size: 18px !important;
                                    }
                                    .stMarkdown h2 {
                                        font-size: 16px !important;
                                    }
                                    .stMarkdown h3 {
                                        font-size: 15px !important;
                                    }
                                    .stMarkdown p {
                                        font-size: 14px !important;
                                        line-height: 1.5;
                                    }
                                    </style>
                                    """, unsafe_allow_html=True)
                                    
                                    # Just display the message as-is, Streamlit will format it
                                    st.markdown(msg)
                                    
                                    # Extract and display final answer if present
                                    if "The final answer is:" in msg:
                                        import re
                                        final_match = re.search(r'The final answer is:\s*(.+?)(?:\n|$)', msg)
                                        if final_match:
                                            answer = final_match.group(1).strip()
                                            st.success(f"**Final Diagnosis: {answer}**")
                                    
                                    break
                    
                    # Clinical Report
                    if result.get('clinical_report'):
                        display_clinical_report(result['clinical_report'])
                    
                    st.download_button(
                        "📥 Download Subtype Analysis Report",
                        result['clinical_report'],
                        f"subtype_report_{result['analysis_id']}.txt",
                        "text/plain",
                        use_container_width=True
                    )


# ============================================
# PAGE 3: BATCH ANALYSIS
# ============================================

elif page == "📊 Batch Analysis":
    # Page Header
    st.markdown("""
    <div class="page-header">
        <h1>📊 Batch Analysis</h1>
        <p>Process multiple patients simultaneously for risk prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload CSV file with patient data", type=['csv'])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea;">{}</h3>
                <p>Total Patients</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #764ba2;">{}</h3>
                <p>Features</p>
            </div>
            """.format(len(df.columns)), unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #f093fb;">{}</h3>
                <p>Missing Values</p>
            </div>
            """.format(df.isnull().sum().sum()), unsafe_allow_html=True)
        
        if st.button("🚀 Run Batch Analysis", use_container_width=True):
            progress_bar = st.progress(0)
            with st.spinner(f"Processing {len(df)} patients..."):
                progress_bar.progress(30)
                
                patients = df.to_dict('records')
                result = call_api("/risk/batch", data=patients, method="POST")
                progress_bar.progress(100)
                
                if result:
                    preds = result['predictions']
                    results_df = pd.DataFrame(preds)
                    
                    st.success(f"✅ Successfully processed {len(preds)} patients")
                    
                    # Results summary
                    st.markdown("### 📊 Results Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        high = len([p for p in preds if "HIGH" in p['risk_level']])
                        st.metric("High Risk", high, f"{high/len(preds)*100:.0f}%")
                    with col2:
                        moderate = len([p for p in preds if "MODERATE" in p['risk_level']])
                        st.metric("Moderate Risk", moderate, f"{moderate/len(preds)*100:.0f}%")
                    with col3:
                        low = len([p for p in preds if "LOW" in p['risk_level']])
                        st.metric("Low Risk", low, f"{low/len(preds)*100:.0f}%")
                    with col4:
                        avg = sum([p['risk_score'] for p in preds]) / len(preds)
                        st.metric("Avg Risk Score", f"{avg:.1f}")
                    
                    # Distribution chart
                    st.markdown("### 📈 Risk Distribution")
                    fig = px.histogram(results_df, x='risk_score', nbins=20,
                                     color_discrete_sequence=['#667eea'],
                                     title="Risk Score Distribution")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.markdown("### 📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv,
                        f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )


# ============================================
# PAGE 4: MODEL INFO
# ============================================

# elif page == "ℹ️ Model Info":
#     # Page Header
#     st.markdown("""
#     <div class="page-header">
#         <h1>ℹ️ Model Information</h1>
#         <p>Technical details and performance metrics for OncoDetect-AI models</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     tab1, tab2, tab3 = st.tabs(["🎯 Risk Model", "🧬 Subtype Model", "🤖 System Architecture"])
    
#     with tab1:
#         st.markdown("### 🎯 Risk Prediction Model")
        
#         info = call_api("/risk/info")
#         if info:
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("""
#                 <div class="info-card">
#                     <h3>📊 Model Details</h3>
#                 """, unsafe_allow_html=True)
                
#                 st.info(f"""
#                 **Version:** {info.get('version', 'Unknown')}  
#                 **Type:** {info.get('model_type', 'Unknown')}  
#                 **Features:** {info.get('feature_count', 'Unknown')}  
#                 **Trained:** {info.get('trained_at', 'Unknown')}
                
#                 **Performance Metrics:**
#                 - Accuracy: 95.6%
#                 - ROC-AUC: 99.1%
#                 - Sensitivity: 94.2%
#                 - Specificity: 96.8%
#                 """)
                
#                 st.markdown("</div>", unsafe_allow_html=True)
            
#             with col2:
#                 st.markdown("### 🔝 Feature Importance")
#                 if info.get('top_features'):
#                     df = pd.DataFrame(info['top_features'])
#                     fig = px.bar(df, x='importance', y='feature', 
#                                orientation='h',
#                                color='importance',
#                                color_continuous_scale='Viridis')
#                     fig.update_layout(showlegend=False, height=400)
#                     st.plotly_chart(fig, use_container_width=True)
    
#     with tab2:
#         st.markdown("### 🧬 Subtype Classification Model")
        
#         info = call_api("/subtype/info")
#         if info:
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.markdown("""
#                 <div class="info-card">
#                     <h3>📊 Model Details</h3>
#                 """, unsafe_allow_html=True)
                
#                 st.info(f"""
#                 **Version:** {info.get('version', 'Unknown')}  
#                 **Classes:** {', '.join(info.get('classes', []))}  
#                 **Features:** 17 biomarkers
                
#                 **Performance Metrics:**
#                 - Overall Accuracy: 89%
#                 - SCLC-A Precision: 91%
#                 - SCLC-N Precision: 85%
#                 - SCLC-P Precision: 88%
#                 - SCLC-Y Precision: 92%
#                 """)
                
#                 st.markdown("</div>", unsafe_allow_html=True)
            
#             with col2:
#                 # Subtype distribution pie chart
#                 fig = px.pie(values=[68, 11, 7, 14],
#                            names=['SCLC-A', 'SCLC-N', 'SCLC-P', 'SCLC-Y'],
#                            title="SCLC Subtype Distribution in Training Data",
#                            color_discrete_map={
#                                'SCLC-A': '#667eea',
#                                'SCLC-N': '#764ba2',
#                                'SCLC-P': '#f093fb',
#                                'SCLC-Y': '#4facfe'
#                            })
#                 st.plotly_chart(fig, use_container_width=True)
    
#     with tab3:
#         st.markdown("""
#         ### 🤖 Agentic System Architecture
        
#         **Framework:** LangGraph multi-agent workflow
        
#         #### Risk Score Agent (3 Nodes)
#         1. **ML Prediction** - Random Forest model prediction
#         2. **Evidence Gathering** - Snowflake similar patient search + LLM analysis
#         3. **Clinical Synthesis** - Report generation with treatment recommendations
        
#         #### Subtype Classifier Agent (4 Nodes)
#         1. **ML Prediction** - Multi-class classification
#         2. **Biomarker Investigation** - Transcription factor analysis + database validation
#         3. **Hypothesis Testing** - Differential diagnosis for all 4 subtypes
#         4. **Clinical Report** - Comprehensive report with citations
        
#         **Technologies:**
#         - **LLM:** OpenAI-compatible (Jetstream/Ollama)
#         - **Database:** Snowflake (4.8M+ data points)
#         - **Backend:** FastAPI
#         - **Pipeline:** DBT (8 models)
        
#         **Performance:**
#         - Quick Prediction: ~100ms
#         - Full AI Analysis: 30-60s
#         - Batch Processing: ~50ms per patient
#         """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 2rem;'>
    <p><strong>OncoDetect-AI</strong> | Advanced Precision Oncology Platform</p>
    <p>For Research Use Only | Not for Clinical Diagnosis</p>
    <p style="font-size: 0.85rem;">© 2024 | Powered by LangGraph • Snowflake • FastAPI</p>
</div>
""", unsafe_allow_html=True)
