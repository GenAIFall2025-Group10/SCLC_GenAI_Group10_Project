# ============================================================
# ONCODETECT AI - INTEGRATED MULTI-AGENT SCLC PLATFORM
# ============================================================
# Advanced precision oncology platform combining:
# - RAG (Retrieval-Augmented Generation) Agent
# - Arxiv Research Agent  
# - Web Search Agent
# - Risk Prediction System
# - Subtype Classification System
# ============================================================

import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, call_builtin, lit
import hashlib
import re
from datetime import datetime, date
import json
import time
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
import uuid
import base64
import altair as alt

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OncoDetect AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CUSTOM CSS STYLING
# ============================================================

st.markdown("""
    <style>
    /* Apply font to most elements BUT exclude Material Icons */
    body, h1, h2, h3, h4, h5, h6, p, div, span, label, input, textarea, select,
    .stMarkdown, .stText, .stTextInput, .stSelectbox, .stDateInput, .stButton,
    .stForm, .stAlert, .stCaption {
        font-family: 'Amasis MT Pro', serif !important;
    }
    
    /* Preserve Material Icons font for Streamlit's icon buttons AND EXPANDERS */
    .material-symbols-rounded,
    [data-testid="collapsedControl"] span,
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] button span,
    button[kind="headerNoPadding"] span,
    .st-emotion-cache-1dp5vir span,
    span[class*="material"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
    }
    
    /* Main button styling - #081f37 */
    .stButton > button {
        background-color: #081f37 !important;
        color: #FFFFFF !important;
        border-radius: 5px;
        font-size: 14px !important;
        height: 40px !important;
        border: none !important;
    }
    
    .stButton > button:hover { 
        background-color: #0a2847 !important; 
    }
    
    h1 { font-size: 20px !important; }
    h2 { font-size: 18px !important; }
    h3 { font-size: 16px !important; }
    p, div, span, label, input, textarea, select { font-size: 14px !important; }
    .stMarkdown, .stText { font-size: 14px !important; }
    .stTextInput label, .stSelectbox label, .stDateInput label { font-size: 14px !important; }
    .stTextInput input, .stSelectbox select, .stDateInput input { font-size: 14px !important; }
    .stCaption { font-size: 13px !important; }
    .stApp { background-color: #FFFFFF; }
    
    /* Form submit button - #081f37 */
    .stFormSubmitButton > button {
        background-color: #081f37 !important;
        color: #FFFFFF !important;
        font-size: 14px !important;
        border: none !important;
    }
    
    .stFormSubmitButton > button:hover {
        background-color: #0a2847 !important;
    }
    
    .stForm {
        background-color: #F9F9F9;
        border: 1px solid #E0E0E0;
        padding: 2rem;
        border-radius: 10px;
    }
    
    .user-message {
        background-color: #E8E8E8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-left: auto;
        margin-right: 0;
        max-width: 80%;
        border-left: 4px solid #808080;
        font-size: 14px !important;
    }
    
    .bot-message {
        background-color: #F8F8F8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        font-size: 14px !important;
    }
    
    .translated-text, .summary-text {
        background-color: #F8F8F8;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        font-size: 14px !important;
    }
    .translated-text { border-left: 4px solid #FFD700; }
    .summary-text { border-left: 4px solid #4CAF50; }
    
    section[data-testid="stSidebar"] { background-color: #FAFAFA !important; }
    button[kind="header"] { display: none !important; }
    
    /* Sidebar buttons - #081f37 */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #081f37 !important;
        color: #FFFFFF !important;
        text-align: left !important;
        font-size: 14px !important;
        padding: 0.6rem 0.8rem !important;
        min-height: 40px !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        width: 100% !important;
        border: none !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #0a2847 !important;
    }
    
    .viewing-note-highlight {
        background-color: #1a1a1a !important;
        padding: 0.8rem !important;
        border-radius: 8px !important;
        margin-bottom: 0.8rem !important;
        border-left: 4px solid #0066CC !important;
        color: #FFFFFF !important;
        font-size: 14px !important;
    }
    
    .typing-cursor {
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .source-badge {
        background-color: #FFF9E6;
        border-left: 4px solid #FFA500;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 13px !important;
        color: #333333;
        font-weight: 500;
    }
    
    .image-source-badge {
        background-color: #E6F3FF;
        border-left: 4px solid #2196F3;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 13px !important;
        color: #333333;
        font-weight: 500;
    }
    
    .profile-card {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        margin: 1rem 0;
    }
    
    .profile-card-item {
        margin: 0.8rem 0;
        font-size: 14px !important;
    }
    
    .profile-card-label {
        color: #666666;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    
    .profile-card-value {
        color: #000000;
        font-weight: 600;
        font-size: 15px !important;
    }
    
    .stAlert[data-baseweb="notification"] {
        background-color: #F0F8FF !important;
        border-left: 4px solid #4A90E2 !important;
    }
    
    /* Clinical Prediction Styles */
    .risk-card {
        padding: 1.5rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 1.25rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInScale 0.5s ease;
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
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.05);
        border-top: 3px solid #667eea;
    }
    
    /* Download button - #081f37 */
    .stDownloadButton > button {
        background-color: #081f37 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    .stDownloadButton > button:hover {
        background-color: #0a2847 !important;
    }
    /* Sidebar Profile Card Styles */
    .sidebar-profile-card {
        background-color: #f0f2f6;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #dce0e6;
    }
    .sidebar-profile-name {
        color: #081f37;
        font-weight: bold;
        font-size: 1.1rem !important;
        margin-bottom: 5px;
    }
    .sidebar-profile-info {
        color: #555;
        font-size: 0.85rem !important;
        margin-bottom: 2px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 3. SESSION STATE INITIALIZATION
# ============================================================

# Authentication state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Navigation state
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'show_research' not in st.session_state:
    st.session_state.show_research = False
if 'show_analytics' not in st.session_state:
    st.session_state.show_analytics = False
if 'show_risk_prediction' not in st.session_state:
    st.session_state.show_risk_prediction = False
if 'show_subtype_classification' not in st.session_state:
    st.session_state.show_subtype_classification = False

# Chat and research state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'research_notes' not in st.session_state:
    st.session_state.research_notes = []
if 'viewing_note' not in st.session_state:
    st.session_state.viewing_note = None
if 'show_sources' not in st.session_state:
    st.session_state.show_sources = {}

# Analytics and feedback state
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False
if 'analytics_history' not in st.session_state:
    st.session_state.analytics_history = []

# Clinical prediction result states
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None

# ============================================================
# 4. DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_snowflake_session():
    """Initialize and cache Snowflake session"""
    return Session.builder.getOrCreate()

session = get_snowflake_session()

# ============================================================
# KAFKA EVENT TRACKING SETUP
# ============================================================

# ============================================================
# KAFKA EVENT TRACKING SETUP (REST API VERSION)
# ============================================================
import requests
import base64
import json
import uuid

@st.cache_resource
def get_kafka_config():
    """Fetch Kafka REST config from table"""
    try:
        # Dictionary to store all needed creds
        config = {}
        
        # Keys to fetch
        keys = ['KAFKA_API_KEY', 'KAFKA_API_SECRET', 'KAFKA_REST_URL', 'KAFKA_CLUSTER_ID']
        
        # Fetch all in one go
        key_list = "', '".join(keys)
        query = f"SELECT CONFIG_KEY, CONFIG_VALUE FROM ONCODETECT_DB.USER_MGMT.SERP_CONFIG WHERE CONFIG_KEY IN ('{key_list}')"
        results = session.sql(query).collect()
        
        for row in results:
            config[row['CONFIG_KEY']] = str(row['CONFIG_VALUE']).strip()
            
        # Verify we have everything
        if all(k in config for k in keys):
            return config
        return None
    except Exception as e:
        print(f"Error fetching Kafka config: {str(e)}")
        return None

def send_kafka_event(topic_name, event_type, button_name, user_id, status):
    """Send event via Confluent REST API (v3)"""
    
    # 1. Get Config
    conf = get_kafka_config()
    if not conf:
        print("Kafka Configuration Missing")
        return False
        
    # 2. Construct URL
    # Endpoint: /kafka/v3/clusters/{cluster_id}/topics/{topic_name}/records
    url = f"{conf['KAFKA_REST_URL']}/kafka/v3/clusters/{conf['KAFKA_CLUSTER_ID']}/topics/{topic_name}/records"

    # 3. Create Event Data
    event_data = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_type": "user",
        "user_id": str(user_id) if user_id else "anonymous",
        "session_id": st.session_state.get('session_id', 'unknown'),
        "button_name": button_name,
        "status": status,
        "source": "snowflake_sis"
    }

    # 4. Construct Payload (Confluent v3 Format)
    payload = {
        "value": {
            "type": "JSON",
            "data": event_data
        }
    }

    # 5. Prepare Headers & Auth
    auth_str = f"{conf['KAFKA_API_KEY']}:{conf['KAFKA_API_SECRET']}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {b64_auth}"
    }

    # 6. Send Request
    try:
        # Short timeout (2s) to ensure UI doesn't freeze if Confluent is slow
        response = requests.post(url, headers=headers, json=payload, timeout=2.0)
        
        if response.status_code == 200:
            return True
        else:
            # Print error to console for debugging (visible in SiS logs)
            print(f"Kafka REST Error ({response.status_code}): {response.text}")
            return False
            
    except Exception as e:
        print(f"Kafka Connection Error: {str(e)}")
        return False

# ============================================================
# 5. AUTHENTICATION FUNCTIONS
# ============================================================

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    """Validate password meets security requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long⚠️"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least 1 uppercase letter⚠️"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least 1 number⚠️"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least 1 special character⚠️"
    return True, "Password is valid✅"

def validate_username(username):
    """Validate username format"""
    if len(username) < 3:
        return False, "Username must be at least 3 characters long⚠️"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores⚠️"
    return True, "Username is valid✅"

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return (True, "Email is valid✅") if re.match(pattern, email) else (False, "Invalid email format⚠️")

def username_exists(username):
    """Check if username already exists in database"""
    try:
        result = session.sql(f"SELECT COUNT(*) as count FROM ONCODETECT_DB.USER_MGMT.USERS WHERE USERNAME = '{username}'").collect()
        return result[0]['COUNT'] > 0
    except:
        return False

def email_exists(email):
    """Check if email already exists in database"""
    try:
        result = session.sql(f"SELECT COUNT(*) as count FROM ONCODETECT_DB.USER_MGMT.USERS WHERE EMAIL = '{email}'").collect()
        return result[0]['COUNT'] > 0
    except:
        return False

def signup_user(first_name, last_name, dob, email, username, password, role):
    """Create new user account"""
    try:
        password_hash = hash_password(password)
        query = f"""
        INSERT INTO ONCODETECT_DB.USER_MGMT.USERS 
        (FIRST_NAME, LAST_NAME, DATE_OF_BIRTH, EMAIL, USERNAME, PASSWORD_HASH, USER_ROLE)
        VALUES ('{first_name}', '{last_name}', '{dob}', '{email}', '{username}', '{password_hash}', '{role}')
        """
        session.sql(query).collect()
        return True, "Account created successfully! Please login✅"
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(username, password, role):
    """Authenticate user and return user data"""
    try:
        password_hash = hash_password(password)
        query = f"""
        SELECT USER_ID, FIRST_NAME, LAST_NAME, USERNAME, USER_ROLE
        FROM ONCODETECT_DB.USER_MGMT.USERS
        WHERE USERNAME = '{username}' 
        AND PASSWORD_HASH = '{password_hash}' 
        AND USER_ROLE = '{role}' 
        AND IS_ACTIVE = TRUE
        """
        result = session.sql(query).collect()
        
        if len(result) > 0:
            user_data = result[0]
            # Update last login timestamp
            session.sql(f"UPDATE ONCODETECT_DB.USER_MGMT.USERS SET LAST_LOGIN = CURRENT_TIMESTAMP() WHERE USER_ID = {user_data['USER_ID']}").collect()
            return True, user_data
        return False, None
    except:
        return False, None

def logout():
    """Clear all session state and logout user"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None
    st.session_state.user_role = None
    st.session_state.show_signup = False
    st.session_state.show_research = False
    st.session_state.show_analytics = False
    st.session_state.show_risk_prediction = False
    st.session_state.show_subtype_classification = False
    st.session_state.chat_history = []
    st.session_state.research_notes = []
    st.session_state.viewing_note = None
    st.session_state.show_sources = {}
    st.session_state.show_feedback = False
    st.session_state.analytics_history = []
    st.session_state.risk_result = None
    st.rerun()

# ============================================================
# 6. NAVIGATION FUNCTIONS
# ============================================================

def go_to_signup():
    """Navigate to signup page"""
    st.session_state.show_signup = True

def go_to_login():
    """Navigate to login page"""
    st.session_state.show_signup = False

def start_research():
    """Navigate to research page and load notes"""
    st.session_state.show_research = True
    st.session_state.show_analytics = False
    st.session_state.show_risk_prediction = False
    st.session_state.show_subtype_classification = False
    st.session_state.chat_history = []
    st.session_state.viewing_note = None
    st.session_state.show_sources = {}
    load_research_notes(st.session_state.user_id)

def start_analytics():
    """Navigate to analytics page and initialize metadata"""
    st.session_state.show_analytics = True
    st.session_state.show_research = False
    st.session_state.show_risk_prediction = False
    st.session_state.show_subtype_classification = False
    ensure_metadata_initialized()

def start_risk_prediction():
    """Navigate to risk prediction page"""
    st.session_state.show_risk_prediction = True
    st.session_state.show_research = False
    st.session_state.show_analytics = False
    st.session_state.show_subtype_classification = False
    st.session_state.risk_result = None

def start_subtype_classification():
    """Navigate to subtype classification page"""
    st.session_state.show_subtype_classification = True
    st.session_state.show_research = False
    st.session_state.show_analytics = False
    st.session_state.show_risk_prediction = False

def go_back_to_profile():
    """Navigate back to profile page"""
    st.session_state.show_research = False
    st.session_state.show_analytics = False
    st.session_state.show_risk_prediction = False
    st.session_state.show_subtype_classification = False
    st.session_state.chat_history = []
    st.session_state.viewing_note = None
    st.session_state.show_sources = {}
    st.session_state.analytics_history = []
    st.session_state.risk_result = None

# ============================================================
# 7. ANALYTICS FUNCTIONS
# ============================================================

# ============================================================
# 7. ANALYTICS FUNCTIONS (ROUTER & INSIGHTS)
# ============================================================

def ensure_metadata_initialized():
    """Initialize table metadata for ALL 3 analytics tables"""
    try:
        # Check if we have populated metadata for all 3 tables
        # We can just check count, if it's low, we truncate and re-seed
        check_query = "SELECT COUNT(*) as count FROM ONCODETECT_DB.PRODUCT_ANALYTICS.TABLE_METADATA"
        result = session.sql(check_query).collect()
        count = result[0]['COUNT']
        
        if count < 3: # If we don't have all 3 tables
            # Clear existing to avoid dupes
            session.sql("TRUNCATE TABLE ONCODETECT_DB.PRODUCT_ANALYTICS.TABLE_METADATA").collect()
            
            # 1. USER FEEDBACK TABLE
            q1 = """
            INSERT INTO ONCODETECT_DB.PRODUCT_ANALYTICS.TABLE_METADATA 
            (TABLE_NAME, TABLE_SCHEMA, FULL_TABLE_PATH, DESCRIPTION, COLUMNS_INFO, METADATA_EMBEDDING)
            SELECT 'USER_FEEDBACK', 'PRODUCT_ANALYTICS', 'ONCODETECT_DB.PRODUCT_ANALYTICS.USER_FEEDBACK',
            'Contains user feedback text, ratings (1-5 stars), and submission dates.',
            'FEEDBACK_ID, USER_ID, USERNAME, FEEDBACK_TEXT, RATING, FEEDBACK_DATE',
            SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', 'feedback rating sentiment comments reviews stars satisfaction')
            """
            
            # 2. USERS TABLE (User Acquisition)
            q2 = """
            INSERT INTO ONCODETECT_DB.PRODUCT_ANALYTICS.TABLE_METADATA 
            (TABLE_NAME, TABLE_SCHEMA, FULL_TABLE_PATH, DESCRIPTION, COLUMNS_INFO, METADATA_EMBEDDING)
            SELECT 'USERS', 'USER_MGMT', 'ONCODETECT_DB.USER_MGMT.USERS',
            'Contains registered user accounts, demographics, and roles. Use for counting total users or user lists.',
            'USER_ID, FIRST_NAME, LAST_NAME, EMAIL, USERNAME, USER_ROLE, IS_ACTIVE, LAST_LOGIN, DATE_OF_BIRTH',
            SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', 'how many users total users accounts registered demographics email list')
            """
            
            # 3. EVENTS VIEW (Feature Usage & Buttons)
            q3 = """
            INSERT INTO ONCODETECT_DB.PRODUCT_ANALYTICS.TABLE_METADATA 
            (TABLE_NAME, TABLE_SCHEMA, FULL_TABLE_PATH, DESCRIPTION, COLUMNS_INFO, METADATA_EMBEDDING)
            SELECT 'USER_ACQUISITION_EVENTS', 'PRODUCT_ANALYTICS', 'ONCODETECT_DB.PRODUCT_ANALYTICS.USER_ACQUISITION_EVENTS',
            'Contains clickstream data: button clicks, logins, signups, feature usage, errors. Use for behavioral analysis.',
            'EVENT_ID, USER_ID, EVENT_TYPE, BUTTON_NAME, STATUS, USER_TYPE, SOURCE, APP_TIMESTAMP, KAFKA_INGEST_TIMESTAMP',
            SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', 'button clicks features usage events login signup failure success trend interaction')
            """
            
            session.sql(q1).collect()
            session.sql(q2).collect()
            session.sql(q3).collect()
            return True
        return True
    except Exception as e:
        print(f"Metadata Init Error: {e}")
        return False

def find_relevant_tables(user_question):
    """
    DETERMINISTIC ROUTER:
    Routes the user question to the exact table based on keywords before trying vector search.
    """
    q = user_question.lower()
    
    # --- RULE 1: FEEDBACK & SENTIMENT ---
    if any(k in q for k in ['feedback', 'rating', 'comment', 'review', 'sentiment', 'opinion']):
        return [{
            'table_name': 'USER_FEEDBACK',
            'schema': 'PRODUCT_ANALYTICS',
            'full_path': 'ONCODETECT_DB.PRODUCT_ANALYTICS.USER_FEEDBACK',
            'description': 'User feedback ratings and comments',
            'columns': 'FEEDBACK_ID, USER_ID, USERNAME, FEEDBACK_TEXT, RATING, FEEDBACK_DATE'
        }]

    # --- RULE 2: USER ACQUISITION / COUNTS ---
    # Phrases like "how many users", "total users", "list of users"
    if any(k in q for k in ['how many users', 'total users', 'count users', 'registered users', 'new accounts']):
        return [{
            'table_name': 'USERS',
            'schema': 'USER_MGMT',
            'full_path': 'ONCODETECT_DB.USER_MGMT.USERS',
            'description': 'Registered user accounts and demographics',
            'columns': 'USER_ID, FIRST_NAME, LAST_NAME, EMAIL, USERNAME, USER_ROLE, CREATED_AT'
        }]

    # --- RULE 3: FEATURE USAGE / CLICKS / EVENTS ---
    # Default for almost everything else related to "usage"
    if any(k in q for k in ['click', 'button', 'usage', 'feature', 'login', 'signup', 'event', 'trend', 'find_centers', 'risk', 'research', 'subtype']):
        return [{
            'table_name': 'USER_ACQUISITION_EVENTS',
            'schema': 'PRODUCT_ANALYTICS',
            'full_path': 'ONCODETECT_DB.PRODUCT_ANALYTICS.USER_ACQUISITION_EVENTS',
            'description': 'Button clicks, logins, feature usage events',
            'columns': 'EVENT_TYPE, BUTTON_NAME, STATUS, USER_ID, APP_TIMESTAMP'
        }]

    # --- FALLBACK: VECTOR SEARCH ---
    # If the rule-based router is unsure, use embeddings
    try:
        question_escaped = user_question.replace("'", "''")
        query = f"""
        WITH question_embedding AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', '{question_escaped}') as embedding
        )
        SELECT TABLE_NAME, TABLE_SCHEMA, FULL_TABLE_PATH, DESCRIPTION, COLUMNS_INFO
        FROM ONCODETECT_DB.PRODUCT_ANALYTICS.TABLE_METADATA m, question_embedding q
        ORDER BY VECTOR_COSINE_SIMILARITY(m.METADATA_EMBEDDING, q.embedding) DESC
        LIMIT 1
        """
        results = session.sql(query).collect()
        if results:
            r = results[0]
            return [{
                'table_name': r['TABLE_NAME'],
                'schema': r['TABLE_SCHEMA'],
                'full_path': r['FULL_TABLE_PATH'],
                'description': r['DESCRIPTION'],
                'columns': r['COLUMNS_INFO']
            }]
    except:
        pass
    
    # Default fallback
    return []

def generate_sql_from_question(user_question, relevant_tables):
    """Generate SQL query using LLM based on user question"""
    try:
        if not relevant_tables: return None
        
        t = relevant_tables[0] # We routed to exactly one table
        
        # Specific Hints based on Table
        hints = ""
        if t['table_name'] == 'USER_ACQUISITION_EVENTS':
            hints = """
            - 'event_type' contains: 'button_click', 'user_login_success', 'user_signup_success'
            - 'button_name' contains: 'risk_quick_prediction', 'find_centers', 'submit_feedback', etc.
            - If asking for 'distribution' or 'breakdown', group by the categorical column (like button_name), NOT time.
            - Only group by time (APP_TIMESTAMP) if the user explicitly asks for "trend", "over time", or "daily".
            """
        elif t['table_name'] == 'USER_FEEDBACK':
            hints = "- Use SNOWFLAKE.CORTEX.SENTIMENT(FEEDBACK_TEXT) if asked for sentiment."
            
        prompt = f"""Generate a Snowflake SQL query to answer: "{user_question}"

        Table: {t['full_path']}
        Description: {t['description']}
        Columns: {t['columns']}

        Instructions:
        - Return ONLY the SQL query. No text.
        - Start with SELECT.
        - Limit to 100 rows.
        {hints}

        SQL Query:"""

        prompt_escaped = prompt.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_escaped}') as sql_query").collect()
        
        # Clean the output
        sql = result[0]['SQL_QUERY'].replace('```sql', '').replace('```', '').strip()
        return sql
    except Exception as e:
        return None

def generate_insights_from_data(user_question, df, sql_query):
    """Generate text insights from query results using LLM"""
    try:
        if df.empty: return "No data found for this query."
        
        data_summary = f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
        data_summary += f"Data Sample:\n{df.head(5).to_string()}\n"
        
        prompt = f"""You are a product analyst. Analyze this data to answer: "{user_question}"

        Data:
        {data_summary}

        Provide 3 short, actionable insights (bullet points). Focus on trends, user behavior, or anomalies.
        """

        prompt_escaped = prompt.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_escaped}') as insights").collect()
        return result[0]['INSIGHTS'].strip()
    except:
        return "Could not generate insights."

def create_visualization(df, user_question, unique_key):
    """Auto-generate Altair chart with Navy Blue theme"""
    if df is None or df.empty: 
        return False
    
    try:
        # Identify column types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        if not numeric_cols: return False

        # --- 1. DETERMINE DEFAULT CHART TYPE ---
        default_chart = "Bar Chart"
        
        # Only default to Line if we have a real Date column AND the user asked for time
        if date_cols:
            default_chart = "Line Chart"
            x_axis_col = date_cols[0]
        elif cat_cols:
            x_axis_col = cat_cols[0]
        else:
            x_axis_col = df.index.name or "Index"

        # --- 2. USER SELECTION ---
        chart_type = st.selectbox(
            "Select Chart Type",
            options=["Bar Chart", "Line Chart", "Area Chart", "Donut Chart", "No Chart"],
            index=["Bar Chart", "Line Chart", "Area Chart", "Donut Chart", "No Chart"].index(default_chart),
            key=f"viz_select_{unique_key}",
            label_visibility="collapsed"
        )
        
        if chart_type == "No Chart": return False

        # --- 3. ALTAIR CHART CONFIGURATION (Navy Blue #081f37) ---
        base = alt.Chart(df).encode(
            tooltip=df.columns.tolist()
        )
        
        y_axis_col = numeric_cols[0]
        
        chart = None
        
        if chart_type == "Bar Chart":
            chart = base.mark_bar(color='#081f37').encode(
                x=alt.X(f'{x_axis_col}:N', sort='-y', title=x_axis_col),
                y=alt.Y(f'{y_axis_col}:Q', title=y_axis_col)
            )
            
        elif chart_type == "Line Chart":
            chart = base.mark_line(color='#081f37', point=True).encode(
                x=alt.X(f'{x_axis_col}', title=x_axis_col), # Temporal inference automatic
                y=alt.Y(f'{y_axis_col}:Q', title=y_axis_col)
            )
            
        elif chart_type == "Area Chart":
            chart = base.mark_area(color='#081f37', opacity=0.7).encode(
                x=alt.X(f'{x_axis_col}', title=x_axis_col),
                y=alt.Y(f'{y_axis_col}:Q', title=y_axis_col)
            )
            
        elif chart_type == "Donut Chart":
            base = alt.Chart(df).encode(
                theta=alt.Theta(f"{y_axis_col}:Q", stack=True)
            )
            pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
                color=alt.Color(f"{x_axis_col}:N", scale=alt.Scale(scheme='blues')), # Blue scheme for categories
                order=alt.Order(f"{y_axis_col}:Q", sort="descending"),
                tooltip=[x_axis_col, y_axis_col]
            )
            text = base.mark_text(radius=140).encode(
                text=alt.Text(f"{y_axis_col}:Q", format=".0f"),
                order=alt.Order(f"{y_axis_col}:Q", sort="descending"),
                color=alt.value("#081f37")  
            )
            chart = pie + text

        if chart:
            st.altair_chart(chart, use_container_width=True)
            return True
            
        return False
    except Exception as e:
        st.error(f"Viz Error: {e}")
        return False

def execute_analytics_query(user_question):
    """Execute complete analytics pipeline"""
    try:
        # 1. Route to Table
        relevant_tables = find_relevant_tables(user_question)
        if not relevant_tables:
            return None, None, None, "Could not identify relevant data for this question."
            
        # 2. Generate SQL
        with st.spinner("💡 Writing SQL..."):
            sql_query = generate_sql_from_question(user_question, relevant_tables)
            if not sql_query: return None, None, None, "Failed to generate SQL."
            
        # 3. Execute
        with st.spinner("📊 Querying Snowflake..."):
            results = session.sql(sql_query).collect()
            df = pd.DataFrame([row.asDict() for row in results])
            
        # 4. Insights
        insights = None
        if not df.empty:
            with st.spinner("🧠 Generating insights..."):
                insights = generate_insights_from_data(user_question, df, sql_query)
                
        return df, sql_query, insights, relevant_tables
        
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

# ============================================================
# 8. RESEARCH NOTE FUNCTIONS
# ============================================================

def save_as_research_note(user_id, username, chat_history, note_title):
    """Save current chat as research note"""
    try:
        total_queries = sum(1 for msg in chat_history if msg['role'] == 'user')
        chat_json_str = json.dumps(chat_history)
        title_escaped = note_title.replace("'", "''")
        username_escaped = username.replace("'", "''")
        
        query = f"""
        INSERT INTO ONCODETECT_DB.UNSTRUCTURED_DATA.RESEARCH_NOTES 
        (USER_ID, USERNAME, NOTE_TITLE, CHAT_HISTORY, TOTAL_QUERIES)
        SELECT {user_id}, '{username_escaped}', '{title_escaped}', PARSE_JSON($${chat_json_str}$$), {total_queries}
        """
        session.sql(query).collect()
        return True
    except:
        return False

def load_research_notes(user_id):
    """Load all research notes for a user"""
    try:
        query = f"""
        SELECT NOTE_ID, NOTE_TITLE, CHAT_HISTORY, TOTAL_QUERIES, CREATED_AT
        FROM ONCODETECT_DB.UNSTRUCTURED_DATA.RESEARCH_NOTES
        WHERE USER_ID = {user_id} 
        ORDER BY CREATED_AT DESC
        """
        results = session.sql(query).collect()
        
        notes = []
        for row in results:
            try:
                chat_history_parsed = json.loads(str(row['CHAT_HISTORY']))
            except:
                chat_history_parsed = row['CHAT_HISTORY']
            
            notes.append({
                'note_id': row['NOTE_ID'],
                'title': row['NOTE_TITLE'],
                'chat_history': chat_history_parsed,
                'total_queries': row['TOTAL_QUERIES'],
                'created_at': row['CREATED_AT']
            })
        st.session_state.research_notes = notes
    except:
        st.session_state.research_notes = []

def view_research_note(note):
    """Load a specific research note for viewing"""
    st.session_state.viewing_note = note
    st.session_state.chat_history = note['chat_history']
    st.session_state.show_sources = {}

def start_new_chat():
    """Clear current chat and start fresh"""
    st.session_state.viewing_note = None
    st.session_state.chat_history = []
    st.session_state.show_sources = {}

# ============================================================
# 9. FEEDBACK FUNCTIONS
# ============================================================

def save_user_feedback(user_id, username, feedback_text, rating):
    """Save user feedback to database"""
    try:
        feedback_escaped = feedback_text.replace("'", "''")
        username_escaped = username.replace("'", "''")
        
        query = f"""
        INSERT INTO ONCODETECT_DB.PRODUCT_ANALYTICS.USER_FEEDBACK 
        (USER_ID, USERNAME, FEEDBACK_TEXT, RATING, FEEDBACK_DATE)
        VALUES ({user_id}, '{username_escaped}', '{feedback_escaped}', {rating}, CURRENT_TIMESTAMP())
        """
        session.sql(query).collect()
        return True, "Feedback submitted successfully! ✅"
    except Exception as e:
        return False, f"Error submitting feedback: {str(e)}"

def toggle_feedback():
    """Toggle feedback form visibility"""
    st.session_state.show_feedback = not st.session_state.show_feedback

# ============================================================
# 10. WEB SEARCH AGENT
# ============================================================

def get_serp_api_key():
    """Retrieve SERP API key from database config table"""
    try:
        result = session.sql("""
            SELECT CONFIG_VALUE 
            FROM ONCODETECT_DB.USER_MGMT.SERP_CONFIG 
            WHERE CONFIG_KEY = 'SERP_API_KEY'
        """).collect()
        
        if result and len(result) > 0:
            api_key = result[0]['CONFIG_VALUE']
            if api_key:
                return str(api_key).strip()
        return None
    except:
        return None

def search_web_serp(query_text, num_results=5):
    """
    Search the web using SERP API for current SCLC information
    Returns: (summary_text, source_metadata_list)
    """
    try:
        # Get API key
        api_key = get_serp_api_key()
        if not api_key:
            return "Web search unavailable - API key not configured", []
        
        # Enhance query for SCLC research
        enhanced_query = f"Small Cell Lung Cancer SCLC {query_text}"
        
        # Make SERP API request
        api_url = "https://serpapi.com/search"
        params = {
            'q': enhanced_query,
            'api_key': api_key,
            'engine': 'google',
            'num': num_results,
            'hl': 'en',
            'gl': 'us'
        }
        
        response = requests.get(api_url, params=params, timeout=30)
        
        # Handle non-200 responses
        if response.status_code == 401:
            return "Authentication failed - API key invalid", []
        elif response.status_code == 403:
            return "Access forbidden", []
        
        response.raise_for_status()
        data = response.json()
        
        organic_results = data.get('organic_results', [])
        
        if not organic_results:
            return "No relevant web results found.", []
        
        # Process results
        web_context = ""
        source_metadata = []
        
        for idx, result in enumerate(organic_results, 1):
            title = result.get('title', 'Unknown Title')
            snippet = result.get('snippet', 'No description')
            link = result.get('link', '')
            
            web_context += f"\n\n[Result {idx}]\nTitle: {title}\nSource: {link}\nContent: {snippet}\n"
            
            source_metadata.append({
                'title': title,
                'url': link,
                'snippet': snippet
            })
        
        # Generate summary using LLM
        query_escaped = query_text.replace("'", "''")
        web_escaped = web_context.replace("'", "''")
        
        prompt = f"""You are an expert medical research assistant. Based on current web search results, provide a comprehensive answer about Small Cell Lung Cancer (SCLC).

WEB RESULTS:
{web_escaped}

QUESTION: {query_escaped}

Instructions:
- Synthesize information from web results
- Provide 350-400 word detailed response
- Use medical terminology appropriately
- Reference current information
- Do not mention source titles or URLs

Answer:"""
        
        prompt_escaped = prompt.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_escaped}') as web_summary").collect()
        
        return result[0]['WEB_SUMMARY'], source_metadata
        
    except requests.exceptions.Timeout:
        return "Web search timed out.", []
    except requests.exceptions.ConnectionError:
        return "Cannot connect to web search service.", []
    except requests.exceptions.HTTPError as e:
        return f"Web search API error (HTTP {e.response.status_code}).", []
    except Exception as e:
        return f"Error during web search: {str(e)}", []

# ============================================================
# 11. ARXIV RESEARCH AGENT
# ============================================================

def search_and_summarize_arxiv(query_text, max_results=3):
    """
    Search Arxiv for SCLC research papers and generate summary
    Returns: (summary_text, paper_metadata_list)
    """
    try:
        # Enhance query for SCLC
        enhanced_query = f"Small Cell Lung Cancer SCLC {query_text}"
        encoded_query = quote(enhanced_query)
        
        # Query Arxiv API
        api_url = f"https://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
        
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            return "No relevant Arxiv papers found.", []
        
        # Process papers
        papers_context = ""
        paper_metadata = []
        
        for idx, entry in enumerate(entries, 1):
            # Extract paper info
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else "Unknown"
            
            authors = entry.findall('atom:author', ns)
            author_names = [a.find('atom:name', ns).text for a in authors[:3] if a.find('atom:name', ns) is not None]
            author_str = ', '.join(author_names) if author_names else "Unknown"
            if len(authors) > 3:
                author_str += " et al."
            
            published_elem = entry.find('atom:published', ns)
            published = published_elem.text[:10] if published_elem is not None else "Unknown"
            
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else "No abstract"
            
            id_elem = entry.find('atom:id', ns)
            arxiv_url = id_elem.text if id_elem is not None else "https://arxiv.org"
            arxiv_id = arxiv_url.split('/')[-1]
            
            papers_context += f"\n\n[Paper {idx}]\nTitle: {title}\nAuthors: {author_str}\nPublished: {published}\nAbstract: {abstract}\n"
            
            paper_metadata.append({
                'title': title,
                'authors': author_str,
                'published': published,
                'arxiv_id': arxiv_id,
                'url': arxiv_url
            })
        
        # Generate summary using LLM
        query_escaped = query_text.replace("'", "''")
        papers_escaped = papers_context.replace("'", "''")
        
        prompt = f"""You are an expert medical research assistant. Based on Arxiv research papers, provide a comprehensive answer about SCLC.

ARXIV PAPERS:
{papers_escaped}

QUESTION: {query_escaped}

Instructions:
- Synthesize from research papers
- Provide 350-400 word detailed response
- Use medical terminology
- Reference findings naturally
- Do not mention paper titles or IDs

Answer:"""
        
        prompt_escaped = prompt.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_escaped}') as arxiv_summary").collect()
        
        return result[0]['ARXIV_SUMMARY'], paper_metadata
        
    except Exception as e:
        return f"Arxiv search error: {str(e)}", []

# ============================================================
# 12. RAG (RETRIEVAL-AUGMENTED GENERATION) AGENT
# ============================================================

def search_text_embeddings(query_text, top_k=5):
    """Search text embeddings for relevant chunks"""
    try:
        query_escaped = query_text.replace("'", "''")
        query = f"""
        WITH query_embedding AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', '{query_escaped}') as embedding
        )
        SELECT t.CHUNK_TEXT, t.PAPER_ID, t.CHUNK_INDEX, t.START_PAGE, t.END_PAGE,
               VECTOR_COSINE_SIMILARITY(t.EMBEDDING, q.embedding) as similarity_score
        FROM ONCODETECT_DB.UNSTRUCTURED_DATA.TEXT_EMBEDDINGS_TABLE t, query_embedding q
        WHERE VECTOR_COSINE_SIMILARITY(t.EMBEDDING, q.embedding) > 0.3
        ORDER BY similarity_score DESC 
        LIMIT {top_k}
        """
        return session.sql(query).collect()
    except:
        return []

def search_image_embeddings(query_text, top_k=5):
    """Search image embeddings for relevant medical images"""
    try:
        query_escaped = query_text.replace("'", "''")
        
        query = f"""
        WITH query_embedding AS (
            SELECT AI_EMBED('voyage-multimodal-3', '{query_escaped}') as embedding
        )
        SELECT i.PAPER_ID, i.PAGE_NUMBER, i.IMAGE_INDEX, i.S3_URL, 
               i.WIDTH, i.HEIGHT, i.FORMAT,
               VECTOR_COSINE_SIMILARITY(i.EMBEDDING, q.embedding) as similarity_score
        FROM ONCODETECT_DB.UNSTRUCTURED_DATA.IMAGE_EMBEDDINGS_TABLE i, query_embedding q
        WHERE VECTOR_COSINE_SIMILARITY(i.EMBEDDING, q.embedding) > 0.25
        ORDER BY similarity_score DESC 
        LIMIT {top_k}
        """
        return session.sql(query).collect()
    except:
        return []

def generate_rag_response(query_text, text_results, image_results):
    """
    Generate response from RAG sources with confidence scoring
    Returns: (response_text, confidence_score)
    """
    try:
        if not text_results and not image_results:
            return "No relevant information found in our database.", 0.0
        
        # Categorize by confidence levels
        high_conf_text = [r for r in text_results if r['SIMILARITY_SCORE'] > 0.7]
        med_conf_text = [r for r in text_results if 0.5 <= r['SIMILARITY_SCORE'] <= 0.7]
        low_conf_text = [r for r in text_results if 0.3 <= r['SIMILARITY_SCORE'] < 0.5]
        
        high_conf_images = [r for r in image_results if r['SIMILARITY_SCORE'] > 0.6]
        med_conf_images = [r for r in image_results if 0.4 <= r['SIMILARITY_SCORE'] <= 0.6]
        low_conf_images = [r for r in image_results if 0.25 <= r['SIMILARITY_SCORE'] < 0.4]
        
        # Calculate weighted confidence
        all_scores = [r['SIMILARITY_SCORE'] for r in text_results] + [r['SIMILARITY_SCORE'] for r in image_results]
        weighted_relevance = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        # Build tiered context
        combined_context = ""
        
        if high_conf_text:
            combined_context += "HIGH CONFIDENCE TEXT:\n"
            for r in high_conf_text:
                combined_context += f"[Paper {r['PAPER_ID']}, Pages {r['START_PAGE']}-{r['END_PAGE']}]\n{r['CHUNK_TEXT']}\n\n"
        
        if high_conf_images:
            combined_context += "HIGH CONFIDENCE IMAGES:\n"
            for r in high_conf_images:
                combined_context += f"[Paper {r['PAPER_ID']}, Page {r['PAGE_NUMBER']}, Image {r['IMAGE_INDEX']}]\nVisual evidence from research.\n\n"
        
        if med_conf_text:
            combined_context += "MEDIUM CONFIDENCE TEXT:\n"
            for r in med_conf_text:
                combined_context += f"[Paper {r['PAPER_ID']}, Pages {r['START_PAGE']}-{r['END_PAGE']}]\n{r['CHUNK_TEXT']}\n\n"
        
        if med_conf_images:
            combined_context += "MEDIUM CONFIDENCE IMAGES:\n"
            for r in med_conf_images:
                combined_context += f"[Paper {r['PAPER_ID']}, Page {r['PAGE_NUMBER']}, Image {r['IMAGE_INDEX']}]\nSupporting visual.\n\n"
        
        if low_conf_text:
            combined_context += "LOWER CONFIDENCE TEXT:\n"
            for r in low_conf_text:
                combined_context += f"[Paper {r['PAPER_ID']}, Pages {r['START_PAGE']}-{r['END_PAGE']}]\n{r['CHUNK_TEXT']}\n\n"
        
        if low_conf_images:
            combined_context += "LOWER CONFIDENCE IMAGES:\n"
            for r in low_conf_images:
                combined_context += f"[Paper {r['PAPER_ID']}, Page {r['PAGE_NUMBER']}, Image {r['IMAGE_INDEX']}]\nAdditional visual.\n\n"
        
        # Generate response using LLM
        prompt = f"""You are an expert medical research assistant specializing in SCLC.

Based on tiered research evidence, provide comprehensive answer.

{combined_context}

QUESTION: {query_text}

Instructions:
- Provide 350-400 word detailed response
- Prioritize HIGH CONFIDENCE sources
- Use MEDIUM CONFIDENCE for supporting details
- Use LOWER CONFIDENCE sparingly
- Synthesize text and image sources
- Use medical terminology
- Do not mention paper IDs or confidence levels

Answer:"""

        prompt_escaped = prompt.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_escaped}') as response").collect()
        
        return result[0]['RESPONSE'], weighted_relevance
        
    except Exception as e:
        return "Error generating response.", 0.0

# ============================================================
# 13. RESPONSE BLENDING FUNCTIONS
# ============================================================

def blend_rag_arxiv_responses(query_text, rag_response, arxiv_response, rag_confidence):
    """Blend RAG and Arxiv responses for medium-high confidence scenarios"""
    try:
        query_escaped = query_text.replace("'", "''")
        rag_escaped = rag_response.replace("'", "''")
        
        arxiv_text = arxiv_response[0] if isinstance(arxiv_response, tuple) else arxiv_response
        arxiv_escaped = arxiv_text.replace("'", "''")
        
        prompt = f"""You are an expert medical research assistant. Synthesize information from in-house database and Arxiv research.

IN-HOUSE DATABASE:
{rag_escaped}

ARXIV RESEARCH:
{arxiv_escaped}

QUESTION: {query_escaped}

Instructions:
- Synthesize BOTH sources into 400-450 word response
- Prioritize in-house database, enhance with Arxiv
- Identify complementary information
- Use medical terminology
- Do not mention sources explicitly

Blended Response:"""

        prompt_escaped = prompt.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_escaped}') as blended").collect()
        
        return result[0]['BLENDED']
        
    except:
        return rag_response

def blend_arxiv_web_responses(query_text, arxiv_response, web_response):
    """Blend Arxiv and Web responses for low confidence scenarios"""
    try:
        query_escaped = query_text.replace("'", "''")
        
        arxiv_text = arxiv_response[0] if isinstance(arxiv_response, tuple) else arxiv_response
        arxiv_escaped = arxiv_text.replace("'", "''")
        
        web_text = web_response[0] if isinstance(web_response, tuple) else web_response
        web_escaped = web_text.replace("'", "''")
        
        prompt = f"""You are an expert medical research assistant. Synthesize Arxiv research and current web information.

ARXIV RESEARCH:
{arxiv_escaped}

WEB SEARCH:
{web_escaped}

QUESTION: {query_escaped}

Instructions:
- Synthesize BOTH sources into 400-450 word response
- Integrate academic research with current web info
- Use medical terminology
- Do not mention sources explicitly

Blended Response:"""

        prompt_escaped = prompt.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_escaped}') as blended").collect()
        
        return result[0]['BLENDED']
        
    except:
        return arxiv_response

def blend_multi_agent_responses(query_text, rag_response, arxiv_response, web_response, rag_confidence):
    """Blend all three agents for comprehensive coverage"""
    try:
        query_escaped = query_text.replace("'", "''")
        rag_escaped = rag_response.replace("'", "''")
        
        arxiv_text = arxiv_response[0] if isinstance(arxiv_response, tuple) else arxiv_response
        arxiv_escaped = arxiv_text.replace("'", "''")
        
        web_text = web_response[0] if isinstance(web_response, tuple) else web_response
        web_escaped = web_text.replace("'", "''")
        
        prompt = f"""You are an expert medical research assistant. Synthesize information from THREE sources.

IN-HOUSE DATABASE:
{rag_escaped}

ARXIV RESEARCH:
{arxiv_escaped}

WEB SEARCH:
{web_escaped}

QUESTION: {query_escaped}

Instructions:
- Synthesize ALL THREE sources into 450-500 word response
- Prioritize in-house, enhance with Arxiv and web
- Integrate current web info with academic research
- Use medical terminology
- Do not mention sources explicitly

Blended Response:"""

        prompt_escaped = prompt.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', '{prompt_escaped}') as blended").collect()
        
        return result[0]['BLENDED']
        
    except:
        return rag_response

# ============================================================
# 14. MULTI-AGENT QUERY PROCESSING
# ============================================================

def process_user_query(query_text, user_id, username):
    """
    Main query processing with intelligent multi-agent routing
    
    Routing Logic:
    - RAG > 60%: Use RAG only
    - RAG < 50%: Use Arxiv + Web
    - RAG 50-60%: Test with Arxiv first
        - If combined > 60%: RAG + Arxiv
        - If combined < 50%: Web only
        - If combined 50-60%: All 3 agents
    
    Returns: (response, text_results, image_results, response_time, has_sources,
              search_status, rag_confidence, combined_confidence, agent_used,
              rag_response, arxiv_response, arxiv_papers, web_response, web_sources)
    """
    start_time = time.time()
    search_status = []
    
    # Step 1: Search text embeddings
    with st.spinner("🔍 Searching through text sources..."):
        text_results = search_text_embeddings(query_text, top_k=5)
        search_status.append(f"📚 Found **{len(text_results)}** text sources")
        time.sleep(1)
    
    # Step 2: Search image embeddings
    with st.spinner("🔍 Searching through image sources..."):
        image_results = search_image_embeddings(query_text, top_k=5)
        search_status.append(f"🖼️ Found **{len(image_results)}** image sources")
        time.sleep(1)
    
    # Step 3: Generate RAG response and calculate confidence
    with st.spinner("🤖 Generating response from in-house sources..."):
        rag_response, rag_confidence = generate_rag_response(query_text, text_results, image_results)
        time.sleep(0.5)
    
    # Initialize variables
    agent_used = "RAG"
    final_response = rag_response
    arxiv_response = None
    arxiv_papers = []
    web_response = None
    web_sources = []
    combined_confidence = rag_confidence
    
    # ========== INTELLIGENT MULTI-AGENT ROUTING ==========
    
    # SCENARIO 1: High Confidence (> 60%) - Use RAG only
    if rag_confidence > 0.6:
        search_status.append(f"**Using RAG Agent** - High quality in-house data available")
        final_response = rag_response
        agent_used = "RAG"
        combined_confidence = rag_confidence
    
    # SCENARIO 2: Low Confidence (< 50%) - Use Arxiv + Web
    elif rag_confidence < 0.5:
        search_status.append(f"**Consulting external sources** - Searching Arxiv & Web...")
        
        # Search Arxiv
        with st.spinner("📚 Searching Arxiv research papers..."):
            arxiv_result = search_and_summarize_arxiv(query_text)
            arxiv_response = arxiv_result[0]
            arxiv_papers = arxiv_result[1]
            time.sleep(1)
        
        # Search Web
        with st.spinner("🌐 Searching the web..."):
            web_result = search_web_serp(query_text)
            web_response = web_result[0]
            web_sources = web_result[1]
            time.sleep(1)
        
        # Calculate combined confidence
        arxiv_boost = 0.15 if arxiv_papers and len(arxiv_papers) >= 2 else 0.1
        web_boost = 0.15 if web_sources and len(web_sources) >= 3 else 0.1
        combined_confidence = min(rag_confidence + arxiv_boost + web_boost, 0.95)
        
        search_status.append(f"**Using Arxiv + Web Agents** - External research sources")
        
        # Blend responses
        with st.spinner("🔄 Synthesizing Arxiv + Web responses..."):
            final_response = blend_arxiv_web_responses(query_text, arxiv_response, web_response)
            time.sleep(0.5)
        
        agent_used = "Arxiv + Web"
    
    # SCENARIO 3: Medium Confidence (50-60%) - Conditional routing
    else:
        search_status.append(f"**Enhancing with Arxiv** - Testing external research...")
        
        # Try Arxiv first
        with st.spinner("📚 Searching Arxiv research papers..."):
            arxiv_result = search_and_summarize_arxiv(query_text)
            arxiv_response = arxiv_result[0]
            arxiv_papers = arxiv_result[1]
            time.sleep(1)
        
        # Calculate confidence after Arxiv
        arxiv_boost = 0.1 if arxiv_papers and len(arxiv_papers) >= 2 else 0.05
        combined_confidence = min(rag_confidence + arxiv_boost, 1.0)
        
        # SUB-SCENARIO 3A: Combined > 60% - Use RAG + Arxiv
        if combined_confidence > 0.6:
            search_status.append(f"**Using RAG + Arxiv Agents** - In-house data enhanced with research")
            
            with st.spinner("🔄 Synthesizing RAG + Arxiv..."):
                final_response = blend_rag_arxiv_responses(query_text, rag_response, arxiv_response, rag_confidence)
                time.sleep(0.5)
            
            agent_used = "RAG + Arxiv"
        
        # SUB-SCENARIO 3B: Combined < 50% - Use Web only
        elif combined_confidence < 0.5:
            search_status.append(f"**Adding Web Search** - Consulting additional sources...")
            
            with st.spinner("🌐 Searching the web..."):
                web_result = search_web_serp(query_text)
                web_response = web_result[0]
                web_sources = web_result[1]
                time.sleep(1)
            
            # Calculate web confidence
            web_boost = 0.2 if web_sources and len(web_sources) >= 3 else 0.15
            combined_confidence = min(rag_confidence + web_boost, 0.7)
            
            search_status.append(f"**Using Web Agent** - Current web information")
            
            final_response = web_response[0] if isinstance(web_response, tuple) else web_response
            agent_used = "Web"
        
        # SUB-SCENARIO 3C: Combined 50-60% - Use all 3 agents
        else:
            search_status.append(f"**Comprehensive search** - Adding Web sources...")
            
            with st.spinner("🌐 Searching the web..."):
                web_result = search_web_serp(query_text)
                web_response = web_result[0]
                web_sources = web_result[1]
                time.sleep(1)
            
            search_status.append(f"**Using All 3 Agents** - RAG + Arxiv + Web (Comprehensive)")
            
            with st.spinner("🔄 Synthesizing RAG + Arxiv + Web..."):
                final_response = blend_multi_agent_responses(query_text, rag_response, arxiv_response, web_response, rag_confidence)
                time.sleep(0.5)
            
            agent_used = "RAG + Arxiv + Web"
    
    # Calculate response time
    response_time_ms = int((time.time() - start_time) * 1000)
    has_sources = len(text_results) > 0 or len(image_results) > 0
    
    return (final_response, text_results, image_results, response_time_ms, has_sources, 
            search_status, rag_confidence, combined_confidence, agent_used, rag_response, 
            arxiv_response, arxiv_papers, web_response, web_sources)

# ============================================================
# 15. UTILITY FUNCTIONS
# ============================================================

def translate_text(text, target_language):
    """Translate text to target language using Snowflake Cortex"""
    try:
        text_escaped = text.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.TRANSLATE('{text_escaped}', 'en', '{target_language}') as translated").collect()
        return result[0]['TRANSLATED']
    except:
        return None

def summarize_text(text):
    """Summarize text to 75 words using Snowflake Cortex"""
    try:
        text_escaped = text.replace("'", "''")
        result = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large', 'Summarize in exactly 75 words:\n\n{text_escaped}') as summary").collect()
        return result[0]['SUMMARY']
    except:
        return None

def display_typing_effect(text, placeholder):
    """Display text with typing animation effect"""
    if isinstance(text, tuple):
        text = text[0]
    
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(f"<div class='bot-message'><strong>OncoDetect AI:</strong><br>{displayed_text}<span class='typing-cursor'>|</span></div>", unsafe_allow_html=True)
        time.sleep(0.01)
    placeholder.markdown(f"<div class='bot-message'><strong>OncoDetect AI:</strong><br>{displayed_text}</div>", unsafe_allow_html=True)

# ============================================================
# 16. UI PAGES - AUTHENTICATION
# ============================================================

def render_sidebar_profile():
    """Render user profile in sidebar"""
    with st.sidebar:
        # Fetch name logic
        try:
            user_query = f"SELECT FIRST_NAME, LAST_NAME FROM ONCODETECT_DB.USER_MGMT.USERS WHERE USER_ID = {st.session_state.user_id}"
            user_result = session.sql(user_query).collect()
            if user_result:
                full_name = f"{user_result[0]['FIRST_NAME']} {user_result[0]['LAST_NAME']}"
            else:
                full_name = st.session_state.username
        except:
            full_name = st.session_state.username

        # Adjusted to match original profile card format (Name & ID only)
        st.markdown(f"""
        <div class='profile-card' style='padding: 1rem; margin-bottom: 1.5rem;'>
            <div class='profile-card-item'>
                <div class='profile-card-label'>Name</div>
                <div class='profile-card-value'>{full_name}</div>
            </div>
            <div class='profile-card-item'>
                <div class='profile-card-label'>User ID</div>
                <div class='profile-card-value'>{st.session_state.user_id}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
def show_login_page():
    """Display login page"""
    st.markdown("<h1 style='text-align: center;'>ONCODETECT AI 🫁🎗️</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h3 style='font-size: 16px;'>Login to Your Account🔐!</h3>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        login_username = st.text_input("Username", placeholder="Enter username")
        login_password = st.text_input("Password", type="password", placeholder="Enter password")
        login_role = st.selectbox("Login as", options=["user", "admin"])
        
        submit_login = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        if submit_login:
            if not login_username or not login_password:
                st.error("Please enter both username and password⚠️")
            else:
                success, user_data = login_user(login_username, login_password, login_role)
                if success:
                    # --- KAFKA EVENT: SUCCESS ---
                    kafka_sent = send_kafka_event(
                        topic_name="sclc_user_authentication",  # Ensure underscores match your topic
                        event_type="user_login_success",
                        button_name="login_submit",
                        user_id=str(user_data['USER_ID']),
                        status="success"
                    )

                    st.session_state.logged_in = True
                    st.session_state.username = user_data['USERNAME']
                    st.session_state.user_id = user_data['USER_ID']
                    st.session_state.user_role = user_data['USER_ROLE']
                    st.success(f"Login successful🔓")
                    st.rerun()
                else:
                    # --- KAFKA EVENT: FAILURE ---
                    send_kafka_event(
                        topic_name="sclc_user_authentication",
                        event_type="user_login_failure",
                        button_name="login_submit",
                        user_id=None, 
                        status="failure"
                    )

                    st.error(f"❌ Invalid credentials")
    
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<p style='font-size: 13px; text-align: center;'>New User? → Create Account👤</p>", unsafe_allow_html=True)
        if st.button("📝 Sign Up", use_container_width=True):
            go_to_signup()
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 2rem;'>
        <p><strong>OncoDetect-AI</strong> | Advanced Precision Oncology Platform</p>
        <p>Powered by Snowflake ML + Cortex AI</p>
        <p style="font-size: 0.85rem;">For Research Use Only | Not for Clinical Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

def show_signup_page():
    """Display signup page"""
    st.markdown("<h1 style='text-align: center;'>ONCODETECT AI 🫁🎗️</h1>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h3 style='font-size: 16px;'>Create New Account</h3>", unsafe_allow_html=True)
    
    with st.form("signup_form"):
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name *", placeholder="First name")
        with col2:
            last_name = st.text_input("Last Name *", placeholder="Last name")
        
        dob = st.date_input("Date of Birth *", min_value=date(1900, 1, 1), max_value=date.today(), value=date(2000, 1, 1))
        email = st.text_input("Email *", placeholder="email@example.com")
        username = st.text_input("Username *", placeholder="Username")
        
        signup_role = st.selectbox("Account Type *", options=["user", "admin"],
                                   help="User: Research | Admin: Analytics")
        
        col3, col4 = st.columns(2)
        with col3:
            password = st.text_input("Password *", type="password", placeholder="Password")
        with col4:
            confirm_password = st.text_input("Confirm *", type="password", placeholder="Confirm")
        
        st.markdown("<p style='font-size: 13px;'><strong>📋 Requirements:</strong> 8+ chars, 1 uppercase, 1 number, 1 special</p>", unsafe_allow_html=True)
        
        submit_signup = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        
        if submit_signup:
            # Validate all fields
            if not all([first_name, last_name, email, username, password, confirm_password]):
                st.error("❌ Fill all fields")
            elif password != confirm_password:
                st.error("❌ Passwords don't match")
            else:
                # Validate email
                email_valid, email_msg = validate_email(email)
                if not email_valid:
                    st.error(f"❌ {email_msg}")
                    st.stop()
                if email_exists(email):
                    st.error("❌ Email registered")
                    st.stop()
                
                # Validate username
                username_valid, username_msg = validate_username(username)
                if not username_valid:
                    st.error(f"❌ {username_msg}")
                    st.stop()
                if username_exists(username):
                    st.error("❌ Username taken")
                    st.stop()
                
                # Validate password
                password_valid, password_msg = validate_password(password)
                if not password_valid:
                    st.error(f"❌ {password_msg}")
                    st.stop()
                
                # --- INDENTATION FIX START ---
                # This code is now INSIDE the 'else' block, so it only runs if validations pass!
                
                # Create account
                success, message = signup_user(first_name, last_name, dob, email, username, password, signup_role)
                
                if success:
                    # ✅ SIGNUP SUCCESS
                    
                    # Get the new user_id
                    try:
                        user_query = f"SELECT USER_ID FROM ONCODETECT_DB.USER_MGMT.USERS WHERE USERNAME = '{username}'"
                        user_result = session.sql(user_query).collect()
                        new_user_id = str(user_result[0]['USER_ID']) if user_result else None
                    except:
                        new_user_id = None
                    
                    # Send SUCCESS event (non-blocking) - Updated for REST API
                    kafka_sent = send_kafka_event(
                        topic_name="sclc_user_authentication", # Ensure this matches your Topic name (underscores vs hyphens)
                        event_type="user_signup_success",
                        button_name="signup_submit",
                        user_id=new_user_id,
                        status="success"
                    )
                    
                    # Show success regardless of Kafka status
                    st.success("✅ " + message)
                    if kafka_sent:
                        st.info("📊 Event tracked successfully")
                    else:
                        st.warning("⚠️ Event tracking failed (signup still successful)")
                    
                    st.balloons()
                    time.sleep(1)
                    st.session_state.show_signup = False
                    st.rerun()
                    
                else:
                    # ❌ SIGNUP FAILURE
                    
                    # Send FAILURE event (non-blocking)
                    kafka_sent = send_kafka_event(
                        topic_name="sclc_user_authentication", # Ensure this matches your Topic name
                        event_type="user_signup_failure",
                        button_name="signup_submit",
                        user_id=None,
                        status="failure"
                    )
                    
                    # Show error regardless of Kafka status
                    st.error(message)
                    if kafka_sent:
                        st.caption("📊 Failure event tracked")
                # --- INDENTATION FIX END ---

    if st.button("← Back to Login", use_container_width=True):
        go_to_login()
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 2rem;'>
        <p><strong>OncoDetect-AI</strong> | Advanced Precision Oncology Platform</p>
        <p>Powered by Snowflake ML + Cortex AI</p>
        <p style="font-size: 0.85rem;">For Research Use Only | Not for Clinical Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 17. UI PAGES - USER PROFILE (MODIFIED WITH NEW BUTTONS)
# ============================================================

def show_user_profile_page():
    render_sidebar_profile()
    st.markdown("<h1 style='text-align: center;'>ONCODETECT AI 🫁🎗️</h1>", unsafe_allow_html=True)
    st.success(f"Welcome, {st.session_state.username}!")
    st.write("---")
    
    # Module selection buttons - 2x2 grid
    st.markdown("### 🔬 Select Module")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔬 SCLC Research Assistant", type="primary", use_container_width=True):
            send_kafka_event("sclc_user_authentication", "button_click", "research_assistant_module", str(st.session_state.user_id), "clicked")
            start_research()
            st.rerun()
        st.caption("Multi-agent research with RAG, Arxiv & Web Search")
    
    with col2:
        if st.button("🎯 Risk Prediction", type="secondary", use_container_width=True):
            send_kafka_event("sclc_user_authentication", "button_click", "risk_prediction_module", str(st.session_state.user_id), "clicked")
            start_risk_prediction()
            st.rerun()
        st.caption("AI-powered survival risk assessment")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🧬 Subtype Classification", type="secondary", use_container_width=True):
            send_kafka_event("sclc_user_authentication", "button_click", "subtype_classification_module", str(st.session_state.user_id), "clicked")
            start_subtype_classification()
            st.rerun()
        st.caption("Molecular subtype prediction (A/N/P/Y)")
    
    st.write("---")
    
    # Feedback section
    st.markdown("### 💬 Share Your Feedback")
    
    if not st.session_state.show_feedback:
        if st.button("✍️ Give Feedback", use_container_width=True):
            toggle_feedback()
            st.rerun()
    else:
        with st.form("feedback_form", clear_on_submit=True):
            st.markdown("**We value your feedback!**")
            
            rating = st.select_slider(
                "Rating ⭐",
                options=[1, 2, 3, 4, 5],
                value=5,
                format_func=lambda x: "⭐" * x
            )
            
            feedback_text = st.text_input(
                "Feedback (max 50 characters)",
                max_chars=50,
                placeholder="Share your thoughts..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submit_feedback = st.form_submit_button("📤 Submit", use_container_width=True)
            with col2:
                if st.form_submit_button("✖️ Cancel", use_container_width=True):
                    st.session_state.show_feedback = False
                    st.rerun()
            
            if submit_feedback:
                if not feedback_text or len(feedback_text.strip()) == 0:
                    st.error("⚠️ Please enter feedback")
                else:
                    success, message = save_user_feedback(
                        st.session_state.user_id,
                        st.session_state.username,
                        feedback_text,
                        rating
                    )
                    if success:
                        send_kafka_event("sclc_user_authentication", "button_click", "feedback_submit", str(st.session_state.user_id), "success")
                        st.success(message)
                        st.balloons()
                        st.session_state.show_feedback = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
    
    st.write("---")
    if st.button("🚪 Logout"):
        logout()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 2rem;'>
        <p><strong>OncoDetect-AI</strong> | Advanced Precision Oncology Platform</p>
        <p>Powered by Snowflake ML + Cortex AI</p>
        <p style="font-size: 0.85rem;">For Research Use Only | Not for Clinical Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

def show_admin_profile_page():
    render_sidebar_profile()
    st.markdown("<h1 style='text-align: center;'>ONCODETECT AI 🫁🎗️</h1>", unsafe_allow_html=True)
    st.success(f"Welcome, Admin {st.session_state.username}!")
    st.write("---")
    
    # Analytics button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 View Product Analytics", type="primary", use_container_width=True):
            send_kafka_event("sclc_user_authentication", "button_click", "admin_view_analytics", str(st.session_state.user_id), "clicked")
            start_analytics()
            st.rerun()
    
    st.write("---")
    if st.button("🚪 Logout"):
        logout()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 2rem;'>
        <p><strong>OncoDetect-AI</strong> | Advanced Precision Oncology Platform</p>
        <p>Powered by Snowflake ML + Cortex AI</p>
        <p style="font-size: 0.85rem;">For Research Use Only | Not for Clinical Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 18. UI PAGES - ANALYTICS
# ============================================================

def show_analytics_page():
    """Display product analytics dashboard for admins"""
    # Assuming render_sidebar_profile() is defined elsewhere in your code based on your snippet
    # If not, you can remove this line or replace it with the sidebar logic we wrote earlier
    if 'render_sidebar_profile' in globals():
        render_sidebar_profile()
    
    st.markdown("<h1 style='text-align: center;'>📊 PRODUCT ANALYTICS DASHBOARD</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size: 16px;'>Get Insights on Your Product! 💡</h3>", unsafe_allow_html=True)
    
    # Check metadata status
    try:
        check_query = "SELECT COUNT(*) as count FROM ONCODETECT_DB.PRODUCT_ANALYTICS.TABLE_METADATA"
        result = session.sql(check_query).collect()
        metadata_count = result[0]['COUNT']
        if metadata_count > 0:
            st.success(f"✅ Analytics ready!")
        else:
            st.warning("⚠️ Metadata not initialized")
    except:
        st.error("❌ Metadata table not found")
    
    # --- LAYOUT FIX START ---
    
    # 1. Info message spans 100% of the width (Outside of columns)
    st.info("💡 Ask questions about your product!")
    
    # 2. Clear history button on a new row, aligned to the right
    if st.session_state.analytics_history:
        col_spacer, col_clear = st.columns([6, 1])
        with col_clear:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.analytics_history = []
                st.rerun()
    
    # --- LAYOUT FIX END ---
    
    st.write("---")
    
# Display analytics history
    for idx, item in enumerate(st.session_state.analytics_history):
        st.markdown(f"<div class='user-message'><strong>You:</strong><br>{item['question']}</div>", unsafe_allow_html=True)
        
        if item['success']:
            st.success(f"✅ Found {len(item['df'])} results!")
            
            # Visualization
            st.markdown("### 📈 Visualization")
            
            # --- UPDATED CALL WITH UNIQUE KEY ---
            chart_created = create_visualization(item['df'], item['question'], f"hist_{idx}")
            
            if not chart_created:
                st.info("📊 View data table below")
            
            # Insights
            if item.get('insights'):
                st.markdown("### 💡 Insights")
                st.info(item['insights'])
            
            # Data table
            st.markdown("### 📋 Data")
            st.dataframe(item['df'], use_container_width=True, height=300)
            
            # Download option
            csv = item['df'].to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"dl_{idx}"
            )
        else:
            st.error(f"❌ {item['error']}")
        
        st.write("---")
    
    # Query input form
    with st.form(key="analytics_form", clear_on_submit=True):
        col1, col2 = st.columns([9, 1])
        with col1:
            user_question = st.text_input(
                "Ask:",
                placeholder="e.g., Show feedback ratings distribution",
                label_visibility="collapsed"
            )
        with col2:
            submit_btn = st.form_submit_button("🔍", use_container_width=True)
    
    if submit_btn and user_question:
        df, sql_query, insights, relevant_tables = execute_analytics_query(user_question)
        
        if df is not None and not df.empty:
            st.session_state.analytics_history.append({
                'question': user_question,
                'df': df,
                'sql_query': sql_query,
                'insights': insights,
                'success': True
            })
        else:
            st.session_state.analytics_history.append({
                'question': user_question,
                'error': relevant_tables if isinstance(relevant_tables, str) else 'No results',
                'success': False
            })
        
        st.rerun()
    
    st.write("---")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Profile", use_container_width=True):
            go_back_to_profile()
            st.rerun()
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 2rem;'>
        <p><strong>OncoDetect-AI</strong> | Advanced Precision Oncology Platform</p>
        <p>Powered by Snowflake ML + Cortex AI</p>
        <p style="font-size: 0.85rem;">For Research Use Only | Not for Clinical Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 19. UI PAGES - RESEARCH (ORIGINAL RESEARCH PAGE)
# ============================================================

def show_research_page():
    """Display SCLC research assistant page with multi-agent system"""
    render_sidebar_profile()
    # ========== SIDEBAR: Research Notes ==========
    with st.sidebar:
        st.markdown("### 📌 Research Notes")
        st.write("")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, key="research_new_chat"):
            start_new_chat()
            st.rerun()
        
        st.write("---")
        
        # Display saved notes (always visible when sidebar is open)
        if st.session_state.research_notes:
            for note in st.session_state.research_notes:
                is_viewing = st.session_state.viewing_note and st.session_state.viewing_note['note_id'] == note['note_id']
                
                if is_viewing:
                    st.markdown(f"""
                    <div style='background-color: #1a1a1a; padding: 0.6rem 0.8rem; 
                                border-radius: 6px; margin: 0.3rem 0; 
                                border-left: 3px solid #4CAF50; color: #FFFFFF;'>
                        📖 <strong>{note['title']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if st.button(f"📄 {note['title']}", key=f"note_{note['note_id']}", use_container_width=True):
                        view_research_note(note)
                        st.rerun()
        else:
            st.markdown("<p style='font-size: 13px; color: #888;'>No saved notes yet</p>", unsafe_allow_html=True)
    
    # ========== MAIN AREA: Chat Interface ==========
    
    # NEW HEADER - Blue background with H2 title
    st.markdown("""
    <div style="background: #081f37; color: white; padding: 30px 20px; text-align: center; 
                margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 10px 10px;">
        <h2 style="color: white !important; margin: 0; font-size: 1.75rem;">SCLC RESEARCH ASSISTANT 📚</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.viewing_note:
        st.info(f"📌 Viewing: {st.session_state.viewing_note['title']}")
    
    st.write("---")
    
    # ========== CHAT HISTORY DISPLAY ==========
    
    for idx, message in enumerate(st.session_state.chat_history):
        
        # USER MESSAGE
        if message['role'] == 'user':
            st.markdown(f"<div class='user-message'><strong>You:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
        
        # ASSISTANT MESSAGE
        else:
            # Display search status (which agents were consulted)
            if 'metadata' in message and 'search_status' in message['metadata']:
                for status_msg in message['metadata']['search_status']:
                    st.info(status_msg)
            
            # Display agent source badge
            if 'metadata' in message:
                agent_used = message['metadata'].get('agent_used', 'RAG')
                has_text = message['metadata'].get('text_sources') and len(message['metadata']['text_sources']) > 0
                has_images = message['metadata'].get('image_sources') and len(message['metadata']['image_sources']) > 0
                
                # Show appropriate badge based on agent
                if agent_used == "RAG":
                    if has_text and has_images:
                        st.markdown("<div class='source-badge'>💡 Response from RAG Agent: In-house database (Text + Images)</div>", unsafe_allow_html=True)
                    elif has_text:
                        st.markdown("<div class='source-badge'>💡 Response from RAG Agent: In-house database (Text)</div>", unsafe_allow_html=True)
                    elif has_images:
                        st.markdown("<div class='image-source-badge'>🖼️ Response from RAG Agent: In-house image database</div>", unsafe_allow_html=True)
                
                elif agent_used == "Arxiv + Web":
                    st.markdown("<div class='source-badge' style='background-color: #FFF3E0; border-left: 4px solid #FF9800;'>🌐 Response from Multi-Agent System: Arxiv + Web (External sources)</div>", unsafe_allow_html=True)
                
                elif agent_used == "RAG + Arxiv":
                    st.markdown("<div class='source-badge' style='background-color: #F3E8FD; border-left: 4px solid #9C27B0;'>🔄 Response from Multi-Agent System: RAG + Arxiv (Blended)</div>", unsafe_allow_html=True)
                
                elif agent_used == "Web":
                    st.markdown("<div class='source-badge' style='background-color: #E3F2FD; border-left: 4px solid #2196F3;'>🌐 Response from Web Search Agent: Current web information</div>", unsafe_allow_html=True)
                
                elif agent_used == "RAG + Arxiv + Web":
                    st.markdown("<div class='source-badge' style='background-color: #E8F5E9; border-left: 4px solid #4CAF50;'>🌟 Response from Complete Multi-Agent System: RAG + Arxiv + Web (Comprehensive)</div>", unsafe_allow_html=True)
            
            # Display bot response
            st.markdown(f"<div class='bot-message'><strong>OncoDetect AI:</strong><br>{message['content']}</div>", unsafe_allow_html=True)
            
            # Display agent summary
            if 'metadata' in message and 'combined_confidence' in message['metadata']:
                agent = message['metadata'].get('agent_used', 'RAG')
                conf_msg = f"✅ **Agent Used:** {agent}"
                st.success(conf_msg)
            
            # ========== VIEW SOURCES SECTION ==========
            
            if 'metadata' in message:
                source_key = f"src_{idx}"
                if source_key not in st.session_state.show_sources:
                    st.session_state.show_sources[source_key] = False
                
                agent_used = message['metadata'].get('agent_used', 'RAG')
                has_rag = (message['metadata'].get('text_sources') or message['metadata'].get('image_sources'))
                
                # Show sources button if any sources exist
                if has_rag or agent_used in ['Arxiv', 'RAG + Arxiv', 'RAG + Arxiv + Web', 'Arxiv + Web', 'Web']:
                    if st.button(f"{'▼' if st.session_state.show_sources[source_key] else '▶'} View Sources!", 
                               key=f"tog_{idx}", use_container_width=False):
                        st.session_state.show_sources[source_key] = not st.session_state.show_sources[source_key]
                        st.rerun()
                    
                    if st.session_state.show_sources[source_key]:
                        st.markdown("---")
                        
                        # RAG Text Sources
                        if agent_used in ['RAG', 'RAG + Arxiv', 'RAG + Arxiv + Web']:
                            if message['metadata'].get('text_sources'):
                                st.markdown(f"**📚 RAG Text Sources ({len(message['metadata']['text_sources'])}):**")
                                for i, s in enumerate(message['metadata']['text_sources'], 1):
                                    rel = s.get('relevance', 0)
                                    st.markdown(f"{i}. Paper: **{s['paper_id']}** | Pages: **{s['pages']}** | Relevance: **{rel:.1%}**")
                                st.write("")
                            
                            # RAG Image Sources
                            if message['metadata'].get('image_sources'):
                                st.markdown(f"**🖼️ RAG Image Sources ({len(message['metadata']['image_sources'])}):**")
                                for i, s in enumerate(message['metadata']['image_sources'], 1):
                                    rel = s.get('relevance', 0)
                                    img_info = f" | {s['format']} ({s['width']}x{s['height']})" if 'format' in s else ""
                                    st.markdown(f"{i}. Paper: **{s['paper_id']}** | Page: **{s['page']}** | Image: **{s['image_index']}**{img_info} | Relevance: **{rel:.1%}**")
                                st.write("")
                        
                        # Arxiv Sources
                        if agent_used in ['RAG + Arxiv', 'Arxiv + Web', 'RAG + Arxiv + Web']:
                            papers = message['metadata'].get('arxiv_papers', [])
                            if papers:
                                st.markdown(f"**📚 Arxiv Research Papers ({len(papers)}):**")
                                for i, p in enumerate(papers, 1):
                                    st.markdown(f"{i}. **{p['title']}**")
                                    st.markdown(f"   Authors: {p['authors']} | Published: {p['published']}")
                                    st.markdown(f"   [🔗 View on Arxiv]({p['url']})")
                                st.write("")
                        
                        # Web Sources
                        if agent_used in ['Web', 'Arxiv + Web', 'RAG + Arxiv + Web']:
                            sources = message['metadata'].get('web_sources', [])
                            if sources:
                                st.markdown(f"**🌐 Web Search Results ({len(sources)}):**")
                                for i, s in enumerate(sources, 1):
                                    st.markdown(f"{i}. **{s['title']}**")
                                    st.markdown(f"   [🔗 {s['url']}]({s['url']})")
                                st.write("")
                        
                        st.markdown(f"<p style='font-size: 13px; color: #888;'>⏱️ {message['metadata']['response_time_ms']}ms</p>", unsafe_allow_html=True)
                        st.markdown("---")
            
            # ========== VIEW INDIVIDUAL AGENT RESPONSES ==========
            
            if 'metadata' in message and message['metadata'].get('agent_used') in ['RAG + Arxiv', 'RAG + Arxiv + Web', 'Arxiv + Web']:
                ind_key = f"ind_{idx}"
                if ind_key not in st.session_state.show_sources:
                    st.session_state.show_sources[ind_key] = False
                
                # Main toggle button with proper arrow
                arrow_icon = "▲" if st.session_state.show_sources[ind_key] else "▼"
                if st.button(f"{arrow_icon} View Individual Agent Responses", 
                           key=f"tog_ind_{idx}", use_container_width=False):
                    st.session_state.show_sources[ind_key] = not st.session_state.show_sources[ind_key]
                    st.rerun()
                
                if st.session_state.show_sources[ind_key]:
                    st.markdown("---")
                    
                    # Initialize individual card states
                    rag_card_key = f"rag_card_{idx}"
                    arxiv_card_key = f"arxiv_card_{idx}"
                    web_card_key = f"web_card_{idx}"
                    
                    if rag_card_key not in st.session_state.show_sources:
                        st.session_state.show_sources[rag_card_key] = True
                    if arxiv_card_key not in st.session_state.show_sources:
                        st.session_state.show_sources[arxiv_card_key] = True
                    if web_card_key not in st.session_state.show_sources:
                        st.session_state.show_sources[web_card_key] = True
                    
                    # RAG Response Card
                    if message['metadata'].get('rag_response'):
                        rag_arrow = "▲" if st.session_state.show_sources[rag_card_key] else "▼"
                        col_rag_btn, col_rag_label = st.columns([0.08, 0.92])
                        with col_rag_btn:
                            if st.button(rag_arrow, key=f"rag_toggle_{idx}", help="Expand/Collapse"):
                                st.session_state.show_sources[rag_card_key] = not st.session_state.show_sources[rag_card_key]
                                st.rerun()
                        with col_rag_label:
                            st.markdown("**🏠 RAG Agent (In-House Database)**")
                        
                        if st.session_state.show_sources[rag_card_key]:
                            st.markdown(f"""
                            <div style='background-color: #F5F5F5; padding: 1rem; border-radius: 8px; 
                                        font-size: 13px; margin-left: 2rem; margin-bottom: 1rem;
                                        border-left: 3px solid #666;'>
                                {message['metadata']['rag_response']}
                            </div>
                            """, unsafe_allow_html=True)
                        st.write("")
                    
                    # Arxiv Response Card
                    if message['metadata'].get('arxiv_response'):
                        arxiv_arrow = "▲" if st.session_state.show_sources[arxiv_card_key] else "▼"
                        col_arxiv_btn, col_arxiv_label = st.columns([0.08, 0.92])
                        with col_arxiv_btn:
                            if st.button(arxiv_arrow, key=f"arxiv_toggle_{idx}", help="Expand/Collapse"):
                                st.session_state.show_sources[arxiv_card_key] = not st.session_state.show_sources[arxiv_card_key]
                                st.rerun()
                        with col_arxiv_label:
                            st.markdown("**📚 Arxiv Agent (Research Papers)**")
                        
                        if st.session_state.show_sources[arxiv_card_key]:
                            arxiv_text = message['metadata']['arxiv_response']
                            if isinstance(arxiv_text, tuple):
                                arxiv_text = arxiv_text[0]
                            st.markdown(f"""
                            <div style='background-color: #E8F4FD; padding: 1rem; border-radius: 8px; 
                                        font-size: 13px; margin-left: 2rem; margin-bottom: 1rem;
                                        border-left: 3px solid #2196F3;'>
                                {arxiv_text}
                            </div>
                            """, unsafe_allow_html=True)
                        st.write("")
                    
                    # Web Response Card
                    if message['metadata'].get('web_response'):
                        web_arrow = "▲" if st.session_state.show_sources[web_card_key] else "▼"
                        col_web_btn, col_web_label = st.columns([0.08, 0.92])
                        with col_web_btn:
                            if st.button(web_arrow, key=f"web_toggle_{idx}", help="Expand/Collapse"):
                                st.session_state.show_sources[web_card_key] = not st.session_state.show_sources[web_card_key]
                                st.rerun()
                        with col_web_label:
                            st.markdown("**🌐 Web Search Agent (Current Web Info)**")
                        
                        if st.session_state.show_sources[web_card_key]:
                            web_text = message['metadata']['web_response']
                            if isinstance(web_text, tuple):
                                web_text = web_text[0]
                            st.markdown(f"""
                            <div style='background-color: #E3F2FD; padding: 1rem; border-radius: 8px; 
                                        font-size: 13px; margin-left: 2rem; margin-bottom: 1rem;
                                        border-left: 3px solid #1976D2;'>
                                {web_text}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
            
            # ========== TRANSLATION & SUMMARY ==========
            
            st.write("")
            col_lang, col_empty, col_translate, col_summarize = st.columns([2, 1, 2.5, 2.5])
            
            with col_lang:
                langs = {"Spanish": "es", "French": "fr", "German": "de", "Italian": "it", "Portuguese": "pt", "Chinese": "zh"}
                selected_language = st.selectbox("lang", options=list(langs.keys()), key=f"lang_{idx}", label_visibility="collapsed")
            
            with col_translate:
                if st.button("🌐", key=f"trans_{idx}", use_container_width=True, help="Translate"):
                    send_kafka_event("sclc_user_authentication", "button_click", "translate_message", str(st.session_state.user_id), f"target_{selected_language}")
                    with st.spinner("Translating..."):
                        translated = translate_text(message['content'], langs[selected_language])
                        if translated:
                            message['translated'] = {'text': translated, 'language': selected_language}
                            st.rerun()
            
            with col_summarize:
                if st.button("📝", key=f"sum_{idx}", use_container_width=True, help="Summarize"):
                    send_kafka_event("sclc_user_authentication", "button_click", "summarize_message", str(st.session_state.user_id), "clicked")
                    with st.spinner("Summarizing..."):
                        summary = summarize_text(message['content'])
                        if summary:
                            message['summary'] = summary
                            st.rerun()
            
            # Display translation
            if 'translated' in message:
                st.markdown(f"<div class='translated-text'><strong>🌐 Translation ({message['translated']['language']}):</strong><br>{message['translated']['text']}</div>", unsafe_allow_html=True)
            
            # Display summary
            if 'summary' in message:
                st.markdown(f"<div class='summary-text'><strong>📝 Summary:</strong><br>{message['summary']}</div>", unsafe_allow_html=True)
            
            st.write("---")
    
    # ========== SAVE RESEARCH NOTE ==========
    
    if len(st.session_state.chat_history) > 0 and not st.session_state.viewing_note:
        st.write("---")
        col_left, col_right = st.columns([5, 2])
        with col_right:
            with st.form("save_note_form"):
                note_title = st.text_input("Title", placeholder="Research note title", label_visibility="collapsed")
                
                if st.form_submit_button("💾 Save as Research Note", use_container_width=True):
                    if note_title:
                        send_kafka_event("sclc_user_authentication", "button_click", "save_research_note", str(st.session_state.user_id), "success")
                        if save_as_research_note(st.session_state.user_id, st.session_state.username, st.session_state.chat_history, note_title):
                            st.success("✅ Saved!")
                            load_research_notes(st.session_state.user_id)
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.error("Enter title")
        st.write("---")
    
    # ========== CHAT INPUT ==========
    
    if not st.session_state.viewing_note:
        with st.form(key="query_form", clear_on_submit=True):
            col1, col2 = st.columns([9, 1])
            with col1:
                user_query = st.text_input("Ask about SCLC:", placeholder="e.g., Current treatments?", 
                                          label_visibility="collapsed")
            with col2:
                submit_button = st.form_submit_button("➤", use_container_width=True)
        
        if submit_button and user_query:
            send_kafka_event("sclc_user_authentication", "button_click", "search_query_submit", str(st.session_state.user_id), "submitted")
            # Add user message to chat
            st.session_state.chat_history.append({'role': 'user', 'content': user_query})
            
            # Process query with multi-agent system
            (response, text_results, image_results, response_time_ms, has_sources, 
             search_status, rag_confidence, combined_confidence, agent_used, rag_response, 
             arxiv_response, arxiv_papers, web_response, web_sources) = process_user_query(
                user_query, st.session_state.user_id, st.session_state.username
            )
            
            # Display search status
            for status_msg in search_status:
                st.info(status_msg)
            
            has_text = len(text_results) > 0
            has_images = len(image_results) > 0
            
            # Display agent source badge
            if agent_used == "RAG":
                if has_text and has_images:
                    st.markdown("<div class='source-badge'>💡 Response from RAG Agent: In-house database (Text + Images)</div>", unsafe_allow_html=True)
                elif has_text:
                    st.markdown("<div class='source-badge'>💡 Response from RAG Agent: In-house database (Text)</div>", unsafe_allow_html=True)
                elif has_images:
                    st.markdown("<div class='image-source-badge'>🖼️ Response from RAG Agent: In-house image database</div>", unsafe_allow_html=True)
            
            elif agent_used == "Arxiv + Web":
                st.markdown("<div class='source-badge' style='background-color: #FFF3E0; border-left: 4px solid #FF9800;'>🌐 Response from Multi-Agent System: Arxiv + Web (External sources)</div>", unsafe_allow_html=True)
            
            elif agent_used == "RAG + Arxiv":
                st.markdown("<div class='source-badge' style='background-color: #F3E8FD; border-left: 4px solid #9C27B0;'>🔄 Response from Multi-Agent System: RAG + Arxiv (Blended)</div>", unsafe_allow_html=True)
            
            elif agent_used == "Web":
                st.markdown("<div class='source-badge' style='background-color: #E3F2FD; border-left: 4px solid #2196F3;'>🌐 Response from Web Search Agent: Current web information</div>", unsafe_allow_html=True)
            
            elif agent_used == "RAG + Arxiv + Web":
                st.markdown("<div class='source-badge' style='background-color: #E8F5E9; border-left: 4px solid #4CAF50;'>🌟 Response from Complete Multi-Agent System: RAG + Arxiv + Web (Comprehensive)</div>", unsafe_allow_html=True)
            
            # Display response with typing effect
            typing_placeholder = st.empty()
            display_typing_effect(response, typing_placeholder)
            
            # Build source summary
            source_summary = []
            
            if agent_used in ["RAG", "RAG + Arxiv", "RAG + Arxiv + Web"]:
                rag_parts = []
                if len(text_results) > 0:
                    rag_parts.append(f"{len(text_results)} text")
                if len(image_results) > 0:
                    rag_parts.append(f"{len(image_results)} image")
                if rag_parts:
                    source_summary.append(f"RAG ({', '.join(rag_parts)})")
            
            if agent_used in ["RAG + Arxiv", "Arxiv + Web", "RAG + Arxiv + Web"]:
                if arxiv_papers:
                    source_summary.append(f"Arxiv ({len(arxiv_papers)} papers)")
                else:
                    source_summary.append("Arxiv")
            
            if agent_used in ["Web", "Arxiv + Web", "RAG + Arxiv + Web"]:
                if web_sources:
                    source_summary.append(f"Web ({len(web_sources)} results)")
                else:
                    source_summary.append("Web")
            
            summary_text = " + ".join(source_summary) if source_summary else "Internal database"
            
            # Display agent summary
            response_summary_msg = f"✅ **Agent:** {agent_used} | **Sources:** {summary_text}"
            st.success(response_summary_msg)
            
            # Build metadata for storage
            metadata = {
                'text_sources': [
                    {'paper_id': r['PAPER_ID'], 'pages': f"{r['START_PAGE']}-{r['END_PAGE']}", 
                     'relevance': float(r['SIMILARITY_SCORE'])}
                    for r in text_results
                ],
                'image_sources': [
                    {'paper_id': r['PAPER_ID'], 'page': r['PAGE_NUMBER'],
                     'image_index': r['IMAGE_INDEX'], 'format': r['FORMAT'],
                     'width': r['WIDTH'], 'height': r['HEIGHT'],
                     'relevance': float(r['SIMILARITY_SCORE'])}
                    for r in image_results
                ],
                'response_time_ms': response_time_ms,
                'has_sources': has_sources,
                'search_status': search_status,
                'response_summary': response_summary_msg,
                'rag_confidence': rag_confidence,
                'combined_confidence': combined_confidence,
                'agent_used': agent_used,
                'rag_response': rag_response if agent_used in ["RAG + Arxiv", "RAG + Arxiv + Web"] else None,
                'arxiv_response': arxiv_response,
                'arxiv_papers': arxiv_papers,
                'web_response': web_response,
                'web_sources': web_sources
            }
            
            # Add to chat history
            st.session_state.chat_history.append({'role': 'assistant', 'content': response, 'metadata': metadata})
            time.sleep(0.5)
            st.rerun()
    else:
        st.info("💡 Viewing saved note. Click 'New Chat' to start new conversation.")
    
    st.write("---")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Profile", use_container_width=True):
            go_back_to_profile()
            st.rerun()
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 2rem;'>
        <p><strong>OncoDetect-AI</strong> | Advanced Precision Oncology Platform</p>
        <p>Powered by Snowflake ML + Cortex AI</p>
        <p style="font-size: 0.85rem;">For Research Use Only | Not for Clinical Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 20. UI PAGES - RISK PREDICTION
# ============================================================

def show_risk_prediction_page():
    render_sidebar_profile()
    
    # Custom CSS matching subtype page
    st.markdown("""
    <style>
    /* Page header styling */
    .page-header {
        background: #081f37;
        color: white;
        padding: 30px 20px;
        text-align: center;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .page-header h1 {
        color: white !important;
        margin-bottom: 10px;
    }
    .page-header p {
        color: white !important;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Main content container - 60% width centered */
    .main-content {
        max-width: 60%;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Input labels - black */
    .stNumberInput label,
    .stSelectbox label {
        color: black !important;
    }
    
    /* PRIMARY Button (Quick Prediction) - NOW DARK NAVY #081f37 */
    .stButton > button[kind="primary"] {
        background-color: #081f37 !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0a2847 !important;
    }

    /* LINK BUTTONS (View Map) - NOW DARK NAVY #081f37 */
    /* This targets the <a> tag generated by st.link_button */
    .stLinkButton > a[kind="primary"] {
        background-color: #081f37 !important;
        color: white !important;
        border: none !important;
    }
    .stLinkButton > a[kind="primary"]:hover {
        background-color: #0a2847 !important;
    }
    
    /* SECONDARY Button (Full AI Analysis) - Dark #081f37 */
    .stButton > button[kind="secondary"] {
        background-color: #081f37 !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #0a2847 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #081f37 !important;
        color: white !important;
    }
    
    /* Risk card - Base styling (color set dynamically in python) */
    .risk-card {
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        transition: background 0.3s ease;
    }
    
    /* Metric cards - light grey */
    .metric-card {
        background: #e8e8e8;
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: black;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: black;
        margin-top: 5px;
    }
    
    /* Analysis cards - ALL GREY like subtype */
    .analysis-card {
        background: #e8e8e8;
        color: black;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .analysis-card h3 {
        color: black;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 1.25rem;
    }
    .analysis-card p {
        color: black;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Section headings */
    .section-h2 {
        color: black;
        font-size: 1.75rem;
        font-weight: bold;
        margin: 20px 0 15px 0;
    }
    .section-h3 {
        color: black;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 15px 0 10px 0;
    }
    
    /* TMB badges */
    .tmb-badge {
        padding: 10px 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        font-weight: 600;
    }
    .tmb-high {
        background: #ffebee;
        color: #c62828;
    }
    .tmb-moderate {
        background: #fff3e0;
        color: #ef6c00;
    }
    .tmb-low {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .results-section {
        text-align: center;
    }
    .results-section .analysis-card {
        text-align: left;
    }
    
    /* Navigation Buttons */
    .nav-button button {
        background-color: black !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # PAGE HEADER
    st.markdown("""
    <div class="page-header">
        <h1>🎯 SCLC RISK PREDICTION</h1>
        <p>Advanced survival risk assessment using clinical features and AI analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'risk_result' not in st.session_state:
        st.session_state.risk_result = None
    
    # START MAIN CONTENT CONTAINER (60% WIDTH)
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    
    # 1. Select Profile (Moved out of expander)
    st.markdown("<h2 class='section-h2'>📝 Patient Information</h2>", unsafe_allow_html=True)
    
    preset = st.selectbox("Select Test Profile",
        ["Custom Input", "78-year-old Male (Very High Risk)", 
         "65-year-old Female (Moderate Risk)", "55-year-old Male (Low Risk)"])
    
    if preset == "78-year-old Male (Very High Risk)":
        age, is_male_idx, smoker_idx, mutations, tmb = 78, 0, 0, 580, 21.5
    elif preset == "65-year-old Female (Moderate Risk)":
        age, is_male_idx, smoker_idx, mutations, tmb = 65, 1, 1, 200, 12.5
    elif preset == "55-year-old Male (Low Risk)":
        age, is_male_idx, smoker_idx, mutations, tmb = 55, 0, 0, 120, 4.5
    else:
        age, is_male_idx, smoker_idx, mutations, tmb = 65, 0, 0, 180, 7.0
    
    st.markdown("<h3 class='section-h3'>Demographics</h3>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age (years)", 18, 100, int(age))
    with col_b:
        is_male = st.selectbox("Sex", ["Male", "Female"], index=is_male_idx)
    
    st.markdown("<h3 class='section-h3'>Clinical History</h3>", unsafe_allow_html=True)
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"], index=smoker_idx)
    
    st.markdown("<h3 class='section-h3'>Genomic Features</h3>", unsafe_allow_html=True)
    mutation_count = st.number_input("Mutation Count", 0, 2000, int(mutations))
    tmb = st.number_input("TMB (mut/Mb)", 0.0, 100.0, float(tmb), 0.1)
    
    if tmb >= 20:
        st.markdown('<div class="tmb-badge tmb-high">🔴 <strong>High TMB</strong></div>', unsafe_allow_html=True)
    elif tmb >= 10:
        st.markdown('<div class="tmb-badge tmb-moderate">🟡 <strong>Intermediate TMB</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tmb-badge tmb-low">🟢 <strong>Low TMB</strong></div>', unsafe_allow_html=True)
    
    is_tmb_high = 1 if tmb >= 20 else 0
    is_tmb_intermediate = 1 if 10 <= tmb < 20 else 0
    is_tmb_low = 1 if tmb < 10 else 0
    is_male_val = 1 if is_male == "Male" else 0
    is_former_smoker = 1 if smoking == "Former" else 0
    smoker_x_tmb = tmb if smoking == "Former" else 0.0
    
    st.markdown("---")
    
    st.markdown("<h3 class='section-h3'>🔬 Analysis Type</h3>", unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        quick_predict = st.button("⚡ Predict Risk Score", type="primary", use_container_width=True)
    with col_btn2:
        full_analysis = st.button("🤖 Full Analysis", type="secondary", use_container_width=True)
    
    # Helper to determine card color
    def get_risk_color(risk_level_str):
        lvl = risk_level_str.upper()
        if "HIGH" in lvl or "VERY" in lvl:
            return "#d32f2f" # Red
        elif "MODERATE" in lvl or "INTERMEDIATE" in lvl:
            return "#f57c00" # Orange/Yellow
        else:
            return "#388e3c" # Green

    # QUICK PREDICTION
    if quick_predict:
        send_kafka_event("sclc_user_authentication", "button_click", "risk_quick_prediction", str(st.session_state.user_id), "clicked")
        with st.spinner("⚡ Running..."):
            query = f"""
            SELECT PREDICT_RISK_SCORE(
                {age}, {is_male_val}, {is_former_smoker}, {mutation_count},
                {tmb}, {is_tmb_high}, {is_tmb_intermediate}, {is_tmb_low}, {smoker_x_tmb}
            ) AS prediction
            """
            result = session.sql(query).collect()
            
            if result:
                pred = json.loads(result[0]['PREDICTION']) if isinstance(result[0]['PREDICTION'], str) else result[0]['PREDICTION']
                
                st.session_state.risk_result = {
                    'risk_score': pred['risk_score'],
                    'risk_level': pred['risk_level'],
                    'confidence': pred['confidence'],
                    'type': 'quick'
                }
                
                st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                st.markdown("<h2 class='section-h2'>📊 Results</h2>", unsafe_allow_html=True)
                
                risk_emoji = "⚠️" if "HIGH" in pred['risk_level'] else "✅"
                
                # Dynamic Background Color
                bg_color = get_risk_color(pred['risk_level'])
                
                st.markdown(f"""
                <div class='risk-card' style='background-color: {bg_color};'>
                    <div style='font-size: 3rem;'>{risk_emoji}</div>
                    <h2 style='margin: 10px 0; color: white;'>{pred['risk_level'].replace('_', ' ')}</h2>
                    <p style='font-size: 1.2rem; color: white;'>Risk Score: {pred['risk_score']:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{pred['risk_score']:.1f}/100</div>
                        <div class='metric-label'>Risk Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{pred['confidence']*100:.1f}%</div>
                        <div class='metric-label'>Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("✅ Complete!")
                st.markdown("</div>", unsafe_allow_html=True)
    
    # FULL ANALYSIS
    if full_analysis:
        send_kafka_event("sclc_user_authentication", "button_click", "risk_full_analysis", str(st.session_state.user_id), "clicked")
        st.markdown("---")
        st.markdown("<h2 class='section-h2'>🔍 Comprehensive Analysis</h2>", unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🧬 Initializing...")
        progress_bar.progress(10)
        
        with st.spinner("Running..."):
            status_text.text("🔬 ML prediction...")
            progress_bar.progress(25)
            
            query = f"""
            CALL RUN_FULL_RISK_ANALYSIS_WITH_RAG(
                {age}, {is_male_val}, {is_former_smoker}, {mutation_count},
                {tmb}, {is_tmb_high}, {is_tmb_intermediate}, {is_tmb_low}, {smoker_x_tmb}
            )
            """
            
            status_text.text("📚 Analyzing cohort...")
            progress_bar.progress(50)
            result = session.sql(query).collect()
            
            status_text.text("🤖 Generating...")
            progress_bar.progress(85)
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            if result:
                raw_result = result[0]['RUN_FULL_RISK_ANALYSIS_WITH_RAG']
                full_result = json.loads(raw_result)
                if isinstance(full_result, str):
                    full_result = json.loads(full_result)
                
                ml_pred = full_result['ml_prediction']
                similar_patients = full_result['similar_patients']
                clinical_analysis = full_result['clinical_analysis']
                timestamp = full_result['timestamp']
                
                st.session_state.risk_result = {
                    'risk_score': ml_pred['risk_score'],
                    'risk_level': ml_pred['risk_level'],
                    'confidence': ml_pred['confidence'],
                    'type': 'full'
                }
                
                st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                
                # Dynamic Background Color
                bg_color = get_risk_color(ml_pred['risk_level'])
                
                # Risk card
                risk_emoji = "⚠️" if "HIGH" in ml_pred['risk_level'] else "✅"
                st.markdown(f"""
                <div class='risk-card' style='background-color: {bg_color};'>
                    <div style='font-size: 3rem;'>{risk_emoji}</div>
                    <h2 style='margin: 10px 0; color: white;'>{ml_pred['risk_level'].replace('_', ' ')}</h2>
                    <p style='font-size: 1.2rem; color: white;'>Risk Score: {ml_pred['risk_score']:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{ml_pred['risk_score']:.1f}/100</div>
                        <div class='metric-label'>Risk Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{ml_pred['confidence']*100:.1f}%</div>
                        <div class='metric-label'>Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m3:
                    st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-value'>✓</div>
                        <div class='metric-label'>Complete</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # COHORT STATISTICS ONLY
                st.markdown("<h3 class='section-h3'>👥 Patient Cohort Statistics</h3>", unsafe_allow_html=True)
                similar_count = similar_patients.get('similar_patients_found', 0)
                
                if similar_count > 0 and 'patients' in similar_patients:
                    patients_list = similar_patients['patients']
                    avg_age = sum(p['age'] for p in patients_list) / len(patients_list)
                    avg_tmb = sum(p['tmb'] for p in patients_list) / len(patients_list)
                    avg_survival = sum(p['survival_months'] for p in patients_list) / len(patients_list)
                    min_survival = min(p['survival_months'] for p in patients_list)
                    max_survival = max(p['survival_months'] for p in patients_list)
                    
                    st.markdown(f"""
                    <div class='analysis-card'>
                        <h3>📊 Similar Patient Cohort (n={similar_count})</h3>
                        <p style='margin: 8px 0;'><strong>Average Age:</strong> {avg_age:.1f} years</p>
                        <p style='margin: 8px 0;'><strong>Average TMB:</strong> {avg_tmb:.1f} mut/Mb</p>
                        <p style='margin: 8px 0;'><strong>Average Survival:</strong> {avg_survival:.1f} months</p>
                        <p style='margin: 8px 0;'><strong>Survival Range:</strong> {min_survival:.1f} - {max_survival:.1f} months</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='analysis-card'>
                        <h3>👥 Patient Cohort</h3>
                        <p>{similar_patients.get('message', 'No similar patients')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # AI Clinical Assessment - Parse ONCE and show each section
                st.markdown("<h3 class='section-h3'>📋 AI Clinical Assessment</h3>", unsafe_allow_html=True)
                st.caption("*Generated by Llama 3.1 405B with Research Evidence*")
                
                # Clean markdown
                clinical_clean = clinical_analysis.replace('**', '').replace('*', '')
                
                # Split by numbered sections (1. 2. 3. etc) OR by section headers
                
                # Try to split by numbered sections first
                numbered_sections = re.split(r'\n\d+\.\s*', clinical_clean)
                
                if len(numbered_sections) > 1:
                    # Has numbered sections
                    for idx, section in enumerate(numbered_sections[1:], 1):  # Skip first empty
                        section = section.strip()
                        if len(section) > 20:  # Only show substantial content
                            # Extract section name and content
                            lines = section.split('\n', 1)
                            section_header = lines[0].strip() if lines else f"Section {idx}"
                            section_content = lines[1].strip() if len(lines) > 1 else section
                            
                            # Determine icon
                            icon = "⚠️" if "RISK" in section_header.upper() else \
                                   "👥" if "PATIENT" in section_header.upper() or "SIMILAR" in section_header.upper() else \
                                   "🔬" if "EVIDENCE" in section_header.upper() or "ANALYSIS" in section_header.upper() else \
                                   "💊" if "TREATMENT" in section_header.upper() else \
                                   "📊" if "PROGNOSIS" in section_header.upper() else \
                                   "📚" if "REFERENCE" in section_header.upper() else "📋"
                            
                            st.markdown(f"""
                            <div class='analysis-card'>
                                <h3>{icon} {section_header.upper()}</h3>
                                <p>{section_content}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # No numbered sections, show full text
                    st.markdown(f"""
                    <div class='analysis-card'>
                        <h3>📋 Clinical Analysis</h3>
                        <p style='white-space: pre-wrap;'>{clinical_clean}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✅ Complete! (Generated: {timestamp})")
                
                # Download
                st.download_button(
                    "📥 Download Risk Report",
                    f"Risk: {ml_pred['risk_score']:.1f}/100\n\n{clinical_clean}",
                    f"risk_report.txt",
                    use_container_width=True
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # HEALTHCARE FINDER
    if st.session_state.risk_result and "HIGH" in st.session_state.risk_result.get('risk_level', ''):
        st.markdown("---")
        st.markdown("<h3 class='section-h3'>🏥 Nearby Cancer Treatment Centers</h3>", unsafe_allow_html=True)
        
        with st.form(key="healthcare_form"):
            location_input = st.text_input("Enter location", "Boston, MA")
            find_centers = st.form_submit_button("🗺️ Find Centers", use_container_width=True)
        
        if find_centers and location_input:
            send_kafka_event("sclc_user_authentication", "button_click", "find_centers", str(st.session_state.user_id), f"search_{location_input}")
            parts = location_input.split(',')
            city = parts[0].strip()
            state_val = parts[1].strip() if len(parts) > 1 else "MA"
            
            try:
                center_query = f"""
                    SELECT FIND_CANCER_CENTERS_WITH_LOCATIONS(
                        '{city}', '{state_val}', '{st.session_state.risk_result["risk_level"]}'
                    ) AS cancer_centers
                """
                center_result = session.sql(center_query).collect()
                
                if center_result:
                    centers_info = center_result[0]['CANCER_CENTERS']
                    
                    pattern = r'\*\*(\d+)\.\s*([^\*]+?)\*\*'
                    matches = list(re.finditer(pattern, centers_info))
                    
                    for i, match in enumerate(matches[:5], 1):
                        center_name = match.group(2).strip()
                        start = match.end()
                        end = matches[i].start() if i < len(matches) else len(centers_info)
                        content = centers_info[start:end].strip()
                        content = '<br>'.join([l.strip() for l in content.split('\n') if l.strip()])
                        
                        maps_url = f"https://www.google.com/maps/search/?api=1&query={center_name.replace(' ', '+')}+{city}+{state_val}"
                        
                        st.markdown(f"""
                        <div class='analysis-card'>
                            <h3>🏥 {i}. {center_name}</h3>
                            <p>{content}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # View Map button
                        st.link_button(
                            f"🗺️ View {center_name} on Map",
                            maps_url,
                            use_container_width=True,
                            type="primary"
                        )
                    
                    st.success(f"✅ Found {len(matches[:5])} centers!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ============================================
    # NAVIGATION BUTTONS - Black color
    # ============================================
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='nav-button'>", unsafe_allow_html=True)
        if st.button("← Back to Profile", use_container_width=True, key="subtype_back"):
            go_back_to_profile()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='nav-button'>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True, key="subtype_logout"):
            logout()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # END MAIN CONTENT
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>OncoDetect-AI</strong> | Precision Oncology Platform</p>
        <p style='font-size: 0.85rem;'>Powered by Snowflake ML + Cortex AI</p>
        <p style='font-size: 0.8rem;'>For Research Use Only</p>
    </div>
    """, unsafe_allow_html=True)
    
    
# ============================================================
# 21. UI PAGES - SUBTYPE CLASSIFICATION (NEW)
# ============================================================


def show_subtype_classification_page():
    render_sidebar_profile()
    
    # Custom CSS for this page
    st.markdown("""
    <style>
    /* Page header styling - DARK NAVY */
    .page-header {
        background: #081f37;
        color: white;
        padding: 30px 20px;
        text-align: center;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .page-header h1 {
        color: white !important;
        margin-bottom: 10px;
    }
    .page-header p {
        color: white !important;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Main content container - 60% Width */
    .main-content {
        max-width: 60%;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Slider styling */
    .stSlider label { color: black !important; }
    .stSlider > div > div > div { color: black !important; }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] { color: black !important; }
    .stSlider > div > div > div > div { background-color: #081f37 !important; }
    .stSlider > div > div > div > div > div { background-color: #081f37 !important; }
    .stSlider [data-testid="stThumbValue"] { color: #081f37 !important; }
    
    /* Input Labels */
    .stNumberInput label, .stSelectbox label { color: black !important; }
    
    /* --- BUTTON STYLING FIX --- */
    
    /* 1. FORCE ALL BUTTONS TO DARK NAVY DEFAULT */
    /* This targets Classify, Full Analysis, and Download buttons */
    .stButton > button {
        background-color: #081f37 !important;
        color: white !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0a2847 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* 2. OVERRIDE FOR NAVIGATION BUTTONS (BLACK) */
    /* We wrap these in a div class='nav-button' in Python */
    .nav-button .stButton > button {
        background-color: #000000 !important;
        color: white !important;
    }
    .nav-button .stButton > button:hover {
        background-color: #333333 !important;
        color: white !important;
    }
    
    /* Download button specific */
    .stDownloadButton > button {
        background-color: #081f37 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Subtype result cards */
    .subtype-card {
        background: #081f37;
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: #e8e8e8;
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: black;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: black;
        margin-top: 5px;
    }
    
    /* Metric cards equal size */
    .metric-card-equal {
        background: #e8e8e8;
        color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-card-equal .metric-value {
        font-size: 1.75rem;
        font-weight: bold;
        color: black;
    }
    .metric-card-equal .metric-label {
        font-size: 0.85rem;
        color: black;
        margin-top: 5px;
    }
    
    /* Analysis section cards */
    .analysis-card {
        background: #e8e8e8;
        color: black;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .analysis-card h3 {
        color: black;
        margin-bottom: 15px;
        font-weight: bold;
        font-size: 1.25rem;
    }
    .analysis-card p {
        color: black;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Probability bars */
    .prob-container { margin: 10px 0; }
    .prob-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-weight: 500;
        color: black;
    }
    .prob-bar-bg {
        background: #e0e0e0;
        border-radius: 10px;
        height: 25px;
        overflow: hidden;
    }
    .prob-bar {
        background: #081f37;
        height: 100%;
        border-radius: 10px;
    }
    
    /* Section headings */
    .section-h2 {
        color: black;
        font-size: 1.75rem;
        font-weight: bold;
        margin: 20px 0 15px 0;
    }
    .section-h3 {
        color: black;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 15px 0 10px 0;
    }
    
    /* Center alignment for results */
    .results-section { text-align: center; }
    .results-section .prob-container,
    .results-section .analysis-card { text-align: left; }
    </style>
    """, unsafe_allow_html=True)
    
    # ============================================
    # PAGE HEADER - DARK NAVY
    # ============================================
    st.markdown("""
    <div class="page-header">
        <h1>🧬 SCLC MOLECULAR SUBTYPE CLASSIFICATION</h1>
        <p>Classify SCLC into molecular subtypes (A/N/P/Y) based on biomarker expression</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for results
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'classification_result' not in st.session_state:
        st.session_state.classification_result = None
    if 'full_analysis_result' not in st.session_state:
        st.session_state.full_analysis_result = None
    
    # ============================================
    # MAIN CONTENT - 60% Width Centered
    # ============================================
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    
    # ============================================
    # BIOMARKER EXPRESSION PANEL
    # ============================================
    st.markdown("<h2 class='section-h2'>🧬 Biomarker Expression Panel</h2>", unsafe_allow_html=True)
    
    # Default values
    defaults = {
        "ascl1": 8.0, "neurod1": 7.0, "pou2f3": 6.5, "yap1": 7.5,
        "tp53": 10.0, "rb1": 6.0, "myc": 8.5, "mycl": 8.0, "mycn": 6.5,
        "dll3": 7.0, "bcl2": 8.0, "notch1": 6.5, "myc_family": 23.0,
        "dual_loss": 1, "mean": 7.5, "stddev": 3.5, "genes": 9000
    }
    
    # Transcription Factors
    st.markdown("<h3 class='section-h3'>Transcription Factors</h3>", unsafe_allow_html=True)
    col_tf1, col_tf2 = st.columns(2)
    with col_tf1:
        ascl1 = st.slider("ASCL1", 0.0, 15.0, float(defaults["ascl1"]), 0.1, 
                          help="Achaete-scute homolog 1")
        neurod1 = st.slider("NEUROD1", 0.0, 15.0, float(defaults["neurod1"]), 0.1, 
                           help="Neurogenic differentiation 1")
    with col_tf2:
        pou2f3 = st.slider("POU2F3", 0.0, 15.0, float(defaults["pou2f3"]), 0.1, 
                          help="POU class 2 homeobox 3")
        yap1 = st.slider("YAP1", 0.0, 15.0, float(defaults["yap1"]), 0.1, 
                        help="Yes-associated protein 1")
    
    # Tumor Suppressors & Markers
    st.markdown("<h3 class='section-h3'>Tumor Suppressors & Markers</h3>", unsafe_allow_html=True)
    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        rb1 = st.slider("RB1", 0.0, 15.0, float(defaults["rb1"]), 0.1)
        tp53 = st.slider("TP53", 0.0, 15.0, float(defaults["tp53"]), 0.1)
        bcl2 = st.slider("BCL2", 0.0, 15.0, float(defaults["bcl2"]), 0.1)
    with col_ts2:
        dll3 = st.slider("DLL3", 0.0, 15.0, float(defaults["dll3"]), 0.1)
        notch1 = st.slider("NOTCH1", 0.0, 15.0, float(defaults["notch1"]), 0.1)
    
    # MYC Family & Calculated Features
    st.markdown("<h3 class='section-h3'>🔬 MYC Family & Calculated Features</h3>", unsafe_allow_html=True)
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        myc = st.slider("MYC", 0.0, 15.0, float(defaults["myc"]), 0.1)
        mycl = st.slider("MYCL", 0.0, 15.0, float(defaults["mycl"]), 0.1)
        mycn = st.slider("MYCN", 0.0, 15.0, float(defaults["mycn"]), 0.1)
    with col_calc2:
        myc_family_score = st.number_input("MYC Family Score", 0.0, 50.0, 
                                           float(defaults["myc_family"]), 0.1)
        mean_expression = st.number_input("Mean Expression", 0.0, 15.0, 
                                          float(defaults["mean"]), 0.1)
        stddev_expression = st.number_input("Std Dev", 0.0, 10.0, 
                                            float(defaults["stddev"]), 0.1)
    
    tp53_rb1_dual_loss = st.selectbox("TP53/RB1 Dual Loss", [0, 1], 
                                    index=int(defaults["dual_loss"]))
    genes_expressed = st.number_input("Genes Expressed", 0, 15000, 
                                    int(defaults["genes"]))
    
    st.markdown("---")
    
    # ============================================
    # ANALYSIS BUTTONS (Re-created)
    # Using default st.button which are now styled Dark Navy by CSS
    # ============================================
    st.markdown("<h3 class='section-h3'>🔬 Analysis Type</h3>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        # Re-creating button 1
        predict_btn = st.button(
            "⚡ Classify Subtype",
            use_container_width=True,
            key="btn_predict"
        )
        
    with col_btn2:
        # Re-creating button 2
        analyze_btn = st.button(
            "🔍 Full Analysis",
            use_container_width=True,
            key="btn_analyze"
        )
    
    # ============================================
    # CLASSIFICATION RESULTS
    # ============================================
    if predict_btn:
        send_kafka_event("sclc_user_authentication", "button_click", "subtype_classify", str(st.session_state.user_id), "clicked")
        with st.spinner("🧬 Classifying subtype..."):
            query = f"""
            SELECT PREDICT_SUBTYPE(
                {ascl1}, {neurod1}, {pou2f3}, {yap1}, {tp53}, {rb1},
                {myc_family_score}, {tp53_rb1_dual_loss}, {dll3}, {bcl2},
                {notch1}, {mean_expression}, {stddev_expression}, {genes_expressed}
            ) AS prediction
            """
            result = session.sql(query).collect()
            
            if result:
                pred_json = result[0]['PREDICTION']
                pred = json.loads(pred_json) if isinstance(pred_json, str) else pred_json
                st.session_state.classification_result = pred
                st.session_state.show_results = True
    
    # Display results if available
    if st.session_state.show_results and st.session_state.classification_result:
        pred = st.session_state.classification_result
        subtype = pred['predicted_subtype']
        confidence = pred['confidence']
        probs = pred['probabilities']
        
        st.markdown("<div class='results-section'>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-h2'>📊 Classification Results</h2>", unsafe_allow_html=True)
        
        # Subtype card - always #081f37
        st.markdown(f"""
        <div class='subtype-card'>
            <div style='font-size: 3rem;'>🧬</div>
            <h2 style='margin: 10px 0; color: white;'>Molecular Subtype: {subtype}</h2>
            <p style='font-size: 1.2rem; color: white;'>Confidence: {confidence*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics - Light grey cards with black text
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{confidence*100:.1f}%</div>
                <div class='metric-label'>Model Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{subtype}</div>
                <div class='metric-label'>Predicted Subtype</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Probability bars - all #081f37
        st.markdown("<h3 class='section-h3'>📊 Subtype Probabilities</h3>", unsafe_allow_html=True)
        for sub in ['SCLC-A', 'SCLC-N', 'SCLC-P', 'SCLC-Y']:
            prob = probs.get(sub, 0)
            prob_pct = prob * 100
            
            st.markdown(f"""
            <div class='prob-container'>
                <div class='prob-label'>
                    <span>{sub}</span>
                    <span>{prob_pct:.1f}%</span>
                </div>
                <div class='prob-bar-bg'>
                    <div class='prob-bar' style='width: {prob_pct}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Therapeutic Implications - Light grey card with H3 heading
        st.markdown("<h3 class='section-h3'>💊 Therapeutic Implications</h3>", unsafe_allow_html=True)
        treatments = {
            "SCLC-A": {
                "type": "Classic Neuroendocrine",
                "marker": "ASCL1-high",
                "therapy": "Standard platinum chemotherapy, DLL3-targeted therapy (Rovalpituzumab)"
            },
            "SCLC-N": {
                "type": "Alternative Neuroendocrine",
                "marker": "NEUROD1-high",
                "therapy": "Platinum chemotherapy, Aurora kinase inhibitors, BCL-2 inhibitors"
            },
            "SCLC-P": {
                "type": "Non-Neuroendocrine (Tuft-like)",
                "marker": "POU2F3-high",
                "therapy": "PARP inhibitors, FGFR inhibitors, Immunotherapy"
            },
            "SCLC-Y": {
                "type": "Non-Neuroendocrine (MYC-driven)",
                "marker": "YAP1-high",
                "therapy": "Immunotherapy, NOTCH/MEK/mTOR inhibitors"
            }
        }
        
        if subtype in treatments:
            info = treatments[subtype]
            st.markdown(f"""
            <div class='analysis-card'>
                <h3>{info['type']}</h3>
                <p><strong>Marker:</strong> {info['marker']}</p>
                <p><strong>Treatment:</strong> {info['therapy']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.success("✅ Subtype classification complete!")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ============================================
    # FULL ANALYSIS SECTION
    # ============================================
    if analyze_btn:
        send_kafka_event("sclc_user_authentication", "button_click", "subtype_full_analysis", str(st.session_state.user_id), "clicked")
        st.markdown("---")
        st.markdown("<h2 class='section-h2'>🔍 Comprehensive Biomarker Analysis</h2>", unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🧬 Initializing analysis...")
        progress_bar.progress(10)
        
        with st.spinner("Running comprehensive biomarker analysis..."):
            status_text.text("🔬 Running ML classification...")
            progress_bar.progress(25)
            
            query = f"""
            CALL RUN_FULL_SUBTYPE_ANALYSIS(
                {ascl1}, {neurod1}, {pou2f3}, {yap1}, {tp53}, {rb1},
                {myc_family_score}, {tp53_rb1_dual_loss}, {dll3}, {bcl2},
                {notch1}, {mean_expression}, {stddev_expression}, {genes_expressed}
            )
            """
            
            status_text.text("🔍 Analyzing biomarker patterns...")
            progress_bar.progress(50)
            
            result = session.sql(query).collect()
            
            status_text.text("🤖 Generating clinical report...")
            progress_bar.progress(85)
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            if result:
                full_result_json = result[0]['RUN_FULL_SUBTYPE_ANALYSIS']
                full_result = json.loads(full_result_json) if isinstance(full_result_json, str) else full_result_json
                st.session_state.full_analysis_result = full_result
                
                ml_pred = full_result['ml_prediction']
                if isinstance(ml_pred, str):
                    ml_pred = json.loads(ml_pred)
                
                biomarker_analysis = full_result['biomarker_analysis']
                similar_profiles = full_result['similar_profiles']
                clinical_analysis = full_result['clinical_analysis']
                timestamp = full_result['timestamp']
                
                subtype = ml_pred['predicted_subtype']
                confidence = ml_pred['confidence']
                
                st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                
                # Subtype card - always #081f37
                st.markdown(f"""
                <div class='subtype-card'>
                    <div style='font-size: 3rem;'>🧬</div>
                    <h2 style='margin: 10px 0; color: white;'>Molecular Subtype: {subtype}</h2>
                    <p style='font-size: 1.2rem; color: white;'>Confidence: {confidence*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics row
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{subtype}</div>
                        <div class='metric-label'>Predicted Subtype</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>{confidence*100:.1f}%</div>
                        <div class='metric-label'>Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_m3:
                    st.markdown("""
                    <div class='metric-card'>
                        <div class='metric-value'>✓</div>
                        <div class='metric-label'>Complete</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Biomarker Analysis
                st.markdown("<h3 class='section-h3'>🔬 Biomarker Analysis</h3>", unsafe_allow_html=True)
                dominant_tf = biomarker_analysis.get('dominant_tf', 'Unknown')
                dominant_value = biomarker_analysis.get('dominant_value', 0)
                ne_score = biomarker_analysis.get('ne_score', 0)
                non_ne_score = biomarker_analysis.get('non_ne_score', 0)
                
                # Equal size metric cards for biomarker analysis
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.markdown(f"""
                    <div class='metric-card-equal'>
                        <div class='metric-label'>Dominant Transcription Factor</div>
                        <div class='metric-value'>{dominant_tf}</div>
                        <div class='metric-label'>Expression: {dominant_value:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b2:
                    st.markdown(f"""
                    <div class='metric-card-equal'>
                        <div class='metric-value'>{ne_score:.2f}</div>
                        <div class='metric-label'>NE Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b3:
                    st.markdown(f"""
                    <div class='metric-card-equal'>
                        <div class='metric-value'>{non_ne_score:.2f}</div>
                        <div class='metric-label'>Non-NE Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Expression levels - H3 heading
                st.markdown("<h3 class='section-h3'>Expression Levels</h3>", unsafe_allow_html=True)
                biomarkers_display = {
                    'ASCL1': biomarker_analysis.get('ascl1', 0),
                    'NEUROD1': biomarker_analysis.get('neurod1', 0),
                    'POU2F3': biomarker_analysis.get('pou2f3', 0),
                    'YAP1': biomarker_analysis.get('yap1', 0)
                }
                
                for marker, value in biomarkers_display.items():
                    is_dominant = (marker == dominant_tf)
                    marker_color = "#081f37" if is_dominant else "#dee2e6"
                    
                    st.markdown(f"""
                    <div class='prob-container'>
                        <div class='prob-label'>
                            <span style='font-weight: {"bold" if is_dominant else "normal"};'>{marker}</span>
                            <span>{value:.2f}</span>
                        </div>
                        <div class='prob-bar-bg'>
                            <div style='background: {marker_color}; height: 100%; width: {min(value/15*100, 100)}%; border-radius: 10px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Database Validation - H3 heading
                st.markdown("<h3 class='section-h3'>👥 Database Search Results</h3>", unsafe_allow_html=True)
                similar_data = full_result.get('similar_profiles_data', {})
                sample_count = similar_data.get('samples_found', 0)
                
                st.markdown(f"""
                <div class='analysis-card'>
                    <p><strong>{sample_count}</strong> similar samples found</p>
                    <p>{similar_profiles}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # AI Clinical Assessment - Each section in separate card with H3 headings
                st.markdown("<h3 class='section-h3'>📋 AI Clinical Assessment</h3>", unsafe_allow_html=True)
                st.caption("*Generated by Llama 3.1 405B (Snowflake Cortex AI)*")
                
                # Parse clinical analysis into sections
                import re
                
                # Define sections to extract
                sections = [
                    ("SUBTYPE INTERPRETATION", "🎯"),
                    ("KEY BIOMARKER FINDINGS", "🔬"),
                    ("DIFFERENTIAL DIAGNOSIS", "🧪"),
                    ("THERAPEUTIC IMPLICATIONS", "💊"),
                    ("PROGNOSIS", "📊")
                ]
                
                # Try to parse sections from clinical_analysis
                for section_name, icon in sections:
                    # Try different patterns to find the section
                    patterns = [
                        rf'\*\*{section_name}:\*\*\s*(.*?)(?=\*\*[A-Z]|\Z)',
                        rf'{section_name}:\s*(.*?)(?=[A-Z][A-Z\s]+:|\Z)',
                        rf'\d+\.\s*{section_name}[:\s]*(.*?)(?=\d+\.\s*[A-Z]|\Z)'
                    ]
                    
                    content = None
                    for pattern in patterns:
                        match = re.search(pattern, clinical_analysis, re.DOTALL | re.IGNORECASE)
                        if match:
                            content = match.group(1).strip()
                            break
                    
                    if content:
                        st.markdown(f"""
                        <div class='analysis-card'>
                            <h3>{icon} {section_name}</h3>
                            <p>{content}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # If no sections were parsed, show the full text
                if not any(re.search(rf'{s[0]}', clinical_analysis, re.IGNORECASE) for s in sections):
                    st.markdown(f"""
                    <div class='analysis-card'>
                        <h3>📋 Clinical Analysis</h3>
                        <p>{clinical_analysis}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success(f"✅ Complete analysis finished! (Generated: {timestamp})")
                
                # Download report - #081f37 button
                report_text = f"""
SCLC Subtype Classification Report
Generated: {timestamp}

BIOMARKER PROFILE:
- ASCL1: {ascl1:.2f}
- NEUROD1: {neurod1:.2f}
- POU2F3: {pou2f3:.2f}
- YAP1: {yap1:.2f}

CLASSIFICATION:
- Predicted Subtype: {subtype}
- Confidence: {confidence*100:.1f}%

AI CLINICAL ASSESSMENT:
{clinical_analysis}
                """
                
                st.download_button(
                    "📥 Download Subtype Report",
                    report_text,
                    f"sclc_subtype_{subtype}_{int(confidence*100)}.txt",
                    "text/plain",
                    use_container_width=True
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close main-content div
    
    # ============================================
    # NAVIGATION BUTTONS - Black color (Wrapped in .nav-button)
    # ============================================
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='nav-button'>", unsafe_allow_html=True)
        if st.button("← Back to Profile", use_container_width=True, key="subtype_back"):
            go_back_to_profile()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='nav-button'>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True, key="subtype_logout"):
            logout()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>OncoDetect-AI</strong> | Advanced Precision Oncology Platform</p>
        <p style='font-size: 0.85rem;'>Powered by Snowflake ML + Cortex AI</p>
        <p style='font-size: 0.8rem;'>For Research Use Only | Not for Clinical Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# 22. MAIN APPLICATION ROUTER
# ============================================================

def main():
    """Main application router"""
    if st.session_state.logged_in:
        # User is logged in - route based on role and active module
        if st.session_state.user_role == 'admin':
            # Admin user
            if st.session_state.show_analytics:
                show_analytics_page()
            else:
                show_admin_profile_page()
        else:
            # Regular user - route to active module
            if st.session_state.show_research:
                show_research_page()
            elif st.session_state.show_risk_prediction:
                show_risk_prediction_page()
            elif st.session_state.show_subtype_classification:
                show_subtype_classification_page()
            else:
                show_user_profile_page()
    else:
        # User not logged in - show auth pages
        if st.session_state.show_signup:
            show_signup_page()
        else:
            show_login_page()

# ============================================================
# 23. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()