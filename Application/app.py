import streamlit as st
import requests
from datetime import date, datetime
import re
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="User Authentication",
    page_icon="🔐",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Black background for the entire app */
    .stApp {
        background-color: black;
    }
    
    /* Center align all titles */
    h1, h2, h3 {
        text-align: center;
        color: white !important;
    }
    
    /* All text white */
    .stMarkdown, p, label, .stCaption {
        color: white !important;
    }
    
    /* Input field labels white */
    .stTextInput label, .stDateInput label {
        color: white !important;
    }
    
    /* Input field text (what user types) black */
    .stTextInput input, .stDateInput input {
        background-color: white !important;
        color: black !important;
    }
    
    /* Placeholder text grey */
    .stTextInput input::placeholder {
        color: rgba(128, 128, 128, 0.7) !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Form styling */
    .stForm {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Success/Error messages - make success box gray with white text */
    .stSuccess {
        background-color: rgba(128, 128, 128, 0.3) !important;
        color: white !important;
    }
    
    .stSuccess p {
        color: white !important;
    }
    
    .stError {
        color: black !important;
    }
    
    .stInfo {
        background-color: rgba(128, 128, 128, 0.3) !important;
        color: white !important;
    }
    
    .stInfo p, .stInfo div, .stInfo strong, .stInfo li {
        color: white !important;
    }
    
    /* Divider white */
    hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Green text for met requirements */
    .requirement-met {
        color: #00ff00 !important;
    }
    
    /* Red text for unmet requirements */
    .requirement-unmet {
        color: #ff4444 !important;
    }
    
    /* Spinner/Loading text white */
    .stSpinner > div {
        color: white !important;
    }
    
    /* Progress bar styling - blue color */
    .stProgress > div > div > div {
        background-color: #1f77b4 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'

def validate_password(password):
    """Validate password requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long!⚠️"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least 1 uppercase letter!⚠️"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least 1 special character!⚠️"
    return True, "Password is valid!"

def check_password_requirements(password):
    """Check individual password requirements and return status"""
    requirements = {
        "length": len(password) >= 8,
        "uppercase": bool(re.search(r'[A-Z]', password)),
        "special": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password))
    }
    return requirements

def signup_page():
    """Render the signup page"""
    st.markdown("<h1>Lets Create Your Account!</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.form("signup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            first_name = st.text_input("First Name*", placeholder="John")
        with col2:
            last_name = st.text_input("Last Name*", placeholder="Doe")
        
        date_of_birth = st.date_input(
            "Date of Birth*",
            min_value=date(1900, 1, 1),
            max_value=date.today(),
            value=date(2000, 1, 1)
        )
        
        email = st.text_input("Email*", placeholder="john.doe@example.com")
        username = st.text_input("Username*", placeholder="johndoe123")
        
        password = st.text_input("Password*", type="password", placeholder="Enter password", key="signup_password")
        confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Re-enter password")
        
        submit_button = st.form_submit_button("Sign Up", use_container_width=True)
        
        if submit_button:
            # Validation
            errors = []
            
            if not all([first_name, last_name, email, username, password, confirm_password]):
                errors.append("All fields are required!⚠️")
            
            if password != confirm_password:
                errors.append("Passwords do not match!⚠️")
            
            is_valid, message = validate_password(password)
            if not is_valid:
                errors.append(message)
            
            if not re.match(r'^[a-zA-Z0-9_]+$', username):
                errors.append("Username can only contain letters, numbers, and underscores!⚠️")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Show progress bar while creating account
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.markdown("<p style='color: white; text-align: center;'>Creating your account...</p>", unsafe_allow_html=True)
                    
                    # Simulate progress
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
                    # Send signup request to API
                    response = requests.post(
                        f"{API_BASE_URL}/signup",
                        json={
                            "first_name": first_name,
                            "last_name": last_name,
                            "date_of_birth": date_of_birth.isoformat(),
                            "email": email,
                            "username": username,
                            "password": password
                        }
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if response.status_code == 201:
                        st.success("Account created successfully!✅ Please login.")
                        time.sleep(2)
                        st.session_state.page = 'login'
                        st.rerun()
                    else:
                        error_detail = response.json().get("detail", "Signup failed!")
                        st.error(f"❌ {error_detail}")
                
                except requests.exceptions.ConnectionError:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Cannot connect to the server. Please ensure the API is running.")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ An error occurred: {str(e)}")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Already have an account?</p>", unsafe_allow_html=True)
    if st.button("Login here!🔑➡️", use_container_width=True):
        st.session_state.page = 'login'
        st.rerun()

def login_page():
    """Render the login page"""
    st.markdown("<h1>ONCODETECT-AI🫁🎗️</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        submit_button = st.form_submit_button("Login", use_container_width=True)
        
        if submit_button:
            if not username or not password:
                st.error("Please enter both username and password!⚠️")
            else:
                # Show progress bar while logging in
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.markdown("<p style='color: white; text-align: center;'>Logging you in...</p>", unsafe_allow_html=True)
                    
                    # Simulate progress
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.01)
                    
                    response = requests.post(
                        f"{API_BASE_URL}/login",
                        json={
                            "username": username,
                            "password": password
                        }
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.logged_in = True
                        st.session_state.username = data['username']
                        st.session_state.email = data['email']
                        st.success(f"✅ Welcome back, {data['username']}!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Invalid username or password!❌")
                
                except requests.exceptions.ConnectionError:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ Cannot connect to the server. Please ensure the API is running.")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ An error occurred: {str(e)}")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Don't have an account?</p>", unsafe_allow_html=True)
    if st.button("Sign up here!👤", use_container_width=True):
        st.session_state.page = 'signup'
        st.rerun()

def dashboard_page():
    """Render the dashboard after successful login"""
    st.markdown(f"<h1>👋 Welcome, {st.session_state.username}!</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.success("You are successfully logged in!✅")
    
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.email = None
        st.session_state.page = 'login'
        st.rerun()

# Main application logic
def main():
    if st.session_state.logged_in:
        dashboard_page()
    else:
        if st.session_state.page == 'signup':
            signup_page()
        else:
            login_page()

if __name__ == "__main__":
    main()