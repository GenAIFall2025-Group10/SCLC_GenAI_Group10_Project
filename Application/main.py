from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator
from datetime import date
import snowflake.connector
from passlib.context import CryptContext
import re
import os
from typing import Optional
from dotenv import load_dotenv

# ADD THESE TWO LINES RIGHT HERE
load_dotenv()  # Load environment variables from .env file
print("Loading Snowflake config...")

app = FastAPI(title="ONCODETECT-AI")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Snowflake connection configuration
SNOWFLAKE_CONFIG = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA")
}

def get_snowflake_connection():
    """Create and return a Snowflake connection"""
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def init_database():
    """Initialize the database table if it doesn't exist"""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS USERS (
                id INTEGER AUTOINCREMENT PRIMARY KEY,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                date_of_birth DATE NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                username VARCHAR(50) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()

# Pydantic models
class SignupRequest(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long!')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least 1 uppercase letter!')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least 1 special character!')
        return v

    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, and underscores!')
        return v

class LoginRequest(BaseModel):
    username: str
    password: str

class MessageResponse(BaseModel):
    message: str

class LoginResponse(BaseModel):
    message: str
    username: str
    email: str

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()

@app.get("/")
async def root():
    return {"message": "User Authentication API is running"}

@app.post("/signup", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def signup(user: SignupRequest):
    """Register a new user"""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Check if username already exists
        cursor.execute("SELECT username FROM users WHERE username = %s", (user.username,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email already exists
        cursor.execute("SELECT email FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Hash the password
        password_hash = pwd_context.hash(user.password)
        
        # Insert new user
        cursor.execute("""
            INSERT INTO users (first_name, last_name, date_of_birth, email, username, password_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            user.first_name,
            user.last_name,
            user.date_of_birth,
            user.email,
            user.username,
            password_hash
        ))
        
        conn.commit()
        return {"message": "User registered successfully!"}
    
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

@app.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    """Authenticate user login"""
    conn = get_snowflake_connection()
    cursor = conn.cursor()
    
    try:
        # Fetch user from database
        cursor.execute(
            "SELECT username, email, password_hash FROM users WHERE username = %s",
            (credentials.username,)
        )
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password!"
            )
        
        username, email, password_hash = user
        
        # Verify password
        if not pwd_context.verify(credentials.password, password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password!"
            )
        
        return {
            "message": "Login successful!",
            "username": username,
            "email": email
        }
    
    finally:
        cursor.close()
        conn.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}