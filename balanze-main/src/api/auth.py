"""
Authentication and Authorization Module for the Gait Analysis API

This module provides JWT-based authentication and role-based access control.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Security constants
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Mock user database (replace with actual database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Administrator",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
        "roles": ["admin"]
    },
    "clinician": {
        "username": "clinician",
        "full_name": "Clinical Staff",
        "email": "clinician@example.com",
        "hashed_password": pwd_context.hash("clinician123"),
        "disabled": False,
        "roles": ["clinician"]
    },
    "researcher": {
        "username": "researcher",
        "full_name": "Research Staff",
        "email": "researcher@example.com",
        "hashed_password": pwd_context.hash("researcher123"),
        "disabled": False,
        "roles": ["researcher"]
    }
}

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None
    roles: list[str] = []

class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    roles: list[str] = []

class UserInDB(User):
    """User model for database operations."""
    hashed_password: str

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)

def get_user(db: dict, username: str) -> Optional[UserInDB]:
    """Get a user from the database."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(fake_db: dict, username: str, password: str) -> Optional[User]:
    """Authenticate a user."""
    user = get_user(fake_db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, roles=payload.get("roles", []))
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

class RoleChecker:
    ""
    Role-based access control.
    
    Example usage:
        @app.get("/admin")
        async def admin_route(current_user: User = Depends(RoleChecker(["admin"]))):
            return {"message": "Admin access granted"}
    """
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: User = Depends(get_current_active_user)):
        if not any(role in self.allowed_roles for role in user.roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted"
            )
        return user

# Dependency to check if user has any of the required roles
def has_role(required_roles: list):
    """Check if the current user has any of the required roles."""
    async def role_checker(current_user: User = Depends(get_current_active_user)):
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker

# Dependency for rate limiting
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests: int, window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests: Number of requests allowed per window
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.timestamps = []
    
    async def __call__(self, request, response):
        """Check if the request is allowed."""
        import time
        
        current_time = time.time()
        
        # Remove timestamps older than the window
        self.timestamps = [t for t in self.timestamps if t > current_time - self.window]
        
        if len(self.timestamps) >= self.requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(self.window)},
            )
        
        self.timestamps.append(current_time)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter(requests=100, window=60)  # 100 requests per minute

# Example usage in FastAPI route:
# @app.get("/api/endpoint")
# async def endpoint(
#     request: Request,
#     response: Response,
#     rate_limit: bool = Depends(rate_limiter)
# ):
#     return {"message": "Request processed"}
