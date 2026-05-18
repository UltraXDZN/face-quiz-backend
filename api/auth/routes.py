import os
import secrets
import jwt
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Response, Cookie
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import httpx

router = APIRouter(prefix="/auth", tags=["auth"])

# Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"

# Temporary state storage (use Redis in production)
state_tokens = {}


class LoginResponse(BaseModel):
    access_token: str
    user: dict


@router.get("/google/login")
async def google_login():
    """Initiate Google OAuth flow"""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    
    state = secrets.token_urlsafe(32)
    state_tokens[state] = True
    
    redirect_uri = f"{BACKEND_URL}/api/auth/google/callback"
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope=openid%20email%20profile&"
        f"state={state}&"
        f"access_type=offline"
    )
    
    return RedirectResponse(url=google_auth_url)


@router.get("/google/callback")
async def google_callback(code: str, state: str):
    """Handle Google OAuth callback"""
    if state not in state_tokens:
        return RedirectResponse(url=f"{FRONTEND_URL}/auth/error?message=Invalid%20state")
    
    del state_tokens[state]
    
    try:
        # Exchange code for tokens
        token_url = "https://oauth2.googleapis.com/token"
        redirect_uri = f"{BACKEND_URL}/api/auth/google/callback"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                token_url,
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                }
            )
        
        if response.status_code != 200:
            return RedirectResponse(url=f"{FRONTEND_URL}/auth/error?message=Token%20exchange%20failed")
        
        tokens = response.json()
        access_token = tokens.get("access_token")
        
        # Get user info from Google
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"}
            )
        
        if user_response.status_code != 200:
            return RedirectResponse(url=f"{FRONTEND_URL}/auth/error?message=Failed%20to%20get%20user%20info")
        
        user_data = user_response.json()
        
        # Create JWT token
        jwt_token = create_access_token({
            "email": user_data.get("email"),
            "name": user_data.get("name"),
            "picture": user_data.get("picture"),
        })
        
        # Redirect to frontend with token
        callback_url = f"{FRONTEND_URL}/auth/callback?token={jwt_token}"
        return RedirectResponse(url=callback_url)
        
    except Exception as e:
        return RedirectResponse(url=f"{FRONTEND_URL}/auth/error?message={str(e)}")


def create_access_token(data: dict, expires_delta: timedelta = timedelta(days=7)):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/verify")
async def verify_user_token(token: str):
    """Verify a JWT token and return user data"""
    return verify_token(token)

