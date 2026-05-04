# app/schemas/auth.py

from pydantic import BaseModel, EmailStr
from typing import Optional


class SignUpRequest(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class BootstrapAdminRequest(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    password: str
    bootstrap_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class AuthTokensResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    user_id: Optional[str] = None
    tier: Optional[str] = None
    role: Optional[str] = None


class SignOutResponse(BaseModel):
    success: bool = True
