"""
Two-token JWT auth: access token (15 min) in JSON, refresh token (7 days) in httpOnly cookie.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Annotated, Optional

import jwt
from fastapi import Request, HTTPException, Depends, Response
from passlib.context import CryptContext

# Load env (same as main)
_root = Path(__file__).resolve().parent.parent
from dotenv import load_dotenv
for _name in (".env", ".env.local"):
    load_dotenv(_root / _name)

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM = "HS256"
ACCESS_EXP_MINUTES = 15
REFRESH_EXP_DAYS = 7
COOKIE_NAME = "refresh_token"
# In production (e.g. Fly.io) we're HTTPS; for local dev allow http
SECURE_COOKIE = os.getenv("ENVIRONMENT", "").lower() in ("production", "prod")

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: int, email: str) -> str:
    exp = _now() + timedelta(minutes=ACCESS_EXP_MINUTES)
    return jwt.encode(
        {"sub": user_id, "email": email, "exp": exp, "type": "access"},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def create_refresh_token(user_id: int, email: str) -> str:
    exp = _now() + timedelta(days=REFRESH_EXP_DAYS)
    return jwt.encode(
        {"sub": user_id, "email": email, "exp": exp, "type": "refresh"},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )


def verify_access_token(token: str) -> dict | None:
    if not JWT_SECRET or not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except jwt.InvalidTokenError:
        return None


def verify_refresh_token(token: str) -> dict | None:
    if not JWT_SECRET or not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return payload
    except jwt.InvalidTokenError:
        return None


def set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=SECURE_COOKIE,
        max_age=REFRESH_EXP_DAYS * 24 * 3600,
        path="/",
    )


def clear_refresh_cookie(response: Response) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value="",
        httponly=True,
        samesite="lax",
        secure=SECURE_COOKIE,
        max_age=0,
        path="/",
    )


class CurrentUser:
    def __init__(self, user_id: int, email: str):
        self.id = user_id
        self.email = email


async def get_current_user(request: Request) -> CurrentUser:
    """Dependency: require valid access token; raise 401 otherwise."""
    auth = request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth[7:].strip()
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    sub = payload.get("sub")
    email = payload.get("email") or ""
    if sub is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return CurrentUser(user_id=int(sub), email=email)


async def get_current_user_optional(request: Request) -> Optional[CurrentUser]:
    """Dependency: return CurrentUser if valid Bearer token present, else None (allows anonymous with X-Instance-ID)."""
    auth = request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    payload = verify_access_token(token)
    if not payload:
        return None
    sub = payload.get("sub")
    email = payload.get("email") or ""
    if sub is None:
        return None
    return CurrentUser(user_id=int(sub), email=email)
