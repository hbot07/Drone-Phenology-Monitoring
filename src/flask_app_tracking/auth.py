"""
Authentication — email/password + Google sign-in, unified behind a session JWT.

Two ways to log in:
  A) Email + password  -> POST /api/auth/signup, POST /api/auth/login
  B) Google ID token   -> POST /api/auth/google

Both issue our own **session JWT** (signed with DPM_SECRET_KEY). The browser
stores it and sends it as  Authorization: Bearer <session_jwt>  on every API
call. get_current_user() validates that session JWT -- it does NOT re-verify
Google on every request (Google is only verified once, at /api/auth/google).

Set DPM_AUTH_ENABLED=false to bypass everything for local dev.
"""
import os
import time
import hashlib
import hmac
import secrets

from fastapi import Header, HTTPException
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import jwt  # PyJWT

import db

AUTH_ENABLED = os.environ.get("DPM_AUTH_ENABLED", "true").lower() == "true"
GOOGLE_CLIENT_ID = os.environ.get("DPM_GOOGLE_CLIENT_ID", "")
SECRET_KEY = os.environ.get("DPM_SECRET_KEY", "dev-secret-change-me")
SESSION_HOURS = int(os.environ.get("DPM_SESSION_HOURS", "24"))

_DEV_USER = {"email": "dev@localhost", "name": "Local Dev", "picture": None}
_google_request = google_requests.Request()


# ---------------------------------------------------------------------------
# Password hashing -- PBKDF2-HMAC-SHA256 (stdlib only)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Return 'salt$hash' -- salt and PBKDF2 hash, both hex-encoded."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored 'salt$hash'."""
    try:
        salt, expected = stored.split("$", 1)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000)
    return hmac.compare_digest(dk.hex(), expected)


# ---------------------------------------------------------------------------
# Session JWT -- our own token, signed with SECRET_KEY
# ---------------------------------------------------------------------------
def issue_session_token(email: str, name: str | None) -> str:
    now = int(time.time())
    payload = {"sub": email, "name": name, "iat": now, "exp": now + SESSION_HOURS * 3600}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def decode_session_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired -- please sign in again")
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Invalid session token: {e}") from e


# ---------------------------------------------------------------------------
# Google ID token verification (only at /api/auth/google)
# ---------------------------------------------------------------------------
def verify_google_token(token: str) -> dict:
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(500, "Server misconfigured: DPM_GOOGLE_CLIENT_ID is not set.")
    try:
        claims = id_token.verify_oauth2_token(token, _google_request, GOOGLE_CLIENT_ID)
    except ValueError as e:
        raise HTTPException(401, f"Invalid Google token: {e}") from e
    if claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(401, "Invalid token issuer")
    if not claims.get("email"):
        raise HTTPException(401, "Token has no email claim")
    if not claims.get("email_verified", False):
        raise HTTPException(401, "Google email is not verified")
    return claims


# ---------------------------------------------------------------------------
# Current-user dependency -- validates OUR session JWT
# ---------------------------------------------------------------------------
async def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not AUTH_ENABLED:
        await db.upsert_user(_DEV_USER["email"], _DEV_USER["name"], _DEV_USER["picture"])
        return dict(_DEV_USER)

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated")

    token = authorization.split(" ", 1)[1].strip()
    payload = decode_session_token(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(401, "Invalid session token")

    user = await db.get_user(email)
    if not user:
        raise HTTPException(401, "User no longer exists")
    return {"email": user["email"], "name": user.get("name"), "picture": user.get("picture")}


async def require_run_owner(run_id: str, user: dict) -> dict:
    run = await db.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    if run["owner_email"] != user["email"]:
        raise HTTPException(404, "Run not found")
    return run


# ---------------------------------------------------------------------------
# Request models for the auth endpoints (imported by server.py)
# ---------------------------------------------------------------------------
class SignupReq(BaseModel):
    email: str
    password: str
    name: str | None = None


class LoginReq(BaseModel):
    email: str
    password: str


class GoogleReq(BaseModel):
    credential: str   # the Google ID token from the browser
