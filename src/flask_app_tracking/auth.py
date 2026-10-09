"""
Authentication - Google sign-in only, backed by a session JWT.

Sign-in flow:
  Browser → Google Identity Services → id_token → POST /api/auth/google
  Server verifies the id_token, issues its own HS256 session JWT.
  Browser stores it in localStorage and sends it as:
    Authorization: Bearer <session_jwt>
  on every API call. get_current_user() validates that session JWT --
  it does NOT re-verify Google on every request.

Set DPM_AUTH_ENABLED=false to bypass everything for local dev.
"""
import os
import time

from fastapi import Header, HTTPException
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import requests as _requests
import jwt  # PyJWT

import db
from log_config import get_logger

_log = get_logger("auth")

AUTH_ENABLED = os.environ.get("DPM_AUTH_ENABLED", "true").lower() == "true"
GOOGLE_CLIENT_ID = os.environ.get("DPM_GOOGLE_CLIENT_ID", "")
SECRET_KEY = os.environ.get("DPM_SECRET_KEY", "dev-secret-change-me")
SESSION_HOURS = int(os.environ.get("DPM_SESSION_HOURS", "24"))

_DEV_USER = {"email": "dev@localhost", "name": "Local Dev", "picture": None}


def _build_google_request() -> google_requests.Request:
    """
    Build a requests.Session that explicitly routes through the institutional
    proxy (HTTPS_PROXY / HTTP_PROXY env vars).

    This is necessary on networks like IITD where all outbound internet traffic
    must go through a proxy. Without this, google.oauth2.id_token.verify_oauth2_token
    tries a direct connection to Google, the firewall blocks it, and the call
    hangs until the upstream proxy (nginx) returns a 504.

    We set the proxy explicitly on the session rather than relying on NO_PROXY
    not listing Google domains — belt-and-suspenders.
    """
    session = _requests.Session()
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY", "")
    if https_proxy:
        session.proxies = {"http": https_proxy, "https": https_proxy}
        _log.info("Google token verification will use proxy: %s", https_proxy)
    else:
        _log.info("Google token verification: no proxy configured (direct connection)")
    return google_requests.Request(session=session)


_google_request = _build_google_request()


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
        _log.info("Session expired for a request")
        raise HTTPException(401, "Session expired -- please sign in again")
    except jwt.InvalidTokenError as e:
        _log.info("Invalid session token: %s", e)
        raise HTTPException(401, f"Invalid session token: {e}") from e


# ---------------------------------------------------------------------------
# Google ID token verification (only at /api/auth/google)
# ---------------------------------------------------------------------------
def verify_google_token(token: str) -> dict:
    if not GOOGLE_CLIENT_ID:
        _log.error("DPM_GOOGLE_CLIENT_ID is not set - SSO misconfigured")
        raise HTTPException(500, "Server misconfigured: DPM_GOOGLE_CLIENT_ID is not set.")
    try:
        claims = id_token.verify_oauth2_token(token, _google_request, GOOGLE_CLIENT_ID)
    except ValueError as e:
        _log.info("Google token verification failed: %s", e)
        raise HTTPException(401, f"Invalid Google token: {e}") from e
    if claims.get("iss") not in ("accounts.google.com", "https://accounts.google.com"):
        _log.info("Google token rejected - bad issuer")
        raise HTTPException(401, "Invalid token issuer")
    if not claims.get("email"):
        _log.info("Google token rejected - no email claim")
        raise HTTPException(401, "Token has no email claim")
    if not claims.get("email_verified", False):
        _log.info("Google token rejected - email not verified")
        raise HTTPException(401, "Google email is not verified")
    _log.info("Google auth success  email=%s", claims["email"])
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
# Request model for the Google auth endpoint (imported by server.py)
# ---------------------------------------------------------------------------
class GoogleReq(BaseModel):
    credential: str   # the Google ID token from the browser
