"""
Authentication routes and helpers.
Routes: /auth/login  /auth/me  /auth/logout
        /auth/users  /auth/users/{id}/role  /auth/users/{id} (DELETE)
"""
import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from bson import ObjectId
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from shared_state import AUTH_USERS_COLLECTION, AUTH_SESSIONS_COLLECTION, VALID_ROLES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

AUTH_LOGIN_ATTEMPTS_COLLECTION = 'auth_login_attempts'
AUTH_SESSION_IDLE_TIMEOUT_MINUTES = int(os.getenv('AUTH_SESSION_IDLE_TIMEOUT_MINUTES', '120'))
AUTH_SESSION_MAX_AGE_MINUTES = int(os.getenv('AUTH_SESSION_MAX_AGE_MINUTES', '480'))
AUTH_LOGIN_MAX_ATTEMPTS = int(os.getenv('AUTH_LOGIN_MAX_ATTEMPTS', '5'))
AUTH_LOGIN_WINDOW_MINUTES = int(os.getenv('AUTH_LOGIN_WINDOW_MINUTES', '15'))
AUTH_LOGIN_BLOCK_MINUTES = int(os.getenv('AUTH_LOGIN_BLOCK_MINUTES', '15'))
ORGANIZATION_EMAIL_DOMAIN = 'neurodetect.ai'


def _get_processor():
    from batch_processor import processor as _p
    return _p


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class RoleUpdateRequest(BaseModel):
    role: str


class UserCreateRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str


def _hash_password(password: str, salt: Optional[str] = None) -> str:
    resolved_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), resolved_salt.encode('utf-8'), 120000)
    return f"{resolved_salt}${digest.hex()}"


def _verify_password(password: str, stored_password: str) -> bool:
    try:
        salt, _ = stored_password.split('$', 1)
        return _hash_password(password, salt) == stored_password
    except Exception:
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _is_valid_organization_email(email: str) -> bool:
    normalized = email.strip().lower()
    if '@' not in normalized:
        return False

    local_part, domain = normalized.rsplit('@', 1)
    return bool(local_part) and domain == ORGANIZATION_EMAIL_DOMAIN


def _ensure_auth_collections() -> None:
    if not _get_processor().db or not _get_processor().db.connected:
        logger.warning("MongoDB not connected - authentication endpoints unavailable")
        return

    users_collection = _get_processor().db.db[AUTH_USERS_COLLECTION]
    sessions_collection = _get_processor().db.db[AUTH_SESSIONS_COLLECTION]
    login_attempts = _get_processor().db.db[AUTH_LOGIN_ATTEMPTS_COLLECTION]
    users_collection.create_index('email', unique=True)
    users_collection.create_index('role')
    sessions_collection.create_index('token_hash', unique=True)
    sessions_collection.create_index('user_id')
    sessions_collection.create_index('expires_at_dt')
    login_attempts.create_index('throttle_key', unique=True)
    login_attempts.create_index('blocked_until_dt')


def _seed_default_auth_users() -> None:
    if not _get_processor().db or not _get_processor().db.connected:
        return

    users_collection = _get_processor().db.db[AUTH_USERS_COLLECTION]
    if users_collection.count_documents({}) > 0:
        return

    now = datetime.utcnow()
    users_collection.insert_many([
        {
            'name': 'System Admin',
            'email': 'admin@neurodetect.ai',
            'password_hash': _hash_password('admin123'),
            'role': 'admin',
            'created_at': now,
            'updated_at': now,
        },
        {
            'name': 'Fraud Analyst',
            'email': 'analyst@neurodetect.ai',
            'password_hash': _hash_password('analyst123'),
            'role': 'analyst',
            'created_at': now,
            'updated_at': now,
        },
        {
            'name': 'Management Viewer',
            'email': 'viewer@neurodetect.ai',
            'password_hash': _hash_password('viewer123'),
            'role': 'viewer',
            'created_at': now,
            'updated_at': now,
        },
    ])
    logger.info("Seeded default auth users in MongoDB")


def _sanitize_auth_user(user_doc: dict[str, Any]) -> dict[str, str]:
    return {
        'id': str(user_doc.get('_id')),
        'name': str(user_doc.get('name', '')),
        'email': str(user_doc.get('email', '')),
        'role': str(user_doc.get('role', 'viewer')),
    }


def _require_auth_backend() -> tuple[Any, Any]:
    if not _get_processor().db or not _get_processor().db.connected:
        raise HTTPException(status_code=503, detail='MongoDB is not connected')

    db = _get_processor().db.db
    return db[AUTH_USERS_COLLECTION], db[AUTH_SESSIONS_COLLECTION]


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail='Missing authorization header')

    parts = authorization.split(' ', 1)
    if len(parts) != 2 or parts[0].lower() != 'bearer' or not parts[1].strip():
        raise HTTPException(status_code=401, detail='Invalid authorization header')

    return parts[1].strip()


def _parse_session_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, str):
        candidate = value.replace('Z', '+00:00')
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).replace(tzinfo=None)
            return parsed
        except Exception:
            return None
    return None


def _session_has_expired(session: dict[str, Any], now: datetime) -> bool:
    absolute_expiry = _parse_session_datetime(session.get('expires_at_dt') or session.get('expires_at'))
    if absolute_expiry and now >= absolute_expiry:
        return True

    last_seen = _parse_session_datetime(session.get('last_seen_at'))
    if not last_seen:
        last_seen = _parse_session_datetime(session.get('created_at'))
    if not last_seen:
        return False

    idle_deadline = last_seen + timedelta(minutes=AUTH_SESSION_IDLE_TIMEOUT_MINUTES)
    return now >= idle_deadline


def _build_login_throttle_key(email: str, ip_address: str) -> str:
    return f"{email}|{ip_address}"


def _get_login_attempts_collection():
    _require_auth_backend()
    return _get_processor().db.db[AUTH_LOGIN_ATTEMPTS_COLLECTION]


def _remaining_block_seconds(email: str, ip_address: str) -> int:
    attempts_collection = _get_login_attempts_collection()
    key = _build_login_throttle_key(email, ip_address)
    doc = attempts_collection.find_one({'throttle_key': key})
    if not doc:
        return 0

    now = datetime.utcnow()
    blocked_until = _parse_session_datetime(doc.get('blocked_until_dt') or doc.get('blocked_until'))
    if blocked_until and blocked_until > now:
        return max(1, int((blocked_until - now).total_seconds()))

    window_start = _parse_session_datetime(doc.get('window_start_dt') or doc.get('window_start'))
    if window_start and now - window_start > timedelta(minutes=AUTH_LOGIN_WINDOW_MINUTES):
        attempts_collection.delete_one({'throttle_key': key})
        return 0

    return 0


def _register_failed_login(email: str, ip_address: str) -> None:
    attempts_collection = _get_login_attempts_collection()
    key = _build_login_throttle_key(email, ip_address)
    now = datetime.utcnow()
    window_duration = timedelta(minutes=AUTH_LOGIN_WINDOW_MINUTES)

    doc = attempts_collection.find_one({'throttle_key': key})
    if not doc:
        attempts_collection.insert_one({
            'throttle_key': key,
            'email': email,
            'ip_address': ip_address,
            'failed_count': 1,
            'window_start_dt': now,
            'updated_at': now,
        })
        return

    window_start = _parse_session_datetime(doc.get('window_start_dt') or doc.get('window_start'))
    if not window_start or (now - window_start) > window_duration:
        attempts_collection.update_one(
            {'throttle_key': key},
            {
                '$set': {
                    'failed_count': 1,
                    'window_start_dt': now,
                    'blocked_until_dt': None,
                    'updated_at': now,
                }
            },
        )
        return

    failed_count = int(doc.get('failed_count', 0)) + 1
    update_payload: dict[str, Any] = {
        'failed_count': failed_count,
        'updated_at': now,
    }

    if failed_count >= AUTH_LOGIN_MAX_ATTEMPTS:
        update_payload['blocked_until_dt'] = now + timedelta(minutes=AUTH_LOGIN_BLOCK_MINUTES)

    attempts_collection.update_one({'throttle_key': key}, {'$set': update_payload})


def _clear_failed_logins(email: str, ip_address: str) -> None:
    attempts_collection = _get_login_attempts_collection()
    attempts_collection.delete_one({'throttle_key': _build_login_throttle_key(email, ip_address)})


def _resolve_current_user(authorization: Optional[str]) -> dict[str, Any]:
    users_collection, sessions_collection = _require_auth_backend()
    token = _extract_bearer_token(authorization)

    session = sessions_collection.find_one({'token_hash': _token_hash(token), 'is_active': True})
    if not session:
        raise HTTPException(status_code=401, detail='Invalid or expired session')

    now = datetime.utcnow()
    if _session_has_expired(session, now):
        sessions_collection.update_one(
            {'_id': session['_id']},
            {'$set': {'is_active': False, 'expired_at': now}},
        )
        raise HTTPException(status_code=401, detail='Session expired')

    sessions_collection.update_one(
        {'_id': session['_id']},
        {'$set': {'last_seen_at': now}},
    )

    user = users_collection.find_one({'_id': session['user_id']})
    if not user:
        raise HTTPException(status_code=401, detail='Session user not found')

    return user


def _require_admin_user(authorization: Optional[str]) -> dict[str, Any]:
    user = _resolve_current_user(authorization)
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail='Admin role required')
    return user



@router.post("/login")
async def auth_login(payload: AuthLoginRequest, request: Request):
    users_collection, sessions_collection = _require_auth_backend()

    normalized_email = payload.email.strip().lower()
    if not _is_valid_organization_email(normalized_email):
        raise HTTPException(status_code=400, detail='Email must use the format xxx@neurodetect.ai')

    client_ip = request.client.host if request.client and request.client.host else 'unknown'
    blocked_seconds = _remaining_block_seconds(normalized_email, client_ip)
    if blocked_seconds > 0:
        raise HTTPException(
            status_code=429,
            detail=f'Too many failed login attempts. Try again in {blocked_seconds} seconds.',
        )

    user = users_collection.find_one({'email': normalized_email})
    if not user or not _verify_password(payload.password, user.get('password_hash', '')):
        _register_failed_login(normalized_email, client_ip)
        raise HTTPException(status_code=401, detail='Invalid email or password')

    _clear_failed_logins(normalized_email, client_ip)

    token = secrets.token_urlsafe(48)
    now = datetime.utcnow()
    sessions_collection.insert_one({
        'user_id': user['_id'],
        'token_hash': _token_hash(token),
        'is_active': True,
        'created_at': now,
        'last_seen_at': now,
        'expires_at_dt': now + timedelta(minutes=AUTH_SESSION_MAX_AGE_MINUTES),
    })

    return JSONResponse({
        'token': token,
        'user': _sanitize_auth_user(user),
    })


@router.get("/me")
async def auth_me(authorization: Optional[str] = Header(default=None)):
    user = _resolve_current_user(authorization)
    return JSONResponse({'user': _sanitize_auth_user(user)})


@router.post("/logout")
async def auth_logout(authorization: Optional[str] = Header(default=None)):
    _, sessions_collection = _require_auth_backend()
    token = _extract_bearer_token(authorization)
    sessions_collection.update_many(
        {'token_hash': _token_hash(token), 'is_active': True},
        {'$set': {'is_active': False, 'revoked_at': datetime.utcnow()}},
    )
    return JSONResponse({'success': True})


@router.get("/users")
async def auth_get_users(authorization: Optional[str] = Header(default=None)):
    _require_admin_user(authorization)
    users_collection, _ = _require_auth_backend()

    users = list(users_collection.find({}, {'password_hash': 0}).sort('email', 1))
    return JSONResponse({'users': [_sanitize_auth_user(user) for user in users]})


@router.patch("/users/{user_id}/role")
async def auth_update_user_role(user_id: str, payload: RoleUpdateRequest, authorization: Optional[str] = Header(default=None)):
    _require_admin_user(authorization)
    users_collection, _ = _require_auth_backend()

    requested_role = payload.role.strip().lower()
    if requested_role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail='Invalid role')

    try:
        user_object_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid user id')

    users_collection.update_one(
        {'_id': user_object_id},
        {'$set': {'role': requested_role, 'updated_at': datetime.utcnow()}},
    )
    updated_user = users_collection.find_one({'_id': user_object_id}, {'password_hash': 0})

    if not updated_user:
        raise HTTPException(status_code=404, detail='User not found')

    return JSONResponse({'user': _sanitize_auth_user(updated_user)})


@router.post("/users")
async def auth_create_user(payload: UserCreateRequest, authorization: Optional[str] = Header(default=None)):
    _require_admin_user(authorization)
    users_collection, _ = _require_auth_backend()

    name = payload.name.strip()
    email = payload.email.strip().lower()
    password = payload.password.strip()
    requested_role = payload.role.strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail='Name is required')
    if not email:
        raise HTTPException(status_code=400, detail='Email is required')
    if not _is_valid_organization_email(email):
        raise HTTPException(status_code=400, detail='Email must use the format xxx@neurodetect.ai')
    if len(password) < 6:
        raise HTTPException(status_code=400, detail='Password must be at least 6 characters')
    if requested_role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail='Invalid role')

    existing_user = users_collection.find_one({'email': email})
    if existing_user:
        raise HTTPException(status_code=409, detail='A user with this email already exists')

    now = datetime.utcnow()
    insert_result = users_collection.insert_one({
        'name': name,
        'email': email,
        'password_hash': _hash_password(password),
        'role': requested_role,
        'created_at': now,
        'updated_at': now,
    })

    created_user = users_collection.find_one({'_id': insert_result.inserted_id}, {'password_hash': 0})
    if not created_user:
        raise HTTPException(status_code=500, detail='User creation failed')

    return JSONResponse({'user': _sanitize_auth_user(created_user)}, status_code=201)


@router.delete("/users/{user_id}")
async def auth_delete_user(user_id: str, authorization: Optional[str] = Header(default=None)):
    admin_user = _require_admin_user(authorization)
    users_collection, sessions_collection = _require_auth_backend()

    try:
        user_object_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid user id')

    if str(admin_user.get('_id')) == user_id:
        raise HTTPException(status_code=400, detail='Admin users cannot delete their own account')

    existing_user = users_collection.find_one({'_id': user_object_id})
    if not existing_user:
        raise HTTPException(status_code=404, detail='User not found')

    users_collection.delete_one({'_id': user_object_id})
    sessions_collection.update_many(
        {'user_id': user_object_id, 'is_active': True},
        {'$set': {'is_active': False, 'revoked_at': datetime.utcnow()}},
    )

    return JSONResponse({'success': True, 'deleted_user_id': user_id})


