# src/auth.py
from fastapi import Depends, HTTPException, status, Request
from jose import jwt
import httpx
import os
from jose import jwt, jwk
from jose.utils import base64url_decode
from dotenv import load_dotenv
load_dotenv()
SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID")  # Ex: abcdefghijklmnop
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")  # Trouvable dans Supabase > Settings > API > JWT secret

from fastapi import HTTPException, status, Request
from jose import jwt, JWTError
import os
from dotenv import load_dotenv

load_dotenv()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

async def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token manquant")

    token = auth_header.split(" ")[1]

    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
    except JWTError as e:
        print(f"JWT Error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")

    return payload
