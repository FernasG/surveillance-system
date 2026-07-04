import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from guard.core.entities import LoginRequest, RegisterRequest
from guard.infrastructure.security import SECRET_KEY, ALGORITHM
from guard.pipeline.auth.auth_service import AuthService

router = APIRouter(tags=["Authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_current_user_token_data(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")
        
        return payload
    except InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")

def require_admin(payload: dict = Depends(get_current_user_token_data)) -> bool:
    if not payload.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Privilégios insuficientes")
    
    return True

def get_auth_service(request: Request) -> AuthService:
    return request.state.auth_service

@router.post("/login")
async def login(request: LoginRequest, auth_service: AuthService = Depends(get_auth_service)):
    token = await auth_service.authenticate_user(request.username, request.password)

    return {"access_token": token, "token_type": "bearer"}

@router.post("/register")
async def register(request: RegisterRequest, is_admin: bool = Depends(require_admin), auth_service: AuthService = Depends(get_auth_service)):
    await auth_service.register_new_user(request.username, request.password, current_user_is_admin=is_admin)
    
    return {"message": "Usuário criado com sucesso"}
