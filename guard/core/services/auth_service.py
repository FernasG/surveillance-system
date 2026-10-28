from typing import Optional
from fastapi import HTTPException, status
from guard.core.entities import User
from guard.core.interfaces import IAuthRepository
from guard.infrastructure.security import verify_password, get_password_hash, create_access_token

MIN_ADMIN_PASSWORD_LENGTH = 8

class AuthService:
    def __init__(self, auth_repo: IAuthRepository):
        self.repo = auth_repo

    async def initialize_admin(self, admin_password: Optional[str]) -> None:
        if not admin_password or admin_password == "admin" or len(admin_password) < MIN_ADMIN_PASSWORD_LENGTH:
            raise ValueError(f"ADMIN_PASSWORD must be set to at least {MIN_ADMIN_PASSWORD_LENGTH} characters and must not be 'admin'")

        admin_user = await self.repo.get_user("admin")

        if admin_user and verify_password(admin_password, admin_user.hashed_password):
            return

        admin_user = User(
            username="admin",
            hashed_password=get_password_hash(admin_password),
            is_admin=True
        )

        await self.repo.save_user(admin_user)

    async def authenticate_user(self, username: str, password: str) -> str:
        user = await self.repo.get_user(username)

        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Usuário ou senha incorretos",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return create_access_token(data={"sub": user.username, "is_admin": user.is_admin})

    async def register_new_user(self, new_username: str, new_password: str, current_user_is_admin: bool) -> None:
        if not current_user_is_admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Apenas administradores podem registrar novos usuários.")
        
        if await self.repo.get_user(new_username):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Usuário já existe.")

        new_user = User(
            username=new_username,
            hashed_password=get_password_hash(new_password),
            is_admin=False
        )

        await self.repo.save_user(new_user)