from pydantic import BaseModel, field_validator

class SignUpRequest(BaseModel):
    email: str
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v

class LoginRequest(BaseModel):
    email: str
    password: str
