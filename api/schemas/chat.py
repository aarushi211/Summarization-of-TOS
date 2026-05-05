from pydantic import BaseModel, field_validator

class ChatRequest(BaseModel):
    query: str
    document_id: str
    service_name: str = "Unknown Service"

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Query too long (max 2000 chars).")
        return v
