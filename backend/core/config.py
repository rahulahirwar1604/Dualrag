from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # App
    # ------------------------------------------------------------------
    DEBUG: bool = Field(default=False)

    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------
    GOOGLE_API_KEY: str = Field(default="")

    # LLM
    LLM_MODEL: str = Field(default="gemini-2.5-flash")
    LLM_TEMPERATURE: float = Field(default=0.2)
    LLM_MAX_TOKENS: int = Field(default=2048)

    # Embeddings
    EMBEDDING_MODEL: str = Field(default="text-embedding-004")
    EMBEDDING_DIMENSIONS: int = Field(default=768)

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------
    QDRANT_URL: str = Field(default="")
    QDRANT_API_KEY: str = Field(default="")
    QDRANT_COLLECTION: str = Field(default="dualrag_documents")

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    DOCUMENT_STORE_PATH: str = Field(
        default="storage/documents.json"
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    CORS_ORIGINS: str = Field(
        default="http://localhost:5173"
    )

    @property
    def cors_origins_list(self):
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()