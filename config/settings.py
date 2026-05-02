"""
Central settings — loaded from environment via pydantic-settings.
Import: from config.settings import settings
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ariran"
    db_user: str = "ariran_pipeline"
    db_password: str = Field("", description="PostGIS DB password")

    @property
    def db_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def db_url_psycopg2(self) -> str:
        return (
            f"host={self.db_host} port={self.db_port} dbname={self.db_name} "
            f"user={self.db_user} password={self.db_password}"
        )

    # ACLED
    acled_api_key: str = ""
    acled_email: str = ""
    acled_password: str = ""
    acled_base_url: str = "https://acleddata.com/api/acled/read"

    # Social
    twitter_api_key: str = ""

    # HTTP
    user_agent: str = "ariran-Bot/1.0"
    request_timeout: int = 30
    request_retries: int = 3

    # Pipeline
    staging_batch_size: int = 500
    log_level: str = "INFO"


settings = Settings()  # type: ignore[call-arg]
