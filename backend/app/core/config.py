from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    database_url: str
    supabase_url: str
    supabase_service_key: str
    supabase_bucket: str = "docusense-files"
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str
    secret_key: str = "changeme"
    environment: str = "development"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
