from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    supabase_url: str
    supabase_key: str
    database_url: str                        # direct psycopg2 — for bulk pipeline ops
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4.1-mini"
    cluster_k: int = 27
    umap_components: int = 5
    batch_size: int = 100

    model_config = {"env_file": ".env"}


settings = Settings()
