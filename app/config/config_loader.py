from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfigLoader(BaseSettings):
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    MODEL_NAME: str
    HOST: str
    PORT: int
    
    model_config = SettingsConfigDict(
        env_file = "app/config/dev.env",
        extra = "ignore"
    )

settings = ConfigLoader()

logger.info("Configuration loaded successfully.")
logger.success(f"Google API Key: {settings.GOOGLE_API_KEY[:4]}****")
logger.success(f"Pinecone API Key: {settings.PINECONE_API_KEY[:4]}****")