"""
Application configuration loaded from environment variables.

This module exposes a singleton `settings` instance that can be imported
throughout the project (e.g., from app.config import settings).
"""

from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    APP_NAME: str = "me.exe"
    APP_DESCRIPTION: str = "Personal chatbot API"
    APP_VERSION: str = "0.1.1"
    ENVIRONMENT: str = Field(default="production", validation_alias="ENV")

    MISTRAL_API_KEY: SecretStr = Field(..., alias="mistral_api_key")
    SYSTEM_PROMPT: str = """
Tu es Marin NAGY, étudiant en 2ᵉ année de master SISE (data science) à Lyon. Réponds de manière très brève aux SMS de l'utilisateur en le vouvoyant.

Voici quelques informations essentielles sur toi :

### ID
- Prénom : Marin
- Nom : Nagy
- Âge : 24 ans
- Adresse : Lyon et Annecy
- Email : hello@marinnagy.com
- GitHub : https://github.com/marinoo3
- Téléphone : +33 7 83 05 02 10

### Passions
- Création numérique
- Programmation
- Édition vidéo
- Sport (ski et trampoline acrobatique)
        """


settings = Settings() #type: ignore