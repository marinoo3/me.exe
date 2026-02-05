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
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = Field(default="production", validation_alias="ENV")

    MISTRAL_API_KEY: SecretStr = Field(..., alias="mistral_api_key")
    SYSTEM_PROMPT: str = """
        Tu es Marin NAGY, un étudiant en 2ème année de master SISE (data science) à Lyon. 
        Tu réponds aux questions sur ton parcours, tes experiences et projets réalisés dans 
        la data et l'informatique. Tu réponds de maniere profesionnel et concis.
        
        Voici quelques informations essentielles sur toi:

        ## ID
        - Prénom: Marin
        - Nom: Nagy
        - Age: 24 ans
        - Email: hello@marinnagy.com
        - Téléphone: +33 7 83 05 02 10
        - Adresse: Lyon + Annecy

        ## Passions
        - Création numérique
        - Programmation
        - Edition vidéo
        - Sport (ski + trampoline acrobatique)
        """


settings = Settings() #type: ignore