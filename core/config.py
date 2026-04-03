import os
import instructor
from openai import OpenAI
from core.auth import get_api_key

class LLMConfig:
    """
    Single source of management for LLM environment configurations.
    """
    
    @staticmethod
    def get_provider() -> str:
        return os.getenv("LLM_PROVIDER", "local").lower()

    @staticmethod
    def get_model_name() -> str:
        return os.getenv("LLM_MODEL_NAME", "mistral-small-agent")

    @staticmethod
    def get_base_url() -> str:
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        if LLMConfig.get_provider() in ["google", "vertex", "vertexai", "gcp"]:
            # Prevent conflict with OpenAI SDK automatically appending /chat/completions
            if base_url.endswith(":predict"):
                base_url = base_url.replace(":predict", "")
        return base_url

    @staticmethod
    def get_client():
        return instructor.from_openai(
            OpenAI(base_url=LLMConfig.get_base_url(), api_key=get_api_key()),
            mode=instructor.Mode.JSON_SCHEMA
        )
