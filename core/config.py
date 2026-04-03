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
            # If the user provides a full aiplatform URL or a projects/ endpoint string, 
            # we convert it to the required dedicated endpoint DNS for vLLM compatibility.
            if "aiplatform.googleapis.com" in base_url or "projects/" in base_url:
                try:
                    from google.cloud import aiplatform
                    
                    # Extract just the endpoint path if they provided a full URL
                    endpoint_path = base_url
                    if "https://" in endpoint_path:
                        endpoint_path = endpoint_path.split("aiplatform.googleapis.com/v1/")[1]
                        
                    if endpoint_path.endswith(":predict"):
                        endpoint_path = endpoint_path.replace(":predict", "")
                        
                    # Initialize aiplatform with the service account from core/auth setup
                    get_api_key() # Calling this ensures GOOGLE_APPLICATION_CREDENTIALS is set in os.environ
                    
                    endpoint = aiplatform.Endpoint(endpoint_path)
                    if endpoint.dedicated_endpoint_dns:
                        # Preserve the accurate API version path (/v1/ or /v1beta1/) before the project string
                        api_ver = "v1" if "/v1/" in base_url else "v1beta1"
                        base_url = f"https://{endpoint.dedicated_endpoint_dns}/{api_ver}/{endpoint_path}"
                except Exception as e:
                    print(f"[VERTEX AI WARNING] Could not resolve dedicated endpoint DNS: {e}")
                    
        return base_url

    @staticmethod
    def get_client():
        return instructor.from_openai(
            OpenAI(base_url=LLMConfig.get_base_url(), api_key=get_api_key()),
            mode=instructor.Mode.JSON_SCHEMA
        )
