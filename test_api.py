import os
from dotenv import load_dotenv
load_dotenv()
import instructor
from openai import OpenAI
from pydantic import BaseModel
from core.auth import get_api_key

class FeatureExtractionResult(BaseModel):
    feature: str

from core.config import LLMConfig

print(f"Initializing client with {LLMConfig.get_base_url()} and model {LLMConfig.get_model_name()}...")
client = LLMConfig.get_client()

print("Sending request (JSON_SCHEMA mode)...")
try:
    res = client.chat.completions.create(
        model=LLMConfig.get_model_name(),
        messages=[{"role": "user", "content": "Extract a behavior from: John avoided eye contact while speaking to the clinician."}],
        response_model=FeatureExtractionResult,
        max_retries=0
    )
    print("Success:", res)
except Exception as e:
    print("Error:", e)
