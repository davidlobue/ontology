import os
from dotenv import load_dotenv
load_dotenv()
import instructor
from openai import OpenAI
from pydantic import BaseModel

class FeatureExtractionResult(BaseModel):
    feature: str

base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
model = os.getenv("OLLAMA_MODEL", "mistral-small-agent")

print(f"Initializing client with {base_url} and model {model}...")
client = instructor.from_openai(
    OpenAI(base_url=base_url, api_key="ollama"),
    mode=instructor.Mode.JSON_SCHEMA
)

print("Sending request (JSON_SCHEMA mode)...")
try:
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Extract a feature from: The dog is brown."}],
        response_model=FeatureExtractionResult,
        max_retries=0
    )
    print("Success:", res)
except Exception as e:
    print("Error:", e)
