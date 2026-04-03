import os
from dotenv import load_dotenv
load_dotenv()
import instructor
from openai import OpenAI
from pydantic import BaseModel

class FeatureExtractionResult(BaseModel):
    feature: str

base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
model = os.getenv("LLM_MODEL_NAME", "mistral-small-agent")
api_key = os.getenv("LLM_API_KEY", "dummy_key")

print(f"Initializing client with {base_url} and model {model}...")
client = instructor.from_openai(
    OpenAI(base_url=base_url, api_key=api_key),
    mode=instructor.Mode.JSON_SCHEMA
)

print("Sending request (JSON_SCHEMA mode)...")
try:
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Extract a behavior from: John avoided eye contact while speaking to the clinician."}],
        response_model=FeatureExtractionResult,
        max_retries=0
    )
    print("Success:", res)
except Exception as e:
    print("Error:", e)
