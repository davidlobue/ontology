import os

def get_api_key() -> str:
    """
    Returns the appropriate API key based on the selected LLM provider.
    Automatically handles Google Cloud Vertex AI authorization if enabled.
    """
    provider = os.getenv("LLM_PROVIDER", "local").lower()
    
    if provider in ["google", "vertex", "vertexai", "gcp"]:
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            
            cred_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not cred_file:
                cred_file = os.path.join(os.getcwd(), "knowledgeontology-1c9b2932ef2d.json")
                if os.path.exists(cred_file):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_file
                else:
                    print(f"[AUTH WARNING] Google Provider selected but {cred_file} not found.")
                    return "dummy_key"
            
            SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
            
            credentials = service_account.Credentials.from_service_account_file(
                cred_file, 
                scopes=SCOPES
            )
            credentials.refresh(Request())
            return credentials.token
        except ImportError:
            print("[AUTH WARNING] `google-auth` is not installed. Run `pip install google-auth`.")
            return "dummy_key"
        except Exception as e:
            print(f"[AUTH WARNING] Could not fetch Google Auth token: {e}")
            return "dummy_key"
            
    # Default for local or standard static API keys
    return os.getenv("LLM_API_KEY", "dummy_key")
