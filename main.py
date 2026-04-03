import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Add the parent directory to sys path so imports work perfectly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import DocumentSource
from pipeline.orchestrator import Orchestrator

def main():
    print("Initializing Agentic Knowledge Engineering Tool...")
    # Use standard models for pure conceptual run without local ollama blocking,
    # or you can change to a real model like "mistral-small-agent"
    
    # We will use openai format but with a mock or base url if needed.
    # The user requested 'local hosted llms'.
    
    orchestrator = Orchestrator(
        model_name=os.getenv("LLM_MODEL_NAME", "mistral-small-agent"), 
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "dummy_key"),
        hallucination_filter=True,
        ontology_depth=None,
        strict_typing=True,
        verbose=True
    )

    doc1 = DocumentSource(
        id="doc_001",
        text_content="""
        Patient Assessment - Oct 4, 2023:
        Jordan exhibited highly repetitive physical motions, specifically hand-flapping, 
        when the classroom environment became too loud. He did not engage in verbal communication for 45 minutes.
        """
    )
    
    doc2 = DocumentSource(
        id="doc_002",
        text_content="""
        In-home Observation - Oct 5, 2023:
        During dinner, Jordan successfully maintained eye contact with his sibling when asking for water.
        However, when transitioned to bedtime, severe physical agitation and distress was observed.
        """
    )
    
    docs = [doc1, doc2]

    try:
        final_schema = orchestrator.run_pipeline(docs)
        print("\nFinal Dynamic Schema JSON Definition:")
        print(final_schema.model_json_schema())
    except Exception as e:
        print("\n[ERROR] Pipeline execution failed.")
        print(f"Details: {e}")
        print("Note: Ensure your LLM connection is running and variables (LLM_BASE_URL, LLM_MODEL_NAME) are correctly set.")

if __name__ == "__main__":
    main()
