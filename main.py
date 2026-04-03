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
        model_name=os.getenv("OLLAMA_MODEL", "mistral-small-agent"), 
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        hallucination_filter=True,
        ontology_depth=3,
        strict_typing=True
    )

    doc1 = DocumentSource(
        id="doc_001",
        text_content="""
        On October 4, 2023, Jonathan Doe (CEO) acquired 1,500 Class A Common Shares 
        of Acme Corp at a price of $12.50 per share. The transaction was open market.
        """
    )
    
    doc2 = DocumentSource(
        id="doc_002",
        text_content="""
        Jane Smith, CFO of Acme Corp, sold 400 shares of Class B Common stock 
        on October 5, 2023 at $14.20 per share to cover tax withholding obligations.
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
        print("Note: Ensure Ollama is running (`ollama serve`) and the model 'mistral-small-agent' is pulled (`ollama pull mistral-small-agent`).")

if __name__ == "__main__":
    main()
