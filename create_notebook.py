import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Testing the Agentic Knowledge Engineering Tool
This notebook sets up the `Orchestrator` to test the Distillation, Designer, and Validation loops with `instructor` and `openai` against local Ollama models.
"""

code_imports = """\
import sys
import os
# Allow imports from local directory if running inside the ontology folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from core.models import DocumentSource
from pipeline.orchestrator import Orchestrator
"""

text_docs = """\
## Define Test Documents
Here we define the input raw texts that need to be distilled into a Knowledge Graph and synthesized into a strict Pydantic schema.
"""

code_docs = """\
doc1 = DocumentSource(
    id="doc_001",
    text_content=\"\"\"
    On October 4, 2023, Jonathan Doe (CEO) acquired 1,500 Class A Common Shares 
    of Acme Corp at a price of $12.50 per share. The transaction was open market.
    \"\"\"
)

doc2 = DocumentSource(
    id="doc_002",
    text_content=\"\"\"
    Jane Smith, CFO of Acme Corp, sold 400 shares of Class B Common stock 
    on October 5, 2023 at $14.20 per share to cover tax withholding obligations.
    \"\"\"
)

docs = [doc1, doc2]
"""

text_orchestrator = """\
## Initialize Orchestrator and Run Pipeline
We point the Orchestrator to the local Ollama instance (acting as an OpenAI compatible endpoint `http://localhost:11434/v1`). Note: ensure `ollama serve` is running and `mistral-small-agent` is pulled."
"""

code_orchestrator = """\
import os
from dotenv import load_dotenv
load_dotenv()

orchestrator = Orchestrator(
    model_name=os.getenv("OLLAMA_MODEL", "mistral-small-agent"), 
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    hallucination_filter=True,
    ontology_depth=3,
    strict_typing=True
)

try:
    final_schema = orchestrator.run_pipeline(docs)
except Exception as e:
    print(f"Pipeline Error: {e}")
    print("Is your Ollama instance running locally, or did you export OLLAMA_BASE_URL?")
    final_schema = None
"""

text_schema = """\
## Synthesized Schema Output
If the validation passes strictly with zero false positives, we can now view the dynamic schema produced.
"""

code_schema = """\
if final_schema:
    print("Final Dynamic Schema JSON Definition:\\n")
    import json
    print(json.dumps(final_schema.model_json_schema(), indent=2))
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_docs),
    nbf.v4.new_code_cell(code_docs),
    nbf.v4.new_markdown_cell(text_orchestrator),
    nbf.v4.new_code_cell(code_orchestrator),
    nbf.v4.new_markdown_cell(text_schema),
    nbf.v4.new_code_cell(code_schema)
]

with open('test_ontology.ipynb', 'w') as f:
    nbf.write(nb, f)
