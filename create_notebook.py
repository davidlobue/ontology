import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Testing the Agentic Knowledge Engineering Tool
This notebook sets up the `Orchestrator` to test the Distillation, Designer, and Validation loops with `instructor` and `openai` against your configured LLM (e.g. Vertex AI or local Ollama).
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
    Patient Assessment - Oct 4, 2023:
    Jordan exhibited highly repetitive physical motions, specifically hand-flapping, 
    when the classroom environment became too loud. He did not engage in verbal communication for 45 minutes.
    \"\"\"
)

doc2 = DocumentSource(
    id="doc_002",
    text_content=\"\"\"
    In-home Observation - Oct 5, 2023:
    During dinner, Jordan successfully maintained eye contact with his sibling when asking for water.
    However, when transitioned to bedtime, severe physical agitation and distress was observed.
    \"\"\"
)

docs = [doc1, doc2]
"""

text_orchestrator = """\
## Initialize Orchestrator and Run Pipeline
We point the Orchestrator to the configured LLM endpoint using generic environment variables (`LLM_BASE_URL`). Note: ensure your endpoint is running or correct credentials are provided."
"""

code_orchestrator = """\
import os
from dotenv import load_dotenv
load_dotenv()

from pipeline.orchestrator import Orchestrator

orchestrator = Orchestrator(
    hallucination_filter=True,
    ontology_depth=None,
    strict_typing=True,
    verbose=True
)

try:
    final_schema = orchestrator.run_pipeline(docs)
except Exception as e:
    print(f"Pipeline Error: {e}")
    print("Is your LLM connection running locally, or did you export LLM_BASE_URL?")
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
