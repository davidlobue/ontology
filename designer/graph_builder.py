import instructor
from typing import List
from core.models import AtomicFeature, EntityOntology, KnowledgeGraph
from core.config import LLMConfig
from core.prompts import Prompts

class GraphBuilderEngine:
    """
    Agentic Knowledge Engineering component explicitly tasked with bridging atomic textual features 
    and formal hierarchical abstractions into a pure logical Graph topology.
    """
    def __init__(self):
        self.model_name = LLMConfig.get_model_name()
        self.client = LLMConfig.get_client()

    def generate_knowledge_graph(self, features: List[AtomicFeature], ontologies: List[EntityOntology]) -> KnowledgeGraph:
        """
        Takes raw features and abstracted categorical ladders to construct a pure JSON Pydantic KnowledgeGraph.
        """
        # Convert objects natively to strings for LLM injection
        features_json = "\\n\\n".join([f.model_dump_json() for f in features])
        ontologies_json = "\\n\\n".join([o.model_dump_json() for o in ontologies])
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": Prompts.CARTOGRAPHER_SYSTEM},
                {"role": "user", "content": Prompts.get_cartographer_user(features_json, ontologies_json)}
            ],
            response_model=KnowledgeGraph,
            max_tokens=16000
        )
        return response
