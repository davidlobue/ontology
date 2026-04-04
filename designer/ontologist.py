import instructor
from openai import OpenAI
from typing import List, Dict, Any, Optional
from core.models import AtomicFeature, EntityOntology, EntityOntologyList, Differentiator, DocumentSource
from core.config import LLMConfig
from core.prompts import Prompts

class OntologistEngine:
    def __init__(self):
        self.model_name = LLMConfig.get_model_name()
        self.client = LLMConfig.get_client()
        
    def _batch_apply_platonic_ladder(self, features: List[AtomicFeature], text_summary: str, ontology_depth: int) -> List[EntityOntology]:
        features_json = "\\n\\n".join([
            f"Feature Name: {f.name}\\nType: {f.type}\\nDescription: {f.description}\\nRelationships: {', '.join([f'({r.relationship_type})->[{r.target_entity}]' for r in f.relationships])}" 
            for f in features
        ])
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": Prompts.ONTOLOGIST_SYSTEM},
                {"role": "user", "content": Prompts.get_ontologist_user(text_summary, features_json)}
            ],
            response_model=EntityOntologyList,
            max_tokens=12000
        )
        return response.ontologies

    def build_concept_matrix(self, features: List[AtomicFeature], documents: List[DocumentSource], ontology_depth: Optional[int] = None) -> List[EntityOntology]:
        """
        Processes a list of Atomic Features into standardized Differentiable Concepts in a clustered batch.
        Builds the hierarchy naturally, then slices to the depth requirement if applied.
        """
        if not features:
            return []
            
        text_summary = "\\n...\\n".join([doc.text_content.strip() for doc in documents])
        ontologies = self._batch_apply_platonic_ladder(features, text_summary, ontology_depth=None)
        
        # Post-generation revision to fit depth bounds
        if ontology_depth is not None:
            for ont in ontologies:
                # Retain the top N levels + the specific leaf node if needed, 
                # or strictly enforce maximum hierarchy length:
                if len(ont.category.hierarchy) > ontology_depth:
                    ont.category.hierarchy = ont.category.hierarchy[:ontology_depth]
                    
        return ontologies
