import instructor
from openai import OpenAI
from typing import List, Dict, Any, Optional
from core.models import AtomicFeature, EntityOntology, EntityOntologyList, KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge, Differentiator, DocumentSource
import networkx as nx
from core.config import LLMConfig
from core.prompts import Prompts

class OntologistEngine:
    def __init__(self):
        self.model_name = LLMConfig.get_model_name()
        self.client = LLMConfig.get_client()
        
    def _batch_apply_platonic_ladder(self, features: List[AtomicFeature], text_summary: str, ontology_depth: int) -> List[EntityOntology]:
        features_json = "\\n\\n".join([
            f"Feature Name: {f.name}\\nType: {f.type}\\nDescription: {f.description}" 
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

    def construct_knowledge_graph(self, ontologies: List[EntityOntology]) -> KnowledgeGraph:
        """
        Converts the Differentiable Matrices into a generic KnowledgeGraph structure, backed by NetworkX for graph logic if needed, but outputting standard Pydantic.
        """
        kg = KnowledgeGraph()
        nx_graph = nx.DiGraph()

        for ont in ontologies:
            hierarchy = ont.category.hierarchy
            if not hierarchy:
                continue
                
            leaf_node = hierarchy[-1]
            parent_node = hierarchy[-2] if len(hierarchy) > 1 else "Root"
            
            # Use differentiators as properties
            properties = {diff.name: diff.value for diff in ont.differentiators}
            
            node = KnowledgeGraphNode(id=leaf_node, type=parent_node, properties=properties)
            kg.nodes.append(node)
            
            for i in range(len(hierarchy) - 1):
                child = hierarchy[i+1]
                parent = hierarchy[i]
                kg.edges.append(KnowledgeGraphEdge(source=child, target=parent, relationship="IS_A"))
                
                nx_graph.add_node(child, type=parent, **(properties if i == len(hierarchy)-2 else {}))
                nx_graph.add_node(parent, type="Categorical_Genus")
                nx_graph.add_edge(child, parent, relationship="IS_A")

        # Complex reasoning logic could happen here using nx_graph
        return kg
