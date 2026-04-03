import instructor
from openai import OpenAI
from typing import List, Dict, Any
from core.models import AtomicFeature, EntityOntology, EntityOntologyList, KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge, Differentiator, DocumentSource
import networkx as nx

class OntologistEngine:
    def __init__(self, model_name: str = "mistral-small-agent", base_url: str = "http://localhost:11434/v1"):
        self.model_name = model_name
        self.base_url = base_url
        self.client = instructor.from_openai(OpenAI(base_url=self.base_url, api_key="ollama"), mode=instructor.Mode.JSON_SCHEMA)
        
    def _batch_apply_platonic_ladder(self, features: List[AtomicFeature], text_summary: str, ontology_depth: int) -> List[EntityOntology]:
        features_json = "\\n\\n".join([
            f"Feature Name: {f.name}\\nType: {f.type}\\nDescription: {f.description}" 
            for f in features
        ])
        
        prompt = f"""
        Given these features, extracted from a text that is summarized as:
        {text_summary}
        
        Categorize them into a shared Platonic hierarchy. 
        Ensure each is distinct (MECE) and assign Genus/Species.
        Find the Genus (broad category) and Species (specific sub-type) for each feature.
        
        Also identify unique "Elements" (Differentiators). How is each unique from the others?
        Depth Limit: {ontology_depth}
        
        Features to Categorize:
        {features_json}
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a master Ontologist identifying universal templates (Forms) from specific instances in grouped clusters."},
                {"role": "user", "content": prompt}
            ],
            response_model=EntityOntologyList,
            max_tokens=8000
        )
        return response.ontologies

    def build_concept_matrix(self, features: List[AtomicFeature], documents: List[DocumentSource], ontology_depth: int = 3) -> List[EntityOntology]:
        """
        Processes a list of Atomic Features into standardized Differentiable Concepts in a clustered batch.
        """
        if not features:
            return []
            
        text_summary = "\\n...\\n".join([doc.text_content.strip() for doc in documents])
        ontologies = self._batch_apply_platonic_ladder(features, text_summary, ontology_depth)
        return ontologies

    def construct_knowledge_graph(self, ontologies: List[EntityOntology]) -> KnowledgeGraph:
        """
        Converts the Differentiable Matrices into a generic KnowledgeGraph structure, backed by NetworkX for graph logic if needed, but outputting standard Pydantic.
        """
        kg = KnowledgeGraph()
        nx_graph = nx.DiGraph()

        for ont in ontologies:
            node_id = ont.category.species  # e.g., 'SEC Form 4 Filing'
            node_type = ont.category.genus  # e.g., 'Transaction'
            
            # Use differentiators as properties
            properties = {diff.name: diff.value for diff in ont.differentiators}
            
            node = KnowledgeGraphNode(id=node_id, type=node_type, properties=properties)
            kg.nodes.append(node)
            
            # In a full implementation, we would extract relationships between THESE ontologies.
            # For now, we link the species to genus as a baseline 'IS_A' edge
            kg.edges.append(KnowledgeGraphEdge(source=node_id, target=node_type, relationship="IS_A"))
            
            nx_graph.add_node(node_id, type=node_type, **properties)
            nx_graph.add_node(node_type, type="Categorical_Genus")
            nx_graph.add_edge(node_id, node_type, relationship="IS_A")

        # Complex reasoning logic could happen here using nx_graph
        return kg
