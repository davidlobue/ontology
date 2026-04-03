from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class SourceQuote(BaseModel):
    quote: str = Field(description="The exact quote from the source text.")
    context: str = Field(description="Brief explanation of the context in which this quote appeared.")

class AtomicFeature(BaseModel):
    name: str = Field(description="The distinct name of the identified entity, object, event, or relationship.")
    type: str = Field(description="Categorization of the feature (e.g., Person, Organization, Event, Tone).")
    description: str = Field(description="Detailed explanation of the feature.")
    source_grounding: SourceQuote = Field(description="Direct evidence from the text.")
    certainty_score: float = Field(description="Confidence score from 0.0 to 1.0 that this feature actually meant what was extracted.")
    shadow_vs_form: str = Field(description="Explanation of how the entity appears in this text (shadow) versus its general representation (form).")

class FeatureExtractionResult(BaseModel):
    features: List[AtomicFeature] = Field(default_factory=list, description="Extracted atomic features from the source text.")

class PlatonicCategory(BaseModel):
    genus: str = Field(description="The broader category (e.g., Transaction, Animal).")
    species: str = Field(description="The specific sub-type (e.g., SEC Form 4 Filing, Dog).")

class Differentiator(BaseModel):
    name: str = Field(description="The unique distinguishing trait.")
    value: Any = Field(description="The specific value of this differentiator that makes it Unique across similar entities.")

class EntityOntology(BaseModel):
    feature_name: str
    category: PlatonicCategory
    differentiators: List[Differentiator]

class EntityOntologyList(BaseModel):
    ontologies: List[EntityOntology] = Field(description="The categorized grouped ontologies ensuring MECE properties.")

class KnowledgeGraphNode(BaseModel):
    id: str = Field(description="Unique identifier for the node (usually the form/species).")
    type: str = Field(description="The generic category Type (Genus).")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Metadata properties derived from differentiators.")

class KnowledgeGraphEdge(BaseModel):
    source: str = Field(description="Source node ID.")
    target: str = Field(description="Target node ID.")
    relationship: str = Field(description="How the source relates to the target.")

class KnowledgeGraph(BaseModel):
    nodes: List[KnowledgeGraphNode] = Field(default_factory=list)
    edges: List[KnowledgeGraphEdge] = Field(default_factory=list)

class DocumentSource(BaseModel):
    id: str
    text_content: str
