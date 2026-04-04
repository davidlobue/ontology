from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DocumentSource(BaseModel):
    id: str = Field(description="Unique identifier for the document.")
    text_content: str = Field(description="Raw text content of the document.")

class SourceQuote(BaseModel):
    quote: str = Field(description="The exact quote from the source text.")
    context: str = Field(description="Brief explanation of the context in which this quote appeared.")

class RawTriple(BaseModel):
    subject: str = Field(description="The source entity node.")
    predicate: str = Field(description="The relationship or action linking Subject to Object.")
    object: str = Field(description="The target entity node or literal value.")

class TripleExtractionResult(BaseModel):
    triples: List[RawTriple] = Field(default_factory=list, description="List of extracted open triples.")

class DiscoveryCluster(BaseModel):
    class_name: str = Field(description="The inferred name for this clustered class (e.g. 'Company', 'Person').")
    nodes: List[str] = Field(description="Entity nodes that belong to this cluster.")
    canonical_predicates: List[str] = Field(description="The canonical structural properties (edges) this class exhibits.")
    negative_constraints: List[str] = Field(description="Fields/predicates that definitely do NOT belong to this class.")
