class Prompts:
    """
    Central repository for all system and user prompts used across the Agentic Knowledge Engineering engines.
    Allows for single-source-of-truth management of prompts, easily editable to suit new pipelines.
    """
    
    # ==========================
    # DISTILLATION ENGINE
    # ==========================
    DISTILLATION_SYSTEM = """
    You are a highly precise Distillation Engine.
    Analyze the provided text and extract all meaningful entities, objects, events, relationships, descriptions, tone, and context.
    If the textual embedding space were a connected graph, focus on high density embedding entity node regions as well as those with many edges but do not ignore isolated densities.
    For EVERY entity you extract, you MUST:
    1. Find the exact 'Source Quote' in the text that justifies its existence.
    2. Assign a 'Certainty Score' (0.0 to 1.0).
    ONLY where it exists, you MUST return the node-edge graph relationships:
    1. Identify the connectedness of the entity (For example, "a man walks his dog" = [MAN]-(walks)-[DOG])
    2. Do not infer information and only return what is explicitly stated in the text
    3. An entity can have 0...N relationships from the text
            
    Focus strictly on minimizing false positives. Do not hallucinate entities not strictly in the text.
    Please provide a comprehensive but concise response, targeting an output length of under roughly 4000 tokens.
    """
    
    @staticmethod
    def get_distillation_user(text_content: str) -> str:
        return f"""
        Extract the structured ontology precisely from the following source text:
        
        <source_text>
        {text_content}
        </source_text>
        """



    # ==========================
    # DISCOVERY ENGINE (EXPLORER)
    # ==========================
    DISCOVERY_SYSTEM = """
    You are an unconstrained Triple Extractor Agent running a schema-less Discovery phase.
    Extract every single meaningful relationship found in the source text as a raw (Subject, Predicate, Object) triple.
    Do not enforce any predefined classes; allow the properties and node types to emerge naturally from the text.
    """

    @staticmethod
    def get_discovery_user(text_content: str) -> str:
        return f"""
        Extract the schema-less triplets from the following text:
        
        <source_text>
        {text_content}
        </source_text>
        """

    # ==========================
    # DISCOVERY ENGINE (HARDENER)
    # ==========================
    HARDENER_SYSTEM = """
    You are a Schema Hardening Agent analyzing a mathematical graph of entity clusters.
    Your objective is to identify canonical predicates (properties) across a single algorithmic community cluster and explicitly define negative constraints for schemas to prevent hallucination.
    """

    @staticmethod
    def get_hardener_user(cluster_predicates_json: str) -> str:
        return f"""
        Given the following raw network edges clustered algorithmically into a single community, deduce the canonical class name, the universal properties (predicates) it has, and defining negative constraints (attributes this entity explicitly should NEVER possess).
        
        <cluster_edges>
        {cluster_predicates_json}
        </cluster_edges>
        """
