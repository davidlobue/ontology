class Prompts:
    """
    Central repository for all system and user prompts used across the Agentic Knowledge Engineering engines.
    Allows for single-source-of-truth management of prompts, easily editable to suit new pipelines.
    """
    
    # ==========================
    # DISTILLATION ENGINE
    # ==========================
    DISTILLATION_SYSTEM = "You are a specialized ontology extraction agent."
    
    @staticmethod
    def get_distillation_user(text_content: str) -> str:
        return f"""
        You are a highly precise Distillation Engine.
        Analyze the following text and extract all meaningful entities, objects, events, relationships, descriptions, tone, and context.
        If the textual embedding space were a connected graph, focus on high density embedding entity node regions as well as those with many edges but do not ignore isolated densities.
        For EVERY entity you extract, you MUST:
        1. Find the exact 'Source Quote' in the text that justifies its existence.
        2. Assign a 'Certainty Score' (0.0 to 1.0).
                
        Focus strictly on minimizing false positives. Do not hallucinate entities not strictly in the text.
        Please provide a comprehensive but concise response, targeting an output length of under roughly 4000 tokens.
        
        Text to analyze:
        {text_content}
        """

    # ==========================
    # ONTOLOGIST ENGINE (DESIGNER)
    # ==========================
    ONTOLOGIST_SYSTEM = "You are a master Ontologist identifying universal templates (Forms) from specific instances to build high density schemas."

    @staticmethod
    def get_ontologist_user(text_summary: str, features_json: str) -> str:
        return f"""
        Given these features, extracted from a text that is summarized as:
        {text_summary}
        
        Extract the elements into the required schema by following these steps for classification:
        1. Distill each feature 'representation' into its fundamental form or type using abstraction.
        2. Ensure each type is distinct (MECE).
        3. Construct the 'hierarchy' as a natural hierarchical chain of abstraction from the broadest category down to the specific sub-type.
        4. Identify unique "Elements" (differentiators) detailing how each is unique from the others.
        Features to Categorize:
        {features_json}
        """
