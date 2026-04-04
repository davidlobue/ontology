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
        (For example: If the text is a clinical behavioral report, extract specific behaviors, interactions, and settings. If it is corporate, extract transactions).
        For EVERY entity you extract, you MUST:
        1. Find the exact 'Source Quote' in the text that justifies its existence.
        2. Assign a 'Certainty Score' (0.0 to 1.0).
        3. Differentiate between the 'shadow' (how it appeared in the text) and the 'form' (its general meaning).
        
        Focus strictly on minimizing false positives. Do not hallucinate entities not strictly in the text.
        Please provide a comprehensive but concise response, targeting an output length of under roughly 4000 tokens.
        
        Text to analyze:
        {text_content}
        """

    # ==========================
    # ONTOLOGIST ENGINE (DESIGNER)
    # ==========================
    ONTOLOGIST_SYSTEM = "You are a master Ontologist identifying universal templates (Forms) from specific instances in grouped clusters."

    @staticmethod
    def get_ontologist_user(text_summary: str, features_json: str) -> str:
        return f"""
        Given these features, extracted from a text that is summarized as:
        {text_summary}
        
        Categorize them into a shared Platonic hierarchy. 
        Ensure each is distinct (MECE).
        Construct the natural hierarchical chain of abstraction from the broadest category down to the specific sub-type.
        (For example, if evaluating clinical behavior: ['Behavior', 'Social Interaction', 'Direct Contact', 'Avoids Eye Contact']).
        
        Also identify unique "Elements" (Differentiators). How is each unique from the others?
        Please provide a comprehensive response but target an output length under roughly 8000 tokens to ensure stability.
        
        Features to Categorize:
        {features_json}
        """
