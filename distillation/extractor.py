import instructor
from openai import OpenAI
from typing import List, Optional
from core.models import DocumentSource, FeatureExtractionResult
from core.config import LLMConfig

class DistillationEngine:
    def __init__(self):
        self.model_name = LLMConfig.get_model_name()
        self.client = LLMConfig.get_client()

    def extract_features(self, document: DocumentSource) -> FeatureExtractionResult:
        """
        Uses Instructor to extract heavily structured Atomic Features from text.
        Forces the LLM to ground entities with Source Quotes and Certainty Scores.
        """
        prompt = f"""
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
        {document.text_content}
        """

        extraction = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a specialized ontology extraction agent.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=FeatureExtractionResult,
            max_tokens=8000
        )
        return extraction

    def multi_source_review(self, documents: List[DocumentSource]) -> List[FeatureExtractionResult]:
        """
        Processes multiple documents through the extraction engine.
        """
        results = []
        for doc in documents:
            results.append(self.extract_features(doc))
        return results

if __name__ == "__main__":
    # Quick test harness
    engine = DistillationEngine(model_name="mistral-small-agent")
    doc = DocumentSource(id="doc_1", text_content="The patient displayed intense focus while stacking blocks for 2 hours, avoiding all eye contact with the clinician.")
    # Assuming local ollama is running or mocked
    try:
        res = engine.extract_features(doc)
        print(f"Extracted {len(res.features)} features.")
    except Exception as e:
        print(f"Error (likely no local LLM running): {e}")

