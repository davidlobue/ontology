import instructor
from openai import OpenAI
from typing import List, Optional
from core.models import DocumentSource, FeatureExtractionResult
from core.config import LLMConfig
from core.prompts import Prompts

class DistillationEngine:
    def __init__(self):
        self.model_name = LLMConfig.get_model_name()
        self.client = LLMConfig.get_client()

    def extract_features(self, document: DocumentSource) -> FeatureExtractionResult:
        """
        Uses Instructor to extract heavily structured Atomic Features from text.
        Forces the LLM to ground entities with Source Quotes and Certainty Scores.
        """
        extraction = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": Prompts.DISTILLATION_SYSTEM,
                },
                {
                    "role": "user",
                    "content": Prompts.get_distillation_user(document.text_content),
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

