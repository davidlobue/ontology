import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Type, Any
from core.models import DocumentSource, FeatureExtractionResult

class FactualAuditResult(BaseModel):
    is_accurate: bool = Field(description="Whether the mapping contains zero false positives.")
    false_positives: list[str] = Field(description="List of fields where the schema hallucinated or incorrectly mapped a decoy.")
    precision_score: float = Field(description="Precision score from 0.0 to 1.0 (accuracy priority).")
    unmapped_information_loss_score: float = Field(description="Information loss (Receptacle Check) from 0.0 to 1.0. Lower is better.")

class AuditorEngine:
    def __init__(self, model_name: str = "mistral-small-agent", base_url: str = "http://localhost:11434/v1", api_key: str = "dummy_key"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = instructor.from_openai(OpenAI(base_url=self.base_url, api_key=self.api_key), mode=instructor.Mode.JSON_SCHEMA)

    def receptacle_check(self, original_doc: DocumentSource, features: FeatureExtractionResult) -> float:
        """
        Compare the extracted features against the original text to identify "Unmapped Data" (Information Loss).
        Returns a score representing the percentage of lost critical data (0.0 means perfect coverage).
        Trigger a feedback loop if score > threshold (e.g., > 0.10).
        """
        # For a full implementation, we ask the LLM to rate the % of critical data missing.
        prompt = f"""
        Evaluate if any major points from the Original Text are MISSING in the Extracted Features.
        Rate the 'Information Loss' from 0.0 (nothing missing) to 1.0 (everything missing).
        
        Original: {original_doc.text_content}
        Extracted: {[f.name for f in features.features]}
        """
        
        class LossEvaluation(BaseModel):
            loss_score: float = Field(..., ge=0.0, le=1.0)
            
        eval_result = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You calculate information loss."},
                {"role": "user", "content": prompt}
            ],
            response_model=LossEvaluation
        )
        return eval_result.loss_score

    def instruction_based_mapping(self, text: str, schema: Type[BaseModel]) -> BaseModel:
        """
        Uses Instructor + Pydantic to map against a given dynamically generated schema.
        """
        result = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Extract information exactly matching the provided JSON schema. Ensure strict accuracy. Please maintain your output under roughly 4000 tokens if possible."},
                {"role": "user", "content": text}
            ],
            response_model=schema,
            max_tokens=8000
        )
        return result

    def factual_audit(self, mapped_output: BaseModel, adversarial_document: str) -> FactualAuditResult:
        """
        Performs a 'Cold Review' of the schema's extraction against the Adversarial Text.
        Focuses heavily on Precision (Zero False Positives).
        """
        prompt = f"""
        Perform a Factual Cold Review on the mapped output derived from the adversarial text.
        Did the extraction model hallucinate any matching data? Did it get tricked by decoys?
        
        Adversarial Text:
        {adversarial_document}
        
        Mapped Output:
        {mapped_output.model_dump_json(indent=2)}
        """
        
        audit_result = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a strict QA auditor enforcing Zero False Positives."},
                {"role": "user", "content": prompt}
            ],
            response_model=FactualAuditResult
        )
        return audit_result
