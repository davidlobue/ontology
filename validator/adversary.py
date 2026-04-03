import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from core.models import DocumentSource
from core.config import LLMConfig

class SyntheticTextResult(BaseModel):
    synthetic_text: str = Field(description="The generated adversarial text.")
    decoy_points: list[str] = Field(description="List of near-neighbor decoys included to test boundaries.")

class AdversaryEngine:
    """
    Generates completely different, synthetic text input that mimics the style and narrative 
    of the original but includes "Near-Neighbor" decoys.
    (e.g., creating a report about 'Toddler Temper Tantrums' to test an 'Autism Spectrum Disorder' schema)
    """
    def __init__(self):
        self.model_name = LLMConfig.get_model_name()
        self.client = LLMConfig.get_client()

    def generate_adversarial_text(self, original_document: DocumentSource) -> SyntheticTextResult:
        prompt = f"""
        Analyze the following original text. 
        Then, generate a completely new, synthetic text that mimics the exact style, structure, and tone.
        However, sharply shift the actual domain/subject matter to a "Near-Neighbor".
        Include specific 'decoy' entities that look similar to the original domain but are fundamentally different.
        Please keep the response concise, targeting an output length under roughly 2000 tokens.
        
        Original Text:
        {original_document.text_content}
        """

        adversarial_result = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are the Adversary. Your job is to generate deceptive texts to break ontology schemas."
                },
                {"role": "user", "content": prompt}
            ],
            response_model=SyntheticTextResult,
            max_tokens=5000
        )
        return adversarial_result

if __name__ == "__main__":
    adv = AdversaryEngine("mistral-small-agent")
    doc = DocumentSource(id="test", text_content="The 6-year-old child resisted transitions between classroom activities, crying heavily for 15 minutes and requiring physical comforting from the teacher.")
    try:
        res = adv.generate_adversarial_text(doc)
        print("Generated text:", res.synthetic_text)
        print("Decoys:", res.decoy_points)
    except Exception as e:
        print("LLM Error:", e)
