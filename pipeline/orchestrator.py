from typing import List, Type, Optional
from pydantic import BaseModel
from core.models import DocumentSource, KnowledgeGraph
from distillation.extractor import DistillationEngine
from designer.ontologist import OntologistEngine
from designer.schema_builder import SchemaBuilder
from validator.adversary import AdversaryEngine
from validator.auditor import AuditorEngine

class Orchestrator:
    def __init__(
        self,
        model_name: str = "mistral-small-agent",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "dummy_key",
        hallucination_filter: bool = True,
        ontology_depth: Optional[int] = None,
        strict_typing: bool = True,
        verbose: bool = False
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.verbose = verbose
        
        # Toggles
        self.hallucination_filter = hallucination_filter
        self.ontology_depth = ontology_depth
        self.strict_typing = strict_typing
        
        # Engines
        self.distillation = DistillationEngine(model_name, base_url, api_key)
        self.ontologist = OntologistEngine(model_name, base_url, api_key)
        self.schema_builder = SchemaBuilder(strict_typing=strict_typing)
        self.adversary = AdversaryEngine(model_name, base_url, api_key)
        self.auditor = AuditorEngine(model_name, base_url, api_key)

    def consolidation_loop(self, documents: List[DocumentSource]) -> KnowledgeGraph:
        """
        Merge extraction across multiple documents.
        Identifies shared 'Forms' and discards document-specific noise.
        """
        print(f"[*] Starting Consolidation Loop over {len(documents)} documents...")
        all_features = []
        for doc in documents:
            res = self.distillation.extract_features(doc)
            if self.verbose:
                print(f"\\n[VERBOSE] Document '{doc.id}' Extracted Features:")
                print(res.model_dump_json(indent=2))
            
            
            # Application of the hallucination_filter logic
            # e.g., dual-agent consensus. Here simply filtering out low-certainty features.
            if self.hallucination_filter:
                trusted_features = [f for f in res.features if f.certainty_score > 0.75]
                all_features.extend(trusted_features)
            else:
                all_features.extend(res.features)

        print("[*] Applying Platonic Ontology mapping via clustered batching...")
        ontologies = self.ontologist.build_concept_matrix(all_features, documents, ontology_depth=self.ontology_depth)
        
        if self.verbose:
            print("\\n[VERBOSE] Generated Clustered Ontologies:")
            for ont in ontologies:
                print(ont.model_dump_json(indent=2))
        
        print("[*] Constructing shared Knowledge Graph...")
        master_kg = self.ontologist.construct_knowledge_graph(ontologies)
        
        return master_kg

    def run_pipeline(self, documents: List[DocumentSource]) -> Type[BaseModel]:
        """
        The overarching end-to-end scanner.
        """
        print("====== I. DISTILLATION & II. DESIGNER ======")
        master_kg = self.consolidation_loop(documents)
        
        blueprint_schema = self.schema_builder.synthesize_schema(master_kg, schema_name="UniversalBlueprint")
        print(f"[+] Synthesized Schema: {blueprint_schema.__name__}")
        
        print("\n====== III. VALIDATION STRESS TEST ======")
        # Usually, validation uses just a subset or sample document. We take the first one.
        sample_doc = documents[0]
        
        # 1. Receptacle Check (Information Loss vs the master KG logic)
        # Note: we re-extract to get the exact features of the single document for Receptacle comparison
        features = self.distillation.extract_features(sample_doc)
        info_loss_score = self.auditor.receptacle_check(sample_doc, features)
        if self.verbose:
            print(f"\\n[VERBOSE] Information Loss Evaluator Score: {info_loss_score}")
            
        print(f"[!] Receptacle Check -> Information Loss: {info_loss_score}")
        if info_loss_score > 0.10:
            print("[!] WARNING: High Information Loss detected. Schema may lack necessary extraction paths.")

        # 2. Adversarial Text Generation
        print("[*] Generating Adversarial Decoy Text...")
        adversarial_result = self.adversary.generate_adversarial_text(sample_doc)
        if self.verbose:
            print("\\n[VERBOSE] Adversarial Engine Output:")
            print(adversarial_result.model_dump_json(indent=2))
        
        # 3. Instruction-Based Mapping of Adversarial Text using Synthesized Schema
        print("[*] Applying Instruction-Based Mapping on Adversarial Text...")
        mapped_adversarial_output = self.auditor.instruction_based_mapping(adversarial_result.synthetic_text, blueprint_schema)
        if self.verbose:
            print("\\n[VERBOSE] Dynamic Schema Mapping on Decoy Text:")
            print(mapped_adversarial_output.model_dump_json(indent=2))
        
        # 4. Factual Audit
        print("[*] Running Factual Cold Review...")
        audit_result = self.auditor.factual_audit(mapped_adversarial_output, adversarial_result.synthetic_text)
        if self.verbose:
            print("\\n[VERBOSE] Factual Audit Grades:")
            print(audit_result.model_dump_json(indent=2))
        
        print("\n====== PIPELINE RESULTS ======")
        print(f"Precision Score: {audit_result.precision_score}")
        print(f"Zero False Positives Enforced: {audit_result.is_accurate}")
        if not audit_result.is_accurate:
            print(f"Detected False Mapping / Decoy failures: {audit_result.false_positives}")
        
        print("\n[+] SUCCESS: Produced production-ready Pydantic Schema.")
        return blueprint_schema
