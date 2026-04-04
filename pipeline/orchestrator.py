from typing import List, Type, Optional
import time
from pydantic import BaseModel
from core.models import DocumentSource, KnowledgeGraph
from distillation.extractor import DistillationEngine
from designer.ontologist import OntologistEngine
from designer.schema_builder import SchemaBuilder

class Orchestrator:
    def __init__(
        self,
        hallucination_filter: bool = True,
        ontology_depth: Optional[int] = None,
        strict_typing: bool = True,
        verbose: bool = False
    ):
        from core.config import LLMConfig
        self.model_name = LLMConfig.get_model_name()
        self.base_url = LLMConfig.get_base_url()
        self.verbose = verbose
        
        # Toggles
        self.hallucination_filter = hallucination_filter
        self.ontology_depth = ontology_depth
        self.strict_typing = strict_typing
        
        print(f"[*] Orchestrator initialized. Model: {self.model_name} | Base URL: {self.base_url}")
        
        # Engines
        self.distillation = DistillationEngine()
        self.ontologist = OntologistEngine()
        self.schema_builder = SchemaBuilder(strict_typing=strict_typing)

    def consolidation_loop(self, documents: List[DocumentSource]) -> KnowledgeGraph:
        """
        Merge extraction across multiple documents.
        Identifies shared 'Forms' and discards document-specific noise.
        """
        print(f"[*] Starting Consolidation Loop over {len(documents)} documents...")
        all_features = []
        for doc in documents:
            start_time = time.time()
            res = self.distillation.extract_features(doc)
            print(f"[TIMER] DistillationEngine.extract_features for doc '{doc.id}' took: {time.time() - start_time:.2f}s")
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

        print("[*] Applying Ontology mapping via clustered batching...")
        start_time = time.time()
        ontologies = self.ontologist.build_concept_matrix(all_features, documents, ontology_depth=self.ontology_depth)
        print(f"[TIMER] OntologistEngine.build_concept_matrix took: {time.time() - start_time:.2f}s")
        
        if self.verbose:
            print("\\n[VERBOSE] Generated Clustered Ontologies:")
            for ont in ontologies:
                print(ont.model_dump_json(indent=2))
        
        print("[*] Constructing shared Knowledge Graph...")
        start_time = time.time()
        master_kg = self.ontologist.construct_knowledge_graph(ontologies)
        print(f"[TIMER] OntologistEngine.construct_knowledge_graph took: {time.time() - start_time:.2f}s")
        
        return master_kg

    def run_pipeline(self, documents: List[DocumentSource]) -> Type[BaseModel]:
        """
        The overarching end-to-end scanner.
        """
        print("====== I. DISTILLATION & II. DESIGNER ======")
        master_kg = self.consolidation_loop(documents)
        
        print("\n====== II. DESIGNER & VISUALIZATION ======")
        start_time = time.time()
        blueprint_schema = self.schema_builder.synthesize_schema(master_kg, schema_name="UniversalBlueprint")
        print(f"[TIMER] SchemaBuilder.synthesize_schema took: {time.time() - start_time:.2f}s")
        print(f"[+] Synthesized Schema: {blueprint_schema.__name__}")
        
        from designer.visualizer import OntologyVisualizer
        output_file = OntologyVisualizer.render_html(master_kg, "knowledge_graph.html")
        print(f"[+] Interactive UI generated at: {output_file}")
        
        # Bypass of prior validation stages over speed optimizations
        print("\n[+] SUCCESS: Produced production-ready Pydantic Schema.")
        return blueprint_schema
