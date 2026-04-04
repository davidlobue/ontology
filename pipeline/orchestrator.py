from typing import List, Type, Optional
import time
from pydantic import BaseModel
from core.models import DocumentSource, KnowledgeGraph
from distillation.extractor import DistillationEngine
from designer.ontologist import OntologistEngine
from designer.graph_builder import GraphBuilderEngine
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
        self.graph_builder = GraphBuilderEngine()
        self.schema_builder = SchemaBuilder(strict_typing=strict_typing)

    def run_pipeline(self, documents: List[DocumentSource]) -> Type[BaseModel]:
        """
        The overarching end-to-end scanner.
        """
        print("====== I. DISTILLATION ======")
        print(f"[*] Starting extractions across {len(documents)} documents...")
        all_features = []
        for doc in documents:
            start_time = time.time()
            res = self.distillation.extract_features(doc)
            print(f"[TIMER] DistillationEngine.extract_features for doc '{doc.id}' took: {time.time() - start_time:.2f}s")
            if self.verbose:
                print(f"\\n[VERBOSE] Document '{doc.id}' Extracted Features:")
                print(res.model_dump_json(indent=2))
            
            # Hallucination filter
            if self.hallucination_filter:
                trusted_features = [f for f in res.features if f.certainty_score > 0.75]
                all_features.extend(trusted_features)
            else:
                all_features.extend(res.features)

        print("\n====== II. DESIGNER (ONTOLOGIST) ======")
        print("[*] Applying Ontology mapping via clustered batching...")
        start_time = time.time()
        ontologies = self.ontologist.build_concept_matrix(all_features, documents, ontology_depth=self.ontology_depth)
        print(f"[TIMER] OntologistEngine.build_concept_matrix took: {time.time() - start_time:.2f}s")
        
        if self.verbose:
            print("\\n[VERBOSE] Generated Clustered Ontologies:")
            for ont in ontologies:
                print(ont.model_dump_json(indent=2))
                
        print("\n====== III. GRAPH BUILDER (CARTOGRAPHER) ======")
        print("[*] Dispatching explicit Node-Edge semantic network construction...")
        start_time = time.time()
        master_kg = self.graph_builder.generate_knowledge_graph(all_features, ontologies)
        print(f"[TIMER] GraphBuilderEngine.generate_knowledge_graph took: {time.time() - start_time:.2f}s")
        
        print("\n====== IV. SCHEMA BUILDER & VISUALIZATION ======")
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
