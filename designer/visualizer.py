import os
from pyvis.network import Network
from core.models import KnowledgeGraph

class OntologyVisualizer:
    @staticmethod
    def render_html(kg: KnowledgeGraph, output_filename: str = "ontology_graph.html"):
        # We construct a clean, dark-mode PyVis network graph that handles physics beautifully.
        net = Network(notebook=False, directed=True, height="800px", width="100%", bgcolor="#0d1117", font_color="white")
        net.force_atlas_2based()
        
        # Add graph nodes
        for node in kg.nodes:
            # Format title as HTML block for hover data showing Pydantic schema parameters
            title_parts = [f"<b>{node.id}</b>", f"<i>Type: {node.type}</i>"]
            if node.properties:
                title_parts.append("<hr>")
                for k, v in node.properties.items():
                    title_parts.append(f"<b>{k}:</b> {v}")
            title_html = "<br>".join(title_parts)
            
            # Stylize root category nodes differently from literal extracted Forms (species)
            color = "#58a6ff" if node.properties else "#8b949e"
            size = 20 if node.properties else 15
            
            net.add_node(
                node.id, 
                label=node.id, 
                title=title_html, 
                group=node.type,
                color=color,
                size=size
            )
            
        # Add edges
        for edge in kg.edges:
            net.add_edge(
                edge.source, 
                edge.target, 
                title=edge.relationship, 
                color="#30363d"
            )
            
        # Optional: Adds physics tuning slider
        # net.show_buttons(filter_=["physics"])
        
        net.write_html(output_filename)
        return output_filename
