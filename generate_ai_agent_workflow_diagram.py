import matplotlib.pyplot as plt
import networkx as nx

def generate_ai_agent_workflow_diagram(output_file="ai_agent_workflow_diagram.png"):
    """
    Generates a simple graph diagram to visualize the workflow of the AI Research Agent using matplotlib and networkx.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Define nodes for each step in the workflow
    G.add_node("Plan Research")
    G.add_node("Gather Information")
    G.add_node("Analyze Information")
    G.add_node("Validate Information")
    G.add_node("Refine Research")
    G.add_node("Generate Report")
    G.add_node("Review Report")
    G.add_node("Finalize Report")

    # Define edges to represent the workflow
    G.add_edge("Plan Research", "Gather Information")
    G.add_edge("Gather Information", "Analyze Information")
    G.add_edge("Analyze Information", "Validate Information")

    # Conditional edge: Validate -> Refine or Generate
    G.add_edge("Validate Information", "Refine Research", label="If Quality Low")
    G.add_edge("Validate Information", "Generate Report", label="If Quality High")

    # Loop back to gather information after refinement
    G.add_edge("Refine Research", "Gather Information")

    # Continue to review after generating the report
    G.add_edge("Generate Report", "Review Report")

    # Conditional edge: Review -> Regenerate or Finalize
    G.add_edge("Review Report", "Generate Report", label="If Quality Low")
    G.add_edge("Review Report", "Finalize Report", label="If Quality High")

    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, font_weight="bold", arrowsize=20)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Save the diagram to a file
    plt.savefig(output_file)
    plt.close()

# Generate the workflow diagram
generate_ai_agent_workflow_diagram()
