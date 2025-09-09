import matplotlib.pyplot as plt
import networkx as nx

# Define the workflow nodes
nodes = [
    "plan_research",
    "gather_information",
    "analyze_information",
    "validate_information",
    "refine_research",
    "generate_report",
    "review_report",
    "finalize_report"
]

# Define the edges (including loops)
edges = [
    ("plan_research", "gather_information"),
    ("gather_information", "analyze_information"),
    ("analyze_information", "validate_information"),
    ("validate_information", "refine_research"),      # Conditional loop
    ("validate_information", "generate_report"),      # Conditional continue
    ("refine_research", "gather_information"),        # Loop back
    ("generate_report", "review_report"),
    ("review_report", "generate_report"),             # Conditional regenerate
    ("review_report", "finalize_report"),
]

# Create directed graph
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrowsize=20)
edge_labels = {
    ("validate_information", "refine_research"): "refine",
    ("validate_information", "generate_report"): "continue",
    ("review_report", "generate_report"): "regenerate",
    ("review_report", "finalize_report"): "finalize"
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9)
plt.title("AI Research Agent Workflow")
plt.axis('off')
plt.show()