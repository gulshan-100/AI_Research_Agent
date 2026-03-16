import matplotlib.pyplot as plt
import networkx as nx

def create_research_workflow_graph():
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    nodes = [
        "plan_research",
        "gather_information",
        "analyze_information",
        "decide_next_step",
        "refine_search",
        "deep_dive_subtopic",
        "generate_report",
        "END"
    ]

    # Add all nodes
    G.add_nodes_from(nodes)

    # Add edges
    edges = [
        ("plan_research", "gather_information"),
        ("gather_information", "analyze_information"),
        ("analyze_information", "decide_next_step"),
        ("decide_next_step", "refine_search"),
        ("decide_next_step", "deep_dive_subtopic"),
        ("decide_next_step", "generate_report"),
        ("refine_search", "gather_information"),
        ("deep_dive_subtopic", "gather_information"),
        ("generate_report", "END")
    ]

    # Add all edges
    G.add_edges_from(edges)

    # Set up the plot
    plt.figure(figsize=(15, 10))
    
    # Define node colors based on their role
    node_colors = {
        "plan_research": "#4CAF50",  # Green for start
        "gather_information": "#2196F3",  # Blue for information gathering
        "analyze_information": "#9C27B0",  # Purple for analysis
        "decide_next_step": "#FFC107",  # Yellow for decision
        "refine_search": "#FF5722",  # Orange for refinement
        "deep_dive_subtopic": "#E91E63",  # Pink for deep dive
        "generate_report": "#795548",  # Brown for report
        "END": "#F44336"  # Red for end
    }

    # Create the layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw the graph
    nx.draw_networkx_nodes(G, pos, 
                          node_color=[node_colors[node] for node in G.nodes()],
                          node_size=3000,
                          alpha=0.7)
    
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True, 
                          arrowsize=20)
    
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_weight='bold',
                           font_color='white')

    # Add a title
    plt.title("Research Agent Workflow Graph", 
              fontsize=16, 
              fontweight='bold',
              pad=20)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=node,
                                 markersize=10)
                      for node, color in node_colors.items()]
    
    plt.legend(handles=legend_elements, 
              loc='center left', 
              bbox_to_anchor=(1, 0.5))

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig('research_workflow_graph.png', 
                dpi=300, 
                bbox_inches='tight')
    
    print("Graph visualization saved as 'research_workflow_graph.png'")

if __name__ == "__main__":
    create_research_workflow_graph()
