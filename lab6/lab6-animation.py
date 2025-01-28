import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
# # Ex2

def generate_ba_graph_animation(n, m):
    G = nx.Graph()
    positions = {}
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(i, j)
    positions.update(nx.spring_layout(G, seed=42))  # Initial positions
    
    degrees = [0] * n
    for u, v in G.edges():
        degrees[u] += 1
        degrees[v] += 1
    
    def update(frame):
        if frame >= len(G.nodes):  # Stop updating if all nodes are added
            return
        
        new_node = len(G.nodes)
        targets = set()
        while len(targets) < m:
            # Choose a target based on preferential attachment
            potential_target = random.choices(
                population=list(G.nodes),
                weights=[degrees[node] for node in G.nodes],
                k=1
            )[0]
            targets.add(potential_target)
        
        for target in targets:
            G.add_edge(new_node, target)
            degrees[new_node] += 1
            degrees[target] += 1
        
        positions.update(nx.spring_layout(G, seed=42, pos=positions, iterations=10))
        
        ax.clear()
        nx.draw(
            G, pos=positions, ax=ax, node_size=30, edge_color="gray", node_color="blue"
        )
        ax.set_title(f"Frame {frame + 1}/{n}: Nodes={len(G.nodes)} Edges={len(G.edges)}")
    
    ani = animation.FuncAnimation(fig, update, frames=n, repeat=False)
    return ani

ani = generate_ba_graph_animation(60, 3) 
plt.show()
