import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

k_e = 1.0  # Electrostatic constant
k_s = 1.0  # Spring constant
L = 1.0    # Equilibrium length of springs
num_vertices = 15

G = nx.random_geometric_graph(num_vertices, radius=0.4)
positions = np.random.rand(num_vertices, 2)

def energy(positions, G):
    """Calculate total energy of the system."""
    electrostatic_energy = 0.0
    spring_energy = 0.0
    
    # Calculate electrostatic energy
    for (i, j) in G.edges():
        r_ij = np.linalg.norm(positions[i] - positions[j])
        if r_ij > 0:
            electrostatic_energy += k_e / r_ij

    # Calculate spring potential energy
    for (i, j) in G.edges():
        d_ij = np.linalg.norm(positions[i] - positions[j])
        spring_energy += 0.5 * k_s * (d_ij - L) ** 2
    return electrostatic_energy + spring_energy

def simulated_annealing(positions, G, initial_temp=1.0, final_temp=0.01, alpha=0.99, max_iter=1000):
    """Applying simulated annealing to find the optimal configuration."""
    temp = initial_temp
    best_positions = positions.copy()
    best_energy = energy(best_positions, G)

    # Store the positions over time for animation
    positions_history = [best_positions.copy()]
    for _ in range(max_iter):
        new_positions = best_positions + (np.random.rand(num_vertices, 2) - 0.5) * 0.1
        new_energy = energy(new_positions, G)

        if new_energy < best_energy or np.random.rand() < np.exp((best_energy - new_energy) / temp):
            best_positions = new_positions
            best_energy = new_energy

        positions_history.append(best_positions.copy())
        temp *= alpha

    return positions_history

fig, ax = plt.subplots()
sc = ax.scatter(positions[:, 0], positions[:, 1])
nx.draw(G, positions, ax=ax, with_labels=True)

positions_history = simulated_annealing(positions, G)

def update(frame):
    ax.clear()
    current_positions = positions_history[frame]
    sc = ax.scatter(current_positions[:, 0], current_positions[:, 1])
    nx.draw(G, current_positions, ax=ax, with_labels=True)
    ax.set_title(f"Frame {frame}/{len(positions_history) - 1}")
    return sc,

ani = FuncAnimation(fig, update, frames=len(positions_history), blit=False, interval=100)

plt.show()
