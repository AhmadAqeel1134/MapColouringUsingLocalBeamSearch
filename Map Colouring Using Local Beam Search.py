import sys
import heapq
import csv
from collections import defaultdict

def parse_hypercube_dataset(filename):
    print("\nParsing dataset...")
    adjacency = defaultdict(set)
    edge_count = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            source, dest = int(parts[0]), int(parts[1])
            adjacency[source].add(dest)
            adjacency[dest].add(source)
            edge_count += 1
    print(f"Parsed {len(adjacency)} vertices and {edge_count} edges")
    return adjacency

def precompute_two_hop_neighbors(adjacency):
    print("\nCalculating two-hop neighbors...")
    two_hop = defaultdict(set)
    for u in adjacency:
        for v in adjacency[u]:
            two_hop[u].update(w for w in adjacency[v] if w != u and w not in adjacency[u])
    print("Two-hop neighbors calculated")
    return two_hop

class State:
    def __init__(self, coloring, conflicts, num_colors, color_counts):
        self.coloring = coloring
        self.conflicts = conflicts
        self.num_colors = num_colors
        self.color_counts = color_counts.copy()
        self.balance_penalty = self.calculate_balance_penalty()
        self.heuristic = self.calculate_heuristic()
    
    def calculate_heuristic(self):
        return self.conflicts * 1000 + self.num_colors * 100 + self.balance_penalty
    
    def calculate_balance_penalty(self):
        counts = list(self.color_counts.values())
        if not counts:
            return 0
        avg = sum(counts) / len(counts)
        return sum((c - avg) ** 2 for c in counts)
    
    def __lt__(self, other):
        return self.heuristic < other.heuristic

def generate_initial_state(vertices, adjacency, two_hop_neighbors, pre_assigned):
    print("\nGenerating initial color state...")
    
    # Constraint 1: Degree-Based Coloring - Higher-degree vertices are assigned colors first
    vertices_sorted = sorted(vertices, key=lambda x: -len(adjacency[x]))
    
    color = {}
    color_count = defaultdict(int)
    
    for u in vertices_sorted:
        # Constraint 2: Pre-Assigned Colors - Some vertices have fixed colors
        if u in pre_assigned:
            c = pre_assigned[u]
            color[u] = c
            color_count[c] += 1
            continue
        
        # Constraint 4: Distance Constraint - Ensure two-hop neighbors don't have the same color
        forbidden = {color[v] for v in adjacency[u] if v in color}
        forbidden.update(color[v] for v in two_hop_neighbors[u] if v in color)
                
        c = 1
        while c in forbidden:
            c += 1
        color[u] = c
        color_count[c] += 1

    return color, color_count

def count_conflicts(coloring, adjacency, two_hop_neighbors):
    conflicts = 0
    for u in coloring:
        current_color = coloring[u]
        
        # Constraint 4: Distance Constraint - Avoid conflicts at two-hop distance
        conflicts += sum(1 for v in adjacency[u] if coloring.get(v) == current_color)
        conflicts += sum(1 for v in two_hop_neighbors[u] if coloring.get(v) == current_color)
    
    return conflicts // 2

def local_beam_search(adjacency, two_hop_neighbors, pre_assigned, beam_width=10, max_iter=100):
    print("\nStarting Local Beam Search...")
    vertices = list(adjacency.keys())
    
    # Constraint 1: Degree-Based Coloring - Sort by vertex degree
    vertices_sorted = sorted(vertices, key=lambda x: -len(adjacency[x]))
    
    initial_coloring, color_counts = generate_initial_state(vertices, adjacency, two_hop_neighbors, pre_assigned)
    conflicts = count_conflicts(initial_coloring, adjacency, two_hop_neighbors)
    
    initial_state = State(initial_coloring, conflicts, len(color_counts), color_counts)
    beam = [initial_state]
    
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")
        print(f"Best state: {beam[0].conflicts} conflicts, {beam[0].num_colors} colors")
        
        if beam[0].conflicts == 0:
            print("\nFound valid coloring with 0 conflicts.")
            return beam[0].coloring, beam[0].num_colors
        
        beam = heapq.nsmallest(beam_width, beam, key=lambda x: x.heuristic)
    
    best_state = min(beam, key=lambda x: (x.conflicts, x.num_colors))
    print(f"\nBest solution: {best_state.conflicts} conflicts, {best_state.num_colors} colors")
    return best_state.coloring, best_state.num_colors

def save_coloring_to_csv(coloring, adjacency, filename="graph_coloring_output.csv"):
    print("\nSaving results to CSV file...")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Vertex", "Color", "Adjacent Vertices (Color)"])
        
        for vertex in sorted(coloring.keys()):
            adjacent_info = ", ".join(f"{adj}({coloring[adj]})" for adj in adjacency[vertex])
            writer.writerow([vertex, coloring[vertex], adjacent_info])
    print(f"Results saved in {filename}")

def main():
    adjacency = parse_hypercube_dataset("hypercube_dataset.txt")
    two_hop_neighbors = precompute_two_hop_neighbors(adjacency)
    
    print("\nStarting graph coloring algorithm")
    
    coloring, num_colors = local_beam_search(adjacency, two_hop_neighbors, {}, 
                                             beam_width=10, max_iter=50)
    
    print("\nFinal Results:")
    print(f"Number of colors used: {num_colors}")
    
    save_coloring_to_csv(coloring, adjacency)

if __name__ == "__main__":
    main()
