import networkx
import math
import time
from config import *
from src.plots import plot_graph
from src.tour import Tour
from src.tour import DeltaTour

def nearest_neighbor(graph, steps, start=ROOT):
    """
    Find a path starting from root node using nearest neighbor algorithm
    
    Args:
        graph: NetworkX graph (complete)
        start: Starting node
        steps: Number of steps to execute
    
    Returns:
        Tour object with path found by algorithm
    """
    path = [start]
    current = start
    
    # Execute specified number of steps
    for _ in range(steps):
        # Get all neighbors and their distances from current node
        neighbors = [(n, graph[current][n]['length'])
                    for n in graph.neighbors(current)
                    if n not in path]
        
        # If no unvisited neighbors left, break
        if not neighbors:
            break
        # Find nearest unvisited neighbor
        next_node = min(neighbors, key=lambda x: x[1])[0]
        path.append(next_node)
        current = next_node
    return Tour(graph, path)

def grow_tour(tour=None, home_graph=None, method='lin_kernighan', start_biggest = False):
    if tour is None:
        if home_graph is None:
            raise ValueError("Either tour or home_graph must be provided.")
        elif start_biggest:
            sorted_nodes = sorted(set(home_graph.nodes()) - {ROOT}, reverse=True)
            tour = Tour(home_graph=home_graph, node_list=[ROOT, sorted_nodes[0], sorted_nodes[1]])
        else:
            tour = nearest_neighbor(home_graph, steps=2)
    n = len(tour.home_graph.nodes())
    current_tour = tour.copy()
    min_cost = math.inf
    min_tour = None
    while len(current_tour.nodes()) < n:
        current_tour.insert_best_node(forced=True)  # grow the tour
        current_tour.two_exchange()
        decreased_tour = current_tour.copy()
        if method == 'two_exchange':
            decreased_tour.two_exchange()
        elif method == 'lin_kernighan':
            decreased_tour = lin_kernighan(decreased_tour)
        decreased_tour.shortcut()
        current_cost = decreased_tour.tour_cost()
        if current_cost < min_cost:
            min_cost = current_cost
            min_tour = decreased_tour.copy()
    return min_tour

def tour_to_delta(tour, v, u):
    '''
    Given a tour and two nodes v and u, returns a delta tour with v as the new startpoint,
    if its cost is less or equal to the current cost of the tour, otherwise returns None.
    '''
    delta_tour = DeltaTour(tour.home_graph, tour.path.copy(), tour.endpoint)
    if v not in delta_tour.nodes() or u not in delta_tour.nodes() or v == u or (v, u) not in delta_tour.edges():
        raise ValueError("Invalid node or edge")
    current_edgecost = delta_tour.edges[v, u]['length']
    delta_tour.startpoint = v
    delta_tour.endpoint = v
    delta_tour.path_endpoint = u
    delta_tour._reset_path()
    for w in set(delta_tour.nodes()) - {u} - set(delta_tour.neighbors(u)):
        if delta_tour.home_graph.edges[w, u]['length'] <= current_edgecost:
            delta_tour.remove_edge(v, u)
            delta_tour.add_edge(w, u, **delta_tour.home_graph.edges[w, u])
            delta_tour.endpoint = w

            delta_tour._reset_path()

            return delta_tour

def delta_to_tour(delta_tour):
    tour = delta_tour.copy()
    tour.remove_edge(tour.path_endpoint, tour.endpoint)
    tour.add_edge(tour.path_endpoint, tour.startpoint, **tour.home_graph.edges[tour.path_endpoint, tour.startpoint])
    tour.endpoint = tour.startpoint
    return Tour(tour.home_graph, tour.path.copy())

def lin_kernighan(tour):
    T = tour.copy()
    T_edges = [{u, v} for (u, v) in T.edges()]
    best_tour = tour.copy()
    for v in tour.nodes():
        for u in tour.neighbors(v):
            delta_tour = tour_to_delta(tour, v, u)
            if delta_tour is None:
                continue

            added_edges = list()

            while True:
                new_pathend_index = delta_tour.path.index(delta_tour.endpoint) + 1
                new_path_endpoint = delta_tour.path[new_pathend_index]

                if delta_tour.cycle_cost() < best_tour.cycle_cost():
                    best_tour = delta_tour.copy()
                success = True
                min_dist = float('inf')
                for i in range(1, len(delta_tour.path)):
                    p = delta_tour.path[i]

                    if (p == new_path_endpoint) or (p in delta_tour.neighbors(new_path_endpoint)):
                        continue
                    current_tour = delta_tour.copy()
                    old_end = current_tour.endpoint

                    current_tour.delta_change(new_endpoint=p)

                    deleted = {old_end, current_tour.path_endpoint}
                    added = {current_tour.path_endpoint, p}
                    added_edges.append(added)
                    if deleted in added_edges or added in T_edges:
                        continue
                    d = current_tour.edges[current_tour.path_endpoint, p]['length']
                    if d < min_dist:
                        new_endpoint = p
                        min_dist = d
                if min_dist == float('inf'):  # no suitable p
                    break

                current_tour = delta_tour.copy()
                current_tour.delta_change(new_endpoint=new_endpoint)
                if current_tour.tour_cost() > T.tour_cost():
                    break
                delta_tour = current_tour
            if best_tour.cycle_cost() < T.tour_cost():
                T = delta_to_tour(best_tour)
    return T

