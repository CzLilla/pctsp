import networkx as nx
import numpy as np
from config import *
from matplotlib import pyplot as plt
from src.plots import plot_graph


class DeltaTour(nx.Graph):
    def __init__(self, home_graph, node_list, endpoint):
        """
            Initializes a Tour graph as a path of nodes from node_list
            plus one edge connecting the last node with an arbitrary node (endpoint) of the path.
                    Args:
                        home_graph (nx.Graph): A complete graph containing the tour
                        node_list (list of ints): A list of nodes specifying the order of the tour
                        endpoint (int): The node connected to the last node of the path
        """
        super().__init__()
        self.home_graph = home_graph
        self.path = node_list
        self.startpoint = node_list[0]
        self.endpoint = endpoint
        self.path_endpoint = node_list[-1]
        if not all(node in self.home_graph for node in node_list):
            raise ValueError("All nodes in node_list must exist in the home graph.")
        for node in node_list:
            self.add_node(node, **self.home_graph.nodes[node])
        for i in range(len(node_list)-1):
            u, v = node_list[i], node_list[(i + 1)]
            if self.home_graph.has_edge(u, v):
                edge_attributes = self.home_graph.edges[u, v].copy()
                self.add_edge(u, v, **edge_attributes)
            else:
                raise ValueError(f"Edge ({u}, {v}) is not found in home graph.")
        if endpoint in node_list:
            edge_attributes = self.home_graph.edges[self.path_endpoint, self.endpoint].copy()
            self.add_edge(self.path_endpoint, self.endpoint, **edge_attributes)
        else:
            raise ValueError(f"Node {endpoint} is not in the tour.")

    def copy(self):
        return DeltaTour(self.home_graph, self.path.copy(), self.endpoint)

    def tour_length(self):
        length = sum(attr['length'] for _, _, attr in self.edges(data=True))
        return length

    def cycle_length(self):
        length = self.tour_length()
        length -= self.edges[self.path_endpoint, self.endpoint]['length']
        length += self.home_graph.edges[self.path_endpoint, self.startpoint]['length']
        return length

    def tour_price(self):
        price = sum(attr['price'] for node, attr in self.nodes(data=True))
        return price

    def tour_cost(self):
        total_price = sum(attr['price'] for node, attr in self.home_graph.nodes(data=True))
        cost = self.tour_length() + total_price - self.tour_price()
        return cost

    def cycle_cost(self):
        total_price = sum(attr['price'] for node, attr in self.home_graph.nodes(data=True))
        cost = self.cycle_length() + total_price - self.tour_price()
        return cost

    def _reset_path(self):
        copygraph = nx.Graph()
        for node in self.nodes():
            copygraph.add_node(node)
        for u, v in self.edges():
            copygraph.add_edge(u, v)
        copygraph.remove_edge(self.path_endpoint, self.endpoint)
        newpath = nx.shortest_path(copygraph, self.startpoint, self.path_endpoint)
        self.path = newpath


    def _reset_endpoints(self):
        self.startpoint = self.path[0]
        self.path_endpoint = self.path[-1]

    def insert_node(self, node, edge=None):
        if node not in set(self.home_graph.nodes()) - set(self.nodes()):
            raise ValueError("Node must be in home graph but not in tour.")
        self.add_node(node, **self.home_graph.nodes[node])
        if edge == None:
            self.add_edge(node, self.startpoint, **self.home_graph.edges[node, self.startpoint])
            self.startpoint = node
            self.path.insert(0, node)
        elif type(edge) is tuple and len(edge) == 2:
            u, v = edge
            self.remove_edge(u, v)  # raises NetworkXError if there is no such edge
            self.add_edge(u, node, **self.home_graph.edges[u, node])
            self.add_edge(v, node, **self.home_graph.edges[v, node])
            if (u == self.endpoint and v == self.path_endpoint) \
                    or (v == self.endpoint and u == self.path_endpoint):
                self.path.append(node)
            else:
                index_u = self.path.index(u)
                index_v = self.path.index(v)
                self.path.insert(max(index_u, index_v), node)
        else:
            raise ValueError("Invalid input for edge.")
        self.path_endpoint = self.path[-1]

    def insert_node_to_best_place(self, node):
        if node not in (set(self.home_graph.nodes()) - set(self.nodes())):
            raise ValueError("Node must be in home graph but not in tour.")
        best_edge = None
        best_cost = self.home_graph.edges[node, self.startpoint]['length']
        for u, v in self.edges():
            cost = self.home_graph.edges[u, node]['length'] \
                   + self.home_graph.edges[v, node]['length'] - self.edges[u, v]['length']
            if cost < best_cost:
                best_cost = cost
                best_edge = (u, v)
        self.insert_node(node, best_edge)

    def insert_biggest_node(self):
        sorted_nodes = sorted(set(self.home_graph.nodes()) - set(self.nodes()), reverse=True)
        self.insert_node_to_best_place(sorted_nodes[0])

    def insert_best_node(self, forced=False):
        best_node = None
        best_cost = float('inf')
        home_nodes = set(self.home_graph.nodes())
        tour_nodes = set(self.nodes())
        available_nodes = home_nodes - tour_nodes
        for node in available_nodes:
            current_tour = self.copy()
            current_tour.insert_node_to_best_place(node)
            cost = current_tour.tour_cost()
            if cost < best_cost:
                best_cost = cost
                best_node = node
        if forced or (best_cost < self.tour_cost()):
            self.insert_node_to_best_place(best_node)

    def drop_node(self, node):
        if node not in set(self.nodes()):
            raise ValueError("Node is not found in tour.")
        elif node == self.endpoint:
            raise ValueError("Node cannot be dropped.")
        neighbors = list(self.neighbors(node))
        self.remove_node(node)  # could be overridden to modify path
        self.path.remove(node)
        if len(neighbors) == 2:
            u, v = neighbors
            self.add_edge(u, v, **self.home_graph.edges[u, v])
        self.startpoint = self.path[0]
        self.path_endpoint = self.path[-1]

    def drop_worst_node(self, forced=False):
        node_to_drop = None
        min_increase = float('inf')
        if len(self.path) <= 3:
            return
        for node in set(self.nodes()):
            if node == self.endpoint or node == ROOT:
                continue
            elif self.endpoint == self.path[-3] \
                    and (node == self.path[-2] or node == self.path[-1]):
                continue
            neighbors = list(self.neighbors(node))
            if len(neighbors) == 1:
                increase = self.nodes[node]['price'] - self.edges[neighbors[0], node]['length']
            else:
                u, v, = neighbors
                increase = self.home_graph.edges[u, v]['length'] - self.edges[u, node]['length'] \
                           - self.edges[v, node]['length'] + self.nodes[node]['price']
            if increase < min_increase:
                min_increase = increase
                node_to_drop = node
        if forced or (min_increase < 0):
            self.drop_node(node_to_drop)


    def delta_change(self, new_endpoint=None):
        new_pathend_index = self.path.index(self.endpoint)+1
        to_reverse = self.path[new_pathend_index:]
        to_reverse.reverse()
        new_path = self.path[:new_pathend_index] + to_reverse
        new_path_endpoint = self.path[new_pathend_index]
        self.remove_edge(self.endpoint, new_path_endpoint)
        self.path = new_path
        self.path_endpoint = new_path_endpoint
        if new_endpoint == None:
            min_dist = float('inf')
            for i in range(len(self.path)-2):
                p = self.path[i]
                d = self.home_graph.edges[self.path_endpoint, p]['length']
                if d < min_dist:
                    new_endpoint = p
                    min_dist = d
        self.endpoint = new_endpoint
        u, v = self.path_endpoint, self.endpoint
        self.add_edge(u, v, **self.home_graph.edges[u, v])

    def swap_two_edges(self, edge1, edge2, forced=False):
        u1, v1 = edge1
        u2, v2 = edge2
        if self.path.index(u1) > self.path.index(v1):
            u1, v1 = v1, u1
        if self.path.index(u2) > self.path.index(v2):
            u2, v2 = v2, u2
        if v1 == self.path_endpoint and u1 == self.endpoint:
            u1, v1 = v1, u1
        if v2 == self.path_endpoint and u2 == self.endpoint:
            u2, v2 = v2, u2
        if (v2, v1) in self.edges():  # special case - we do not do a swap
            return False
        if forced or (self.home_graph.edges[u1, u2]['length'] + self.home_graph.edges[v1, v2]['length']
                      - self.edges[u1, v1]['length'] - self.edges[u2, v2]['length'] < 0):
            self.remove_edge(u1, v1)
            self.remove_edge(u2, v2)
            self.add_edge(u1, u2, **self.home_graph.edges[u1, u2])
            self.add_edge(v1, v2, **self.home_graph.edges[v1, v2])
            if u1 == self.path_endpoint:
                self.path_endpoint = v2
            elif u2 == self.path_endpoint:
                self.path_endpoint = v1
            self._reset_path()
            return True
        else:
            return False

    def two_exchange(self):
        max_gain = 0
        best_edge1 = None
        best_edge2 = None
        edges = list(self.edges()).copy()
        edges_to_check = edges.copy()
        for i, edge1 in enumerate(edges):
            if edge1 not in edges_to_check:
                continue
            for edge2 in edges[i+1:]:  # Start from i+1 to avoid comparing an edge with itself
                if edge2 not in edges_to_check:
                    continue
                # Check if edges share a vertex
                if edge1[0] in edge2 or edge1[1] in edge2:
                    continue
                swapped = self.swap_two_edges(edge1, edge2)
                if swapped:
                    edges_to_check.remove(edge1)
                    edges_to_check.remove(edge2)
                    break


class Tour(DeltaTour):
    def __init__(self, home_graph, node_list):
        super().__init__(home_graph, node_list, node_list[0])

    def copy(self):
        return Tour(self.home_graph, self.path.copy())

    def _shortcut(self, chord):
        u, v = chord
        if self.path.index(u) < self.path.index(v):
            for i in range(self.path.index(u)+1, self.path.index(v)):
                self.remove_node(self.path[i])
        else:
            for i in range(self.path.index(u)+1, len(self.path)):
                self.remove_node(self.path[i])
            for i in range(self.path.index(v)):
                self.remove_node(self.path[i])
        self.add_edge(u, v, **self.home_graph.edges[u, v])
        if self.startpoint not in self.nodes():  # in case startpoint gets cut out
            self.startpoint = ROOT
            self.endpoint = ROOT
        if self.path_endpoint not in self.neighbors(self.startpoint):
            self.path_endpoint = next(self.neighbors(self.startpoint))
        self._reset_path()

    def _shortcut_value(self, chord):
        u, v = chord
        u_index = self.path.index(u)
        v_index = self.path.index(v)
        value = self.home_graph.edges[u, v]['length']
        if u_index < v_index:
            value -= self.edges[u, self.path[u_index+1]]['length']
            for i in range(u_index+1, v_index):
                value += self.nodes[self.path[i]]['price']
                value -= self.edges[self.path[i], self.path[i+1]]['length']
        else:
            for i in range(u_index+1, len(self.path)):
                value += self.nodes[self.path[i]]['price']
                value -= self.edges[self.path[i-1], self.path[i]]['length']
            value -= self.edges[self.path[-1], self.path[0]]['length']
            for i in range(v_index):
                value += self.nodes[self.path[i]]['price']
                value -= self.edges[self.path[i], self.path[i+1]]['length']
        return value

    def shortcut(self):
        min_value = 0
        for u in self.nodes():
            for v in self.nodes():
                if u == v or self.has_edge(u, v):
                    continue
                if self.path.index(u) < self.path.index(ROOT) and self.path.index(ROOT) < self.path.index(v):
                    continue  # ROOT would be cut out
                if self.path.index(v) < self.path.index(u) and (self.path.index(u) < self.path.index(ROOT) or self.path.index(ROOT) < self.path.index(v)):
                    continue  # ROOT would be cut out
                chord = (u, v)
                value = self._shortcut_value(chord)
                if value < min_value:
                    min_value = value
                    best_chord = chord
        if min_value < 0:
            self._shortcut(best_chord)

