from pulp import *
import numpy as np
import math
import time
import networkx as nx
from config import *

class PriceCollectingLpProblem(LpProblem):
    def __init__(self, graph, name='Price_Collecting_TSP', sense=LpMinimize):
        super().__init__(name, sense)
        self.graph = graph
        self.n = graph.number_of_nodes()

        # Define decision variables
        self.y = LpVariable.dicts('y', range(self.n), lowBound=0, upBound=1, cat='Continuous')
        self.x = {}
        for i in range(self.n):
            for j in range(i+1, self.n):
                self.x[i, j] = LpVariable(f'x_{i}_{j}', lowBound=0, upBound=1, cat='Continuous')

        # Add constraints
        vertex_constraints = []
        vertex_constraints.append(lpSum(self.x[0, i] for i in range(1, self.n)) <= 2)
        for i in range(1, self.n):
            vertex_constraints.append(
                (lpSum(self.x[i, j] for j in range(i + 1, self.n)) +\
                 lpSum(self.x[j, i] for j in range(i))) ==\
                2 * self.y[i]
            )
        root_constraint = (self.y[0] == 1)
        for const in vertex_constraints:
            self += const
        self += root_constraint

        # Add the objective function
        obj_func = lpSum(self.graph.nodes[i]['price'] * (1 - self.y[i]) for i in range(self.n)) + \
                   lpSum(self.graph.edges[i, j]['length'] * self.x[i, j] for i in range(self.n) for j in range(i + 1, self.n))
        self += obj_func

    def _result_graph(self):
        '''
        After solving the model, gives a networkx graph instance
        with edge and node weights representing the values of the optimum.
        '''
        H = nx.Graph()
        # First, add all nodes that will be needed
        nodes_to_add = set()
        
        # Add nodes with significant y values
        for i in range(self.n):
            if self.y[i].value() > SENSITIVITY_CONSTANT:
                nodes_to_add.add(i)
        
        # Add nodes that are part of significant edges
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.x[i, j].value() > SENSITIVITY_CONSTANT:
                    nodes_to_add.add(i)
                    nodes_to_add.add(j)
        
        # Add all needed nodes with their attributes
        for i in nodes_to_add:
            H.add_node(i, **self.graph.nodes[i])
            nx.set_node_attributes(H, {i: self.y[i].value()}, 'weight')
        
        # Add edges
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.x[i, j].value() > SENSITIVITY_CONSTANT:
                    H.add_edge(i, j, weight=self.x[i, j].value())
        
        return H

    def _calculate_delta_S(self, S):
        delta_S = 0
        for u in S:
            delta_S += (lpSum(self.x[u, i] for i in range(u + 1, self.n) if i not in S))
            delta_S += (lpSum(self.x[i, u] for i in range(u) if i not in S))
        return delta_S

    def add_constraints_for_minimum_cuts(self, G):
        '''Extends the LP with set constraints from minimum cuts.'''
        set_constraints = []
        for v in G.nodes() - {0}:
            S = nx.minimum_cut(G, 0, v, 'weight')[1][1]
            delta_S = self._calculate_delta_S(S)
            if delta_S.value() < 2 * self.y[v].value():
                set_constraints.append(delta_S >= 2 * self.y[v])
        for const in set_constraints:
            self += const

    def add_constraints_for_connected_components(self, G):
        '''Extends the LP with set constaints for the connected components.'''
        set_constraints_connected = []
        for S in sorted(nx.connected_components(G)):
            delta_S = self._calculate_delta_S(S)
            for v in S:
                if delta_S.value() < 2 * self.y[v].value():
                    set_constraints_connected.append(delta_S >= 2 * self.y[v])
        for const in set_constraints_connected:
            self += const

    def solve_lp(self, solver=None, time_limit=float('inf'), iter=NUMBER_OF_ITERATIONS):
        start_time = time.time()
        if solver is None:
            solver = PULP_CBC_CMD(msg=1, gapRel=0.1)
        self.solve(solver)
        result_LP = self.objective.value()
        G = self._result_graph()
        const_num = len(self.constraints)
        for i in range(iter-1):
            if nx.is_connected(G):
                self.add_constraints_for_minimum_cuts(G)
            else:
                self.add_constraints_for_connected_components(G)
            if len(self.constraints) == const_num:      # no new constraints have been added
                print('Number of constraints: ', len(self.constraints))
                print('No new constraints added. Terminating the algorithm in iteration ', i+1)
                break
            if time.time() - start_time > time_limit:
                print('Time limit exceeded. Terminating the algorithm in iteration ', i+1)
                break
            self.solve(solver)
            result_LP = self.objective.value()
            G = self._result_graph()
            const_num = len(self.constraints)
        end_time = time.time()
        print("Running time of solve_LP:", end_time - start_time)
        return G