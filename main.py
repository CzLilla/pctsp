import ast
import time

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pulp
from matplotlib.pyplot import title
from networkx.classes import nodes_with_selfloops
from src.heuristics import tour_to_delta
from src.random_graph import RandomGraph
from src.tour import DeltaTour, Tour
from src.LP_relaxation import PriceCollectingLpProblem
import src.heuristics as heuristics
from src.plots import plot_graph, plot_LP_graph, plot_comparison
from config import *


def make_heuristic_tours(folder, filename, method='lin_kernighan', start_biggest=False, seed_range=range(50)):
    graph_data = pd.DataFrame(columns=['seed', 'path', 'cost', 'length', 'prize', 'number of nodes'])
    for i in seed_range:
        home_graph = RandomGraph(seed=i)
        tour = heuristics.grow_tour(home_graph=home_graph, method=method, start_biggest=start_biggest)
        newrow = pd.Series([i, tour.path, tour.tour_cost(), tour.tour_length(), tour.tour_price(), len(tour.path)])
        graph_data.loc[i-1] = newrow.values
        print('seed =', i, ', cost =', tour.tour_cost())
    graph_data.to_csv(folder + filename, index=False) # csv file
    return graph_data

def write_lp_data(folder, filename, seed_range):
    lp_costs = []
    numbers_of_nodes = []
    for i in seed_range:
        lp = PriceCollectingLpProblem(RandomGraph(seed=i))
        lp_graph = lp.solve_lp()
        lp_costs.append(lp.objective.value())
        numbers_of_nodes.append(lp_graph.number_of_nodes())
        node_weights = nx.get_node_attributes(lp_graph, 'weight')
        plot_LP_graph(lp_graph, title='Solution of the LP - Graph ' + str(i))
        with open(folder + filename, 'a') as f:  # txt file
            f.write(f'Seed: {i}\n')
            f.write(f'Cost: {lp.objective.value()}\n')
            f.write(f'Number of nodes: {lp_graph.number_of_nodes()}\n')
            f.write(f'Nodes:\n')
            f.write(f'{node_weights}\n')
            f.write(f'Edges:\n')
            f.write(f'{lp_graph.edges(data=True)}\n')
            f.write(f'------------------------------------\n')
    print('LP costs:', lp_costs)
    print('numbers of nodes:', numbers_of_nodes)


if __name__ == '__main__':
    # Example 1
    home_graph_ex_1 = RandomGraph(seed=32)
    lp_ex_1 = PriceCollectingLpProblem(home_graph_ex_1)
    lp_graph_ex = lp_ex_1.solve_lp()
    plot_LP_graph(lp_graph_ex, title='Solution of the LP - Graph ' + str(32))

    # Example 2
    home_graph_ex_2 = RandomGraph(seed=17)
    lp_ex_2 = PriceCollectingLpProblem(home_graph_ex_2)
    lp_tour_ex = lp_ex_2.solve_lp()
    tour_ex = heuristics.grow_tour(home_graph=home_graph_ex_2, method='lin_kernighan')
    plot_comparison(home_graph_ex_2, lp_tour_ex, tour_ex, title='LP and heuristic tour comparison - Graph ' + str(17))
