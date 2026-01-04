import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from config import *

def plot_graph(graph, title=None, node_size=50, with_labels=False, grid=True, **kwargs):
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw_networkx(graph, pos=pos, node_size=node_size, with_labels=with_labels, **kwargs)
    if grid:
        plt.grid(True)
    plt.axis('on')
    ax = plt.gca()  # Get current axes
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_LP_graph(graph, with_labels=False, with_edge_labels=False, node_size=None, grid=True, **kwargs):
    if node_size is None:
        weights = nx.get_node_attributes(graph, 'weight')
        node_size = [weights.get(node, 1)  * 50 for node in graph.nodes()]
    weights = nx.get_edge_attributes(graph, 'weight')
    edge_width = [weights.get(edge, 1)  * 2 for edge in graph.edges()]
    prices = nx.get_node_attributes(graph, 'price')
    min_price = min(prices.values())
    max_price = max(prices.values())
    node_color = [(prices.get(node, 0) - min_price) / (max_price - min_price) for node in graph.nodes()]
    pos = nx.get_node_attributes(graph, 'pos')
    if with_edge_labels:
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    plot_graph(graph, node_size=node_size, with_labels=with_labels, width=edge_width, grid=grid, node_color=node_color, cmap='Reds', **kwargs)

def plot_comparison(home_graph, tour1, tour2, title=None, **kwargs):
    pos_home = nx.get_node_attributes(home_graph, 'pos')
    prices = nx.get_node_attributes(home_graph, 'price')
    node_size = [prices.get(node, 1) * 0.5 for node in home_graph.nodes()]
    pos1 = nx.get_node_attributes(tour1, 'pos')
    pos2 = nx.get_node_attributes(tour2, 'pos')
    weights = nx.get_edge_attributes(tour1, 'weight')
    edge_width = [weights.get(edge, 1) * 2 for edge in tour1.edges()]
    nx.draw_networkx_nodes(home_graph, pos=pos_home, node_color='black', node_size=node_size, **kwargs)
    nx.draw_networkx_edges(tour1, pos=pos1, edge_color='lightsalmon', width=edge_width)
    nx.draw_networkx_edges(tour2, pos=pos2, edge_color='darkred', width=2, style='dashed')
    if title is not None:
        plt.title(title)
    plt.show()