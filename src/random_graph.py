import networkx as nx
import numpy as np
from config import *

class RandomGraph(nx.Graph):
    def __init__(self, *args, seed=None, n=N, field_size=FIELD_SIZE, max_price=MAX_PRICE,  **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.field_size = field_size
        self.max_price = max_price
        rng = np.random.default_rng(self.seed)
        x_coords = rng.random(n) * self.field_size
        y_coords = rng.random(n) * self.field_size
        vertices = rng.integers(self.max_price, size=n)
        for i in range(n):
            self.add_node(i, price=vertices[i], pos=(x_coords[i], y_coords[i]))
        for i in range(n):
            for j in range(i+1, n):
                dist = (x_coords[i] - x_coords[j]) ** 2 + (y_coords[i] - y_coords[j]) ** 2
                edge = np.ceil(np.sqrt(dist))
                self.add_edge(i, j, length=edge)