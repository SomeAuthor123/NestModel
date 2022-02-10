import numpy as np
import networkx as nx
from copy import deepcopy

class HashFunction:
    def __init__(self):
        self.reset()

    def apply(self, value):
        if value not in self.hash_dict:
            self.hash_dict[value] = self.hash_counter
            self.hash_counter += 1
        return self.hash_dict[value]

    def reset(self):
        self.hash_dict = {}
        self.hash_counter = 2

def is_equiv_subroutine(c1, c2):
    color_map = {}
    for i in range(len(c1)):
        if c1[i] not in color_map:
            color_map[c1[i]] = c2[i]
        else:
            if color_map[c1[i]] != c2[i]:
                return False
    return True

def is_equivalent(c1, c2):
    return is_equiv_subroutine(c1, c2) and is_equiv_subroutine(c2, c1)

wl_hash = HashFunction()

def weisfeiler_lehman(graph1: nx.Graph, iterations=-1, early_stopping=True, hash=wl_hash):
    if iterations == -1:
        iterations = len(graph1)

    Gamma1 = np.ones(len(graph1), dtype=int)
    set_colors_by_iteration = []
    colors_by_iteration = []

    for t in range(iterations):
        tmp_Gamma1 = np.copy(Gamma1)
        colors_by_iteration.append(deepcopy(Gamma1))
        set_colors_by_iteration.append(set(Gamma1))
        for node in graph1.nodes:
            Gamma1[node] = hash.apply((Gamma1[node], tuple(sorted([tmp_Gamma1[n] for n in graph1[node]]))))
        if is_equivalent(Gamma1, tmp_Gamma1) and early_stopping:
            return Gamma1, t, set_colors_by_iteration, colors_by_iteration

    colors_by_iteration.append(deepcopy(Gamma1))
    set_colors_by_iteration.append(set(Gamma1))
    return Gamma1, iterations, set_colors_by_iteration, colors_by_iteration