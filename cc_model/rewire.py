from collections import defaultdict
import numpy as np
from numba import njit
from cc_model.utils import graph_tool_from_edges
import graph_tool.all as gt
from collections import Counter

from cc_model.my_graph_tool import random_rewire


@njit
def swab_colors_and_edges(color_arr, edges):
    for i in range(color_arr.shape[0]):
        if color_arr[i,0] > color_arr[i,1]:
            tmp = color_arr[i,0]
            color_arr[i,0]=color_arr[i,1]
            color_arr[i,1]=tmp
            
            tmp2 = edges[i,0]
            edges[i,0]=edges[i,1]
            edges[i,1]=tmp2



def rename_edges(edges, mapping):
    new_edges = np.empty_like(edges)
    for i,(a, b) in enumerate(edges):
        new_edges[i,0] = mapping[a]
        new_edges[i,1] = mapping[b]
    return new_edges

def rename_edges_numpy(edges, mapping_numpy):
    new_edges = np.empty_like(edges)
    new_edges[:,0] =  mapping_numpy[edges[:,0]]
    new_edges[:,1] =  mapping_numpy[edges[:,1]]
    return new_edges




class LocalGraph:
    def __init__(self, edges, block_membership, is_directed):
        assert edges.shape[1]==2
        nodes = np.unique(edges.flatten())
        
        self.external_to_internal = {external : i for i, external in enumerate(nodes)}
        self.internal_to_external = {i : external for i, external in enumerate(nodes)}
        self.internal_to_external_numpy = nodes
        internal_edges = rename_edges(edges, self.external_to_internal)
        self.n_edges = len(internal_edges)
        self.is_dead = False
        
        self.graph = graph_tool_from_edges(internal_edges, len(nodes), is_directed)
        self.graph.set_fast_edge_removal(True)
        self.pin = self.graph.new_edge_property("bool")._get_any()

        self.is_bipartite=False

        if not block_membership is None:
            self.is_bipartite = True
            # pick only those colors that are also there in edges
            block_membership = block_membership[nodes]

            assert len(np.unique(block_membership))==2
            # if there is only one node on one side there is nothing to rewire
            if Counter(block_membership).most_common(2)[1][1]==1:
                self.is_dead = True
                self.dead_edges = edges

            self.block_membership = self.graph.new_vertex_property("int", vals=block_membership)._get_any()
        else:
            self.block_membership = self.graph.new_vertex_property("int")._get_any()


        
        
        
    def randomize(self, min_factor, max_factor):
        if self.is_dead:
            return
        low = round((1+min_factor)*self.n_edges)
        high = round((1+max_factor)*self.n_edges)
        n_iter = np.random.randint(low, high)
        
        if not self.is_bipartite:# one class case (Red, Red)
            model="configuration"
        else: # Two class case (Red, Blue)
            model = "constrained-configuration"
        random_rewire(self.graph,
                        model=model,
                        block_membership = self.block_membership,                              
                        n_iter=n_iter,
                        pin=self.pin,
                        edge_sweep=False)
        
        
    def get_edges(self):
        if self.is_dead:
            return self.dead_edges
        edges = self.graph.get_edges()
        assert len(edges)==self.n_edges
        #print(len(self.graph.get_vertices()))
        return rename_edges_numpy(edges, self.internal_to_external_numpy)

def split_edges_by_color(g, colors):
    edges = g.get_edges()
    color_arr = np.vstack((colors[edges[:,0]], colors[edges[:,1]])).T
    if not g.is_directed():
        # sort colors such that (Blue,Red) and (Red,Blue) edges are treated the same
        swab_colors_and_edges(color_arr, edges)

    e1=color_arr[:,0]
    e2=color_arr[:,1]

    mono_edges = defaultdict(list)
    dual_edges = defaultdict(list)

    for c1,c2,edge in zip(e1,e2,edges):
        if c1==c2:
            mono_edges[c1].append(tuple(edge))
        else:
            dual_edges[(c1,c2)].append(tuple(edge))
    return mono_edges, dual_edges


class LocalHistogramRewiring:
    def __init__(self, g, colors):
        mono_edges, dual_edges = split_edges_by_color(g, colors)
        #print(list(map(len, mono_edges.values())))
        #print(list(map(len, dual_edges.values())))
        
        self.n_nodes = len(g.get_vertices())
        self.is_directed = g.is_directed()
        self.min_factor=0
        self.max_factor=1
        self.n_edges = len(g.get_edges())
        
        local_graphs = []
        dead_edges = []
        for edges in mono_edges.values():
            if len(edges) > 1:
                local_graphs.append(LocalGraph(np.array(edges, dtype=int), None, is_directed=self.is_directed))
            else:
                dead_edges.extend(edges)
        for edges in dual_edges.values():
            if len(edges) > 1:
                local_graphs.append(LocalGraph(np.array(edges, dtype=int), colors, is_directed=self.is_directed))
            else:
                dead_edges.extend(edges)
        self.sub_graphs = local_graphs
        self.dead_edges = np.array(dead_edges, dtype=int)
        if len(dead_edges)== 0:
            self.dead_edges = np.empty((0,2),dtype=int)
        
        
    def get_sample(self):
        edges = np.empty((self.n_edges, 2), dtype=int)
        i = 0
        for sub_graph in self.sub_graphs:
            sub_graph.randomize(self.min_factor, self.max_factor)
            
            #copy edges to total_edges
            sub_edges = sub_graph.get_edges()
            edges[i:i+len(sub_edges),:]=sub_edges
            i+=sub_edges.shape[0]
        assert i==len(edges)-len(self.dead_edges), "could not fully fill the array"
        edges[i:,:]=self.dead_edges
        return graph_tool_from_edges(edges, self.n_nodes, self.is_directed)


import time
def block_rewiring(g, WL_round,n_iter, verbosity=0):
    if verbosity > 3:
        print("starting rewiring")
        time.sleep(0.01)
    gt.random_rewire(g,
                     model="constrained-configuration",
                     block_membership = g.vp[f"color_{WL_round}"], n_iter=n_iter)
    if verbosity > 3:
        print("done rewiring")
    return g