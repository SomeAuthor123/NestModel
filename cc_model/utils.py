import time
import numpy as np
import graph_tool.all as gt


def nx_to_gt(G, verbosity=0):
    if verbosity>2:
        print(repr(G), len(G.nodes), len(G.edges))
    if verbosity>3:
        print("creating edge list")

    edge_list = np.array(list(G.edges), dtype=int)
    while edge_list.min()>0:
        edge_list-=1
    if verbosity>3:
        print("done creating edge list")
        print("creating graph")
        time.sleep(0.0001)
    g = gt.Graph(directed = False)
    g.add_vertex(len(G.nodes))
    g.add_edge_list(edge_list)
    if verbosity>3:
        print("done creating graph")
        time.sleep(0.0001)
    return g


def graph_tool_from_edges(edges, size, is_directed):

    if size is None:
        unique = np.unique(edges.flatten())
        assert unique[0]==0, "expecting to start from 0 " + str(unique[:10])
        size = len(unique)
    graph =  gt.Graph(directed=is_directed)
    graph.add_vertex(size)
    graph.add_edge_list(edges)
    return graph


def networkx_from_edges(edges, size, is_directed):

    if size is None:
        unique = np.unique(edges.flatten())
        assert unique[0]==0, "expecting to start from 0 " + str(unique[:10])
        size = len(unique)
    import networkx as nx
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(list(range(size)))
    G.add_edges_from(edges)
    return G


from collections import Counter, defaultdict
def calc_color_histogram(edges, labels, is_directed):
    assert is_directed

    outs = defaultdict(Counter)
    ins = defaultdict(Counter)
    for e1,e2 in edges:
        l1 = labels[e1]
        l2 = labels[e2]
        outs[e1][l2]+=1
        ins[e2][l1]+=1
    return outs, ins


def check_color_histograms_agree(hist1, hist2):
    assert len(hist1)==len(hist2)
    for key in hist1:
        assert key in hist2, f"{key} not in hist2"

    for key in hist1:
        val1 = hist1[key]
        val2 = hist2[key]

        assert val1.most_common()==val1.most_common(), f"{key} {val1} {val2}"


def compare_partitions(p1s, p2s):
    #print(p1s.shape)
    for depth, (p1, p2) in enumerate(zip(p1s, p2s)):
        same = p1==p2
        if not np.all(same):
            print("current depth", depth)
            for i, (a,b) in enumerate(zip(p1, p2)):
                if a!=b:
                    print(i, a,b)
            print(np.vstack((p1[~same], p2[~same])))
            print()

def compare_edges(edges1, edges2):
    o1 = np.lexsort(edges1.T)
    o2 = np.lexsort(edges2.T)
    edges1 = edges1[o1,:]
    edges2 = edges2[o2,:]
    diffs = np.all(edges1==edges2,axis=1)
    if not np.all(diffs):
        print(np.hstack((edges1[~diffs,:], edges2[~diffs,:])))
        print()


from cc_model.wl import WL, labelings_are_equivalent
def check_colors_are_correct(G, max_depth):
    _, labelings, = WL(G.to_gt(False))
    assert len(labelings)==len(G.base_partitions)-1
    for i, (p1,p2) in enumerate(zip(labelings, G.base_partitions)):
        if i > max_depth:
            print(f"skipped {i}")
            continue
        if labelings_are_equivalent(p1,p2):
            continue
        print(labelings_are_equivalent(p1,p2))
        print(p1)
        print(p2)
        agree = p1==p2
        #print(np.unique(p1.ravel()))
        #print(np.unique(p2.ravel()))
        print(len(np.unique(p1.ravel())),len(np.unique(p2.ravel())))
        print("uniques", np.all((np.unique(p1.ravel())==np.unique(p2.ravel()))))
        print(agree.sum())


        print(p1[~agree])
        print(p2[~agree])
        assert np.all(p1==p2)
    print("WL colors agree")