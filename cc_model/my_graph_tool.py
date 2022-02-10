import graph_tool
import graph_tool.all as gt
import graph_tool
from graph_tool.dl_import import dl_import
dl_import("from graph_tool.generation import libgraph_tool_generation")
dl_import("from graph_tool import libgraph_tool_core as libcore")
def _prop(t, g, prop):
    """Return either a property map, or an internal property map with a given
    name."""
    if isinstance(prop, str):
        try:
            pmap = g.properties[(t, prop)]
        except KeyError:
            raise KeyError("no internal %s property named: %s" %\
                           ("vertex" if t == "v" else \
                            ("edge" if t == "e" else "graph"), prop))
    else:
        pmap = prop
    if pmap is None:
        return libcore.any()
    if t != prop.key_type():
        names = {'e': 'edge', 'v': 'vertex', 'g': 'graph'}
        raise ValueError("Expected '%s' property map, got '%s'" %
                         (names[t], names[prop.key_type()]))
    u = pmap.get_graph()
    if u is None:
        raise ValueError("Received orphaned property map")
    if g.base is not u.base:
        raise ValueError("Received property map for graph %s (base: %s), expected: %s (base: %s)" %
                         (str(g), str(g.base), str(u), str(u.base)))
    print("last")
    return pmap._get_any()




def random_rewire(g, model="configuration", n_iter=1, edge_sweep=True,
                  parallel_edges=False, self_loops=False, configuration=True,
                  edge_probs=None, block_membership=None, cache_probs=True,
                  persist=False, pin=None, ret_fail=False, verbose=False):



    traditional = True
    micro = False


    if pin is None:
        raise ValueError()

    
    pcount = libgraph_tool_generation.random_rewire(g._Graph__graph,
                                                    model,
                                                    n_iter,
                                                     not edge_sweep,
                                                    self_loops,
                                                     parallel_edges,
                                                    configuration, 
                                                    traditional,
                                                    micro, 
                                                    persist, 
                                                    None,
                                                    pin,
                                                    block_membership,
                                                    cache_probs,
                                                    graph_tool._get_rng(), verbose)
    return pcount