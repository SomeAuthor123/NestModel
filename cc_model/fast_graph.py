import numpy as np
from cc_model.utils import networkx_from_edges, graph_tool_from_edges, calc_color_histogram, check_color_histograms_agree
from cc_model.fast_pagerank import pagerank_numba_sym, pagerank_numba_dir
from cc_model.fast_eigenvector import eigenvector_numba_sym, eigenvector_numba_dir
from cc_model.fast_wl import WL_fast

from cc_model.fast_rewire import rewire_fast, sort_edges, get_block_indices


def switch_in_out(edges):
    """small helper function that switches in edges to out edges and vice versa"""
    edges_tmp = np.empty_like(edges)
    edges_tmp[:,0]=edges[:,1]
    edges_tmp[:,1]=edges[:,0]
    return edges_tmp

class FastGraph:
    """A custom class representing Graphs through edge lists that can be used to efficiently be rewired"""
    def __init__(self, edges, is_directed, check_results=False):
        assert edges.dtype==np.uint32 or edges.dtype==np.uint64
        self._edges = edges.copy()
        self.edges_ordered = None
        self.is_directed = is_directed
        self.base_partitions = None
        self.latest_iteration_rewiring = 1000000
        self.num_nodes = edges.ravel().max()+1
        self.check_results = check_results
        self.wl_iterations = None

        # these will be set in reset_edges_ordered
        self.edges_classes = None
        self.dead_arr = None
        self.is_mono = None
        self.block_indices = None

        self.out_degree = np.array(np.bincount(edges[:,0].ravel(), minlength=self.num_nodes), dtype=np.uint32)
        self.in_degree = np.array(np.bincount(edges[:,1].ravel(), minlength=self.num_nodes), dtype=np.uint32)

        if self.is_directed:
            self.out_dead_ends = np.nonzero(self.out_degree==0)[0]
            self.corr_out_degree=self.out_degree.copy()
            self.corr_out_degree[self.out_dead_ends]+=1

            self.in_dead_ends = np.nonzero(self.in_degree==0)[0]
            self.corr_in_degree=self.in_degree.copy()
            self.corr_in_degree[self.in_dead_ends]+=1

            #print(len(self.out_dead_ends), len(self.in_dead_ends))
        else:
            self.out_degree=self.out_degree+self.in_degree
            self.in_degree=self.out_degree

        #raise ValueError
    @property
    def edges(self,):
        if self.edges_ordered is None:
            #print("_", id(self._edges))
            return self._edges
        else:
            #print("ordered", id(self.edges_ordered))
            return self.edges_ordered


    def to_gt(self, switch=False):
        edges = self.edges
        if switch:
            edges = switch_in_out(edges)
        return graph_tool_from_edges(edges, self.num_nodes, self.is_directed)

    def to_nx(self, switch=False):
        edges = self.edges
        if switch:
            edges = switch_in_out(edges)
        return networkx_from_edges(edges, self.num_nodes, self.is_directed)


    def calc_pagerank(self, mode, epsilon, max_iter, alpha, return_err=False):

        if not self.is_directed:
            #print((self.edges).dtype, (self.in_degree+self.out_degree).dtype)
            vector, err =  pagerank_numba_sym(self.edges,
                                              self.in_degree+self.out_degree,
                                              epsilon,
                                              max_iter,
                                              alpha)
        else:
            if "base" in mode:
                edges = self._edges
            else:
                edges = self.edges
            if "in" in mode:
                corr_degree = self.corr_out_degree
                dead_ends = self.out_dead_ends

            elif "out" in mode:
                corr_degree = self.corr_in_degree
                dead_ends = self.in_dead_ends
                edges = switch_in_out(edges)

            vector, err =  pagerank_numba_dir(edges,
                                               corr_degree,
                                               dead_ends,
                                               epsilon,
                                               max_iter,
                                               alpha)
        if return_err:
            return vector, err
        else:
            return vector


    def calc_ev(self, mode, epsilon, max_iter, return_err=False):
        if not self.is_directed:
            #print((self.edges).dtype, (self.in_degree+self.out_degree).dtype)
            vector, err =  eigenvector_numba_sym(self.edges, len(self.in_degree),
                                            epsilon,
                                            max_iter)
        else:

            if "base" in mode:
                edges = self._edges
            else:
                edges = self.edges
            if "in" in mode:
#                dead_ends = self.out_dead_ends
                pass
            elif "out" in mode:
#                dead_ends = self.in_dead_ends
                edges = switch_in_out(edges)

            vector, err =  eigenvector_numba_dir(edges,
                                            len(self.in_degree),
                                            epsilon,
                                            max_iter)
        if return_err:
            return vector, err
        else:
            return vector


    def calc_wl(self, edges = None, initial_colors=None):
        if edges is None:
            edges = self.edges
        if not self.is_directed:
            edges2 = np.vstack((edges[:,1], edges[:,0])).T
            edges = np.vstack((edges, edges2))

        if type(initial_colors).__module__ == np.__name__:
            return WL_fast(edges, labels = initial_colors)
        elif initial_colors is not None and "out_degree" in initial_colors:
            return WL_fast(edges, labels = self.out_degree)
        else:
            return WL_fast(edges)


    def calc_wl_arr(self, initial_colors=None):
        return np.array(self.calc_wl(initial_colors=initial_colors), dtype=np.uint32)


    def ensure_base_wl(self, initial_colors=None):
        if self.base_partitions is None:
            self.calc_base_wl(initial_colors=initial_colors)


    def calc_base_wl(self, initial_colors=None, both=False):
        if self.latest_iteration_rewiring != 1000000:
            raise ValueError("Seems some rewiring only employed cannot calc base WL")
        if both is False:
            partitions = self.calc_wl(self._edges, initial_colors=initial_colors)
        else:
            partitions = self.calc_wl_both(self._edges, initial_colors=initial_colors)

        self.base_partitions = np.array(partitions, dtype=np.uint32)
        self.wl_iterations = len(self.base_partitions)


    def ensure_edges_prepared(self, initial_colors=None, both=False):
        if self.base_partitions is None:
            self.calc_base_wl(initial_colors=initial_colors, both=both)
        if self.edges_ordered is None:
            self.reset_edges_ordered()


    def reset_edges_ordered(self):
        print("resetting")
        self.edges_ordered, self.edges_classes, self.dead_arr, self.is_mono = sort_edges(self._edges, self.base_partitions)
        self.block_indices = get_block_indices(self.edges_classes, self.dead_arr)


    def rewire(self, depth):
        assert depth < len(self.base_partitions), f"{depth} {len(self.base_partitions)}"
        assert depth <= self.latest_iteration_rewiring
        self.latest_iteration_rewiring = depth

        self.ensure_edges_prepared()
        if self.check_results:
            if self.is_directed:
                ins, outs = calc_color_histogram(self._edges, self.base_partitions[depth], self.is_directed)

        rewire_fast(self.edges_ordered,
                     self.edges_classes[:,depth],
                     self.is_mono[depth],
                    self.block_indices[depth],
                    self.is_directed)

        if self.check_results:
            if self.is_directed:
                ins2, outs2 = calc_color_histogram(self.edges_ordered, self.base_partitions[depth], self.is_directed)
                check_color_histograms_agree(ins, ins2)
                check_color_histograms_agree(outs, outs2)

                assert np.all(self.in_degree == np.bincount(self.edges[:,1].ravel(), minlength=self.num_nodes))
                assert np.all(self.out_degree == np.bincount(self.edges[:,0].ravel(), minlength=self.num_nodes))

                #check_colors_are_correct(self, depth)

            else:
                #print("checking degree")
                degree = self.in_degree + self.out_degree
                curr_degree1 = np.bincount(self.edges[:,0].ravel(), minlength=self.num_nodes)
                curr_degree2 = np.bincount(self.edges[:,1].ravel(), minlength=self.num_nodes)
                assert np.all(degree == (curr_degree1+curr_degree2))
