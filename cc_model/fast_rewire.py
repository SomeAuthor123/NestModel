from itertools import chain
from numba.np.ufunc import parallel
import numpy as np
from numba import njit, int64, uint32, uint64, prange, bool_
from numba.typed import List, Dict
from numpy.lib.arraysetops import unique


def get_dead_edges(labels, edges, dead_colors):
    is_dead_end1 = dead_colors[labels[edges[:,0]]]
    is_dead_end2 = dead_colors[labels[edges[:,1]]]
    return np.logical_or(is_dead_end1, is_dead_end2)



def get_dead_edges_full(edge_with_node_labels, edges, order, num_nodes):
    """ returns arrays which indicate whether an edge is a dead edge
    edges are dead when in the subgraph there is only one node involved on either side

    """


    num_labelings = edge_with_node_labels.shape[1]//2

    dead_indicators = np.zeros((edges.shape[0], num_labelings), dtype=np.bool)
    for i in range(num_labelings):
        _get_dead_edges(edge_with_node_labels[:,i*2:i*2+2], edges, order, num_nodes, dead_indicators[:,i])
    return dead_indicators

@njit
def _get_dead_edges(edge_with_node_labels, edges, order, num_nodes, out):
    #print(edge_with_node_labels.shape)
    start_edge = order[0]
    last_label_0 = edge_with_node_labels[start_edge, 0]
    last_label_1 = edge_with_node_labels[start_edge, 1]

    last_id_0 = edges[start_edge, 0]
    last_id_1 = edges[start_edge, 1]

    start_of_last_group = 0
    last_group_is_dead_0 = False
    last_group_is_dead_1 = False
    len_last_group = 0

    for i in range(order.shape[0]):
        curr_edge = order[i]
        curr_label_0 = edge_with_node_labels[curr_edge, 0]
        curr_label_1 = edge_with_node_labels[curr_edge, 1]

        curr_id_0 = edges[curr_edge, 0]
        curr_id_1 = edges[curr_edge, 1]

        if curr_label_0 != last_label_0 or curr_label_1 != last_label_1:
            if (last_group_is_dead_0 or last_group_is_dead_1) or len_last_group==1:
                for j in range(start_of_last_group, i):
                    out[order[j]] = True
            last_group_is_dead_0 = True
            last_group_is_dead_1 = True

            start_of_last_group = i
            len_last_group = 0
            last_label_0 = curr_label_0
            last_label_1 = curr_label_1

            last_id_0 = curr_id_0
            last_id_1 = curr_id_1
        if last_id_0 != curr_id_0:
            last_group_is_dead_0 = False
        if last_id_1 != curr_id_1:
            last_group_is_dead_1 = False
        len_last_group+=1
    if (last_group_is_dead_0 and last_group_is_dead_1) or len_last_group==1:
        for j in range(start_of_last_group, len(out)):
            out[order[j]] = True


    return out



#@njit
def get_edge_id1(edge_with_node_labels, order, out):
    #order = np.lexsort(edge_with_node_labels.T)
    return _get_edge_id(edge_with_node_labels, order, out)

@njit
def _get_edge_id(edge_with_node_labels, order, out):
    last_label_0 = edge_with_node_labels[order[0],0]
    last_label_1 = edge_with_node_labels[order[0],1]

    if last_label_0==last_label_1:
        is_mono = {0 : True}
    else:
        is_mono = {0 : False}
    num_edge_colors = 0
    for i in range(order.shape[0]):
        curr_edge = order[i]
        node_label_0 = edge_with_node_labels[curr_edge,0]
        node_label_1 = edge_with_node_labels[curr_edge,1]
        if node_label_0!=last_label_0 or node_label_1!=last_label_1:
            num_edge_colors += 1
            last_label_0=node_label_0
            last_label_1=node_label_1
            if node_label_0==node_label_1:
                is_mono[num_edge_colors] = True

        out[curr_edge] = num_edge_colors

    return out, is_mono

def get_edge_id(labels, edges):
    max_label = labels.max()
    edge_id =  max_label*(labels[edges[:,0]]) + labels[edges[:,1]]
    return edge_id

@njit
def get_edge_id2(labels, edges, out):


    d = {(0, 0) : 0}
    del d[(0, 0)]
    is_mono = {0 : True}
    for i in range(edges.shape[0]):
        e1 = edges[i,0]
        e2 = edges[i,1]
        tpl = (labels[e1], labels[e2])
        if tpl in d:
            out[i] = d[tpl]
        else:
            n = len(d)
            out[i] = n
            d[tpl] = n
            if labels[e1] == labels[e2]:
                is_mono[n] = True

    return out, is_mono


@njit
def get_edge_id3(labels, edges, out):


    d = {uint64(0) : 0}
    del d[uint64(0)]
    is_mono = {0 : True}
    for i in range(edges.shape[0]):
        e1 = edges[i,0]
        e2 = edges[i,1]
        tpl = uint64(labels[e1]) * uint64(labels[e2])
        if tpl in d:
            out[i] = d[tpl]
        else:
            n = len(d)
            out[i] = n
            d[tpl] = n
            if labels[e1] == labels[e2]:
                is_mono[n] = True

    return out, is_mono

#@njit([(int64[:], int64[:,:], int64[:]), (uint32[:], int64[:,:], uint32[:])])
def get_edge_id4(labels, edges, out):
    arr = np.array([ {-1 : -1} for _ in range(int64(labels.max())+1) ])

    is_mono = {0 : True}
    n=0
    for i in range(edges.shape[0]):
        l1 = labels[edges[i,0]]
        l2 = labels[edges[i,1]]
        d=arr[l1]
        if (l2 in d):
            out[i]=d[l2]
        else:
            d[l2] = n
            out[i]=n
            n+=1
            if l1== l2:
                is_mono[n] = True
    return out, is_mono
@njit(parallel=True)
def fill_arr(arr, labels, edges):
    for i in prange(edges.shape[0]):
        l1 = labels[edges[i,0]]
        l2 = labels[edges[i,1]]
        arr[i] = l1 * l2


def get_edge_id5(labels, edges, out):
    m = np.uint64(labels.max())
    assert m * m < np.iinfo(np.uint32).max

    arr = np.empty(edges.shape[0], dtype=np.uint32)


    fill_arr(arr, labels, edges)


    #uniques = np.unique(arr)
    #out = np.searchsorted(uniques, arr)

    unique_labels = np.square(np.unique(labels))

    #monos = np.searchsorted(uniques, unique_labels)
    is_mono = {label : True for label in unique_labels}


    return arr, is_mono
import time


@njit(parallel=True)
def assign_node_labels(labels, edges, out):
    for i in prange(edges.shape[0]):
        node_0 = edges[i,0]
        node_1 = edges[i,1]
        out[i,0]=labels[node_0]
        out[i,1]=labels[node_1]

def sort_edges(edges, labelings, directed = True):
    """Sort edges such that that edges of similar classes are consecutive

    additionally puts dead edges at the end

    """
    #print("sort_edges2")
    # WARNING If network is undirected edges need to be sorted first
    if directed is False:
        raise ValueError()


    edges_classes = []
    is_mono = []
    edge_with_node_labels = np.empty((edges.shape[0], 2*labelings.shape[0]), dtype=labelings.dtype)

    edge_with_node_labels
    for i in range(labelings.shape[0]):
        assign_node_labels(labelings[i,:], edges , edge_with_node_labels[:,i*2:i*2+2])
    #print(edge_with_node_labels.max())
    order = np.lexsort(edge_with_node_labels[:,::-1].T)
    #order = get_order(edge_with_node_labels)

    #print(edge_with_node_labels.shape)
    #print(len(order))
    #print(edges.shape)
    #print(edge_with_node_labels[order,:])
    for i in range(labelings.shape[0]):
        #assign_node_labels(labelings[i,:], edges , edge_with_node_labels)
        edge_class, mono = get_edge_id1(edge_with_node_labels[:,i*2:i*2+2], order, np.empty(len(edges), dtype=np.uint32))

        edges_classes.append(edge_class)
        is_mono.append(mono)


    dead_indicator = get_dead_edges_full(edge_with_node_labels, edges, order, labelings.shape[1]).T
    #print(np.hstack((edges, edge_with_node_labels, dead_indicator.T))[order,:])
    #raise ValueError
    tmp = list(chain.from_iterable(zip(edges_classes, dead_indicator)))
    #print(tmp)
    #(list(tmp))
    edges_classes_arr = np.vstack(edges_classes)
    to_sort_arr = np.vstack(tmp)#[dead_ids]+ edges_classes)

    # sort edges such that each of the classes are in order
    edge_order = np.lexsort(to_sort_arr[::-1,:])
    #print(edge_order)
    edges_ordered = edges[edge_order,:]
    #print(np.hstack((edges_ordered, edges_classes_arr[:, edge_order].T, dead_indicator[:, edge_order].T))[edge_order,:])

    return edges_ordered, edges_classes_arr[:, edge_order].T, dead_indicator[:, edge_order], is_mono




@njit
def rewire_mono_small(edges, n_rewire):
    """
    Rewires a single class network specified by edges in place!
    There are n_rewire steps of rewiring attempted.

    This function is optimized for small networks because it does linear search to resolve
        potential double edges
    """
    delta = len(edges)


    for _ in range(n_rewire):
        index1 = np.random.randint(0, delta)
        offset = np.random.randint(1, delta)
        i2_1 = np.random.randint(0, 2)
        i2_2 = 1 - i2_1
        index2 = (index1 + offset) % (delta)
        e1_l, e1_r = edges[index1,:]
        e2_l = edges[index2, i2_1]
        e2_r = edges[index2, i2_2]


        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue

        can_flip = True
        for i in range(len(edges)):
            ei_l, ei_r = edges[i,:]
            if ((ei_l == e1_l and ei_r == e2_r) or (ei_l == e2_l and ei_r == e1_r)
            or (ei_l == e1_r and ei_r == e2_l) or (ei_l == e2_r and ei_r == e1_l)):
                can_flip = False
                break
        if can_flip:
            edges[index1, 1] = e2_r
            edges[index2, 0] = e2_l
            edges[index2, 1] = e1_r



@njit
def rewire_mono_large(edges, n_rewire):
    """
    Rewires a single class network specified by edges in place!
    There are n_rewire steps of rewiring attempted.

    This function is optimized for larger networks it does dictionary lookups to avoid multi-edges
    """

    delta = len(edges)
    neigh = Dict()
    neigh[0] = List([-1])
    del neigh[0]
    for l,r in edges:
        if l not in neigh:
            tmp = List([-1])
            tmp.pop()
            neigh[l] = tmp
        if r not in neigh:
            tmp = List([-1])
            tmp.pop()
            neigh[r] = tmp
        neigh[l].append(r)
        neigh[r].append(l)

    # start:
    #  e1_l <-> e1_r
    #  e2_l <-> e2_r
    # after
    #  e1_l <-> e2_r
    #  e2_l <-> e1_r

    for _ in range(n_rewire):
        index1 = np.random.randint(0, delta)
        offset = np.random.randint(1, delta)
        i2_1 = np.random.randint(0, 2)
        i2_2 = 1 - i2_1
        index2 = (index1 + offset) % (delta)
        e1_l, e1_r = edges[index1,:]
        e2_l = edges[index2, i2_1]
        e2_r = edges[index2, i2_2]


        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue

        can_flip = True
        if e2_r in neigh[e1_l] or e1_r in neigh[e2_l]:
            can_flip = False

        if can_flip:
            edges[index1, 1] = e2_r
            edges[index2, 0] = e2_l
            edges[index2, 1] = e1_r
            neigh[e1_l].remove(e1_r)
            neigh[e1_r].remove(e1_l)

            neigh[e2_l].remove(e2_r)
            neigh[e2_r].remove(e2_l)

            neigh[e1_l].append(e2_r)
            neigh[e2_r].append(e1_l)

            neigh[e2_l].append(e1_r)
            neigh[e1_r].append(e2_l)


def rewire_bipartite(edges, lower, upper, n_rewire):
    """rewires a two class graph

    notice that also a one class _directed_ graph is a two class graph
    """
    if upper-lower < 2:
        raise ValueError

    _rewire_bipartite_small(edges[lower:upper], n_rewire)
    #print(edges[lower:upper])

@njit
def _rewire_bipartite_small(edges, n_rewire):
    """ Rewires a bipartite network specified in edges

    This is optimized for smaller networks. It uses linear search to avoid multi edges
    """


    # can do further optimization because the left side is always in a block
    #  => can limit search range

    delta = len(edges)


    for _ in range(n_rewire):
        index1 = np.random.randint(0, delta)
        offset = np.random.randint(1, delta)
        index2 = (index1 + offset) % (delta)
        e1_l, e1_r = edges[index1,:]
        e2_l, e2_r = edges[index2 ,:]

        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue

        can_flip = True
        for i in range(len(edges)):
            ei_l, ei_r = edges[i,:]
            if (ei_l == e1_l and ei_r == e2_r) or (ei_l == e2_l and ei_r == e1_r):
                can_flip = False
                break
        if can_flip:
            edges[index1, 1] = e2_r
            edges[index2, 1] = e1_r


@njit
def _rewire_bipartite_large(edges, n_rewire):
    """ Rewires a bipartite network specified in edges

    This is optimized for larger networks and uses a dictionary lookup to avoid multi edges
    """

    delta = len(edges)
    neigh = Dict()
    neigh[0] = List([-1])
    del neigh[0]
    for l,r in edges:
        if l not in neigh:
            tmp = List([-1])
            tmp.pop()
            neigh[l] = tmp
        neigh[l].append(r)

    for _ in range(n_rewire):
        index1 = np.random.randint(0, delta)
        offset = np.random.randint(1, delta)
        index2 = (index1 + offset) % (delta)
        e1_l, e1_r = edges[index1,:]
        e2_l, e2_r = edges[index2 ,:]

        if (e1_r == e2_r) or (e1_l == e2_l): # swap would do nothing
            continue

        if (e1_l == e2_r) or (e1_r == e2_l): # no self loops after swab
            continue

        can_flip = True
        if e2_r in neigh[e1_l] or e1_r in neigh[e2_l]:
            can_flip = False

        if can_flip:
            edges[index1, 1] = e2_r
            edges[index2, 1] = e1_r

            neigh[e1_l].remove(e1_r)
            neigh[e2_l].remove(e2_r)
            neigh[e1_l].append(e2_r)
            neigh[e2_l].append(e1_r)

@njit
def _get_block_indices(arr_in, is_dead, out):
    """Returns the indices of block changes in arr
    input [4,4,2,2,3,5]
    output = [0,2,4,5,6]
    lower inclusive, upper exclusive

    """
    indices = np.arange(len(arr_in))[~is_dead]
    arr = arr_in[~is_dead]
    #print(arr)
    #print(indices)
    #print()
    if len(arr)==0:
        return out[:0, :]
    last_val = arr[0]
    out[0,0] = indices[0]
    n=0
    last_index=0
    for i, val in zip(indices, arr):

        if val == last_val:
            last_index=i
            continue
        else:
            last_val=val
            out[n,1]=last_index+1
            out[n+1,0]=i
            n+=1
            last_index=i
    out[n,1] = last_index+1
    if out[n,1]-out[n,0]>1:
        n+=1
    return out[:n,:]

def check_blocks(out_arr):
    block_lengths = out_arr[1:]-out_arr[0:len(out_arr)-1]
    inds = block_lengths <= 1
    assert np.all(block_lengths>1), f"{block_lengths[inds]} {out_arr[1:][inds]}"


#@njit
def get_block_indices(edges_classes, dead_arrs):
    """Returns an arr that contains the start and end of blocks"""
    out = []
    for arr, dead_arr in zip(edges_classes.T, dead_arrs):

        out_arr =_get_block_indices(arr, dead_arr, np.empty((len(arr),2), dtype=np.int32))
        #print(arr)
        #print(dead_arr)
        #c=45673
        #d=3
        #print(arr[c-d:c+d])
        #print(dead_arr[c-d:c+d])
        #print(out_arr)

        #check_blocks(out_arr)
        #print(dead_arr.sum()+np.sum(out_arr[:,1]-out_arr[:,0]))
        #print("block", np.sum(out_arr[:,1]-out_arr[:,0]))
        #print(len(edges_classes))
        out.append(out_arr)


    return out



#@njit
def rewire_fast(edges, edge_class, current_mono, block, is_directed):
    """This function rewires the edges in place thereby preserving the WL classes

    This function assumes edges to be ordered according to the classes

    """
    # assumes edges to be ordered


    #deltas=[]
    for i in range(len(block)):

        lower = block[i,0]
        upper = block[i,1]
        delta=upper-lower
        #if delta<=1:
        #    continue
        #deltas.append(delta)
        current_class = edge_class[lower]

        if (not is_directed) and (current_mono.get(current_class, False)):
            #print(f"---{delta}")
            if delta< 50:
                rewire_mono_small(edges[lower:upper], np.random.randint(delta, 2*delta))
            else:
                rewire_mono_large(edges[lower:upper], np.random.randint(delta, 2*delta))
        else:
            #print(f"-{delta}")
            if delta< 50:
                _rewire_bipartite_small(edges[lower:upper], np.random.randint(delta, 2*delta))
            else:
                _rewire_bipartite_large(edges[lower:upper], np.random.randint(delta, 2*delta))