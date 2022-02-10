import numpy as np
from numba import njit
from numba import prange
from numba import int64, uint32, float64, uint64

from cc_model.fast_wl import to_in_neighbors


@njit([(int64[:,:], int64[:], float64, int64, float64), (uint32[:,:], uint32[:], float64, int64, float64)])
def pagerank_numba_sym(edges, degrees, epsilon, max_iter, alpha):
    """
    G: Graph
    beta: teleportation parameter
    S: teleport set for biased random walks
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """
    num_nodes = len(degrees)
    factors = alpha/degrees

    v = np.zeros(num_nodes)
    v_tmp = np.zeros(num_nodes)
    teleport_factor = (1.0 - alpha) / num_nodes
    v_old = np.full(shape=num_nodes, fill_value =1.0/num_nodes)
    num_iter = 0
    last_err = 1
    while last_err > epsilon and num_iter < max_iter:
        for i in range(num_nodes):
            v_old[i] = v[i]
        for i in range(num_nodes):
            v_tmp[i] = v[i]*factors[i]


        # clear old vector
        for i in range(num_nodes):
            v[i] = teleport_factor

        for i in range(edges.shape[0]):
            e1 = edges[i,0]
            e2 = edges[i,1]
            v[e1] += v_tmp[e2]
            v[e2] += v_tmp[e1]

        # compute error
        last_err = 0
        for i in range(num_nodes):
            last_err += abs(v[i] - v_old[i])
        num_iter += 1

    if num_iter >= max_iter:
        last_err = - last_err
         #warnings.warn("Power iteration has not converged up to specified tolerance")

    return v, last_err



@njit([(uint64[:,:], int64[:], int64[:], float64, int64, float64), (uint32[:,:], uint32[:], int64[:], float64, int64, float64)], parallel=True)
def pagerank_numba_dir(edges, out_degrees, dead_ends, epsilon, max_iter, alpha):
    """
    G: Graph
    beta: teleportation parameter
    S: teleport set for biased random walks
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """
    num_nodes = len(out_degrees)
    factors = alpha/out_degrees
    #print(edges.shape)
    #edges = np.vstack((edges[:,1], edges[:,0])).T
    #print(edges.shape)
    startings, in_neighbors, _ = to_in_neighbors(edges)

    v = np.full(shape=num_nodes, fill_value = 1.0/num_nodes)
    v_tmp = np.zeros(num_nodes)
    #teleport_factor = float64((1.0 - alpha) / num_nodes)
    v_old = np.full(shape=num_nodes, fill_value =1.0/num_nodes)
    num_iter = 0
    last_err = 1
    while last_err > epsilon and num_iter < max_iter:
        danglesum = float64(0.0)
        for i in prange(len(dead_ends)):
            danglesum += v[dead_ends[i]]
        danglesum = (alpha * danglesum + (1.0 - alpha))/num_nodes
        # create copy of old vector
        for i in prange(num_nodes):
            v_old[i] = v[i]

        # initialize
        for i in prange(num_nodes):
            v_tmp[i] = v[i]*factors[i]

        # reset vector with constant values
        for i in prange(num_nodes):
            v[i] = danglesum# + teleport_factor

        for i in prange(len(startings)-1):
            lb = startings[i]
            ub = startings[i+1]
            for j in range(lb, ub):
                neib = in_neighbors[j]
                v[i] += v_tmp[neib]

        #for i in prange(len(dead_ends)):
        #    node_id = dead_ends[i]
        #    v[node_id] += v_tmp[node_id]

        # compute error
        last_err = 0
        for i in prange(num_nodes):
            last_err += abs(v[i] - v_old[i])
        num_iter += 1
    #s = 0
    #print(num_iter)

    #for i in prange(num_nodes):
    #    s+=abs(v[i])
    #

    if num_iter >= max_iter:
        last_err = - last_err
         #warnings.warn("Power iteration has not converged up to specified tolerance")

    return v, last_err
