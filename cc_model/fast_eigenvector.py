import numpy as np
from numba import njit
from numba import int64, uint32, float64

@njit([(int64[:,:], int64, float64, int64), (uint32[:,:], int64, float64, int64, )])
def eigenvector_numba_sym(edges, num_nodes, epsilon, max_iter):
    """
    G: Graph
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """
    #num_nodes = len(degrees)

    v = 1/num_nodes * np.ones(num_nodes, dtype=np.float64)
    v_old = 1/num_nodes * np.ones(num_nodes, dtype=np.float64)
    num_iter = 0
    last_err = 1
    while last_err > epsilon and num_iter < max_iter:
        for i in range(num_nodes):
            v_old[i] = v[i]

        for i in range(edges.shape[0]):
            e1 = edges[i,0]
            e2 = edges[i,1]
            v[e1] += v_old[e2]
            v[e2] += v_old[e1]
        norm = np.sum(v)#np.sqrt(np.sum(np.square(v)))
        v/=norm

        # compute error
        last_err = 0
        for i in range(num_nodes):
            last_err += abs(v[i] - v_old[i])
        num_iter += 1

    if num_iter >= max_iter:
        last_err = - last_err
         #warnings.warn("Power iteration has not converged up to specified tolerance")

    return v, last_err


@njit([(int64[:,:], int64, float64, int64), (uint32[:,:], int64, float64, int64, )])
def eigenvector_numba_dir(edges, num_nodes, epsilon, max_iter):
    """
    G: Graph
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """
    #num_nodes = len(degrees)

    v = 1/num_nodes * np.ones(num_nodes, dtype=np.float64)
    v_old = 1/num_nodes * np.ones(num_nodes, dtype=np.float64)
    num_iter = 0
    last_err = 1
    while last_err > epsilon and num_iter < max_iter:
        for i in range(num_nodes):
            v_old[i] = v[i]



        for i in range(edges.shape[0]):
            e1 = edges[i,0]
            e2 = edges[i,1]
            #v[e1] += v_old[e2]
            v[e2] += v_old[e1]
        norm = np.sum(v)
        v/=norm

        # compute error
        last_err = 0
        for i in range(num_nodes):
            last_err += abs(v[i] - v_old[i])
        num_iter += 1

    if num_iter >= max_iter:
        last_err = - last_err
         #warnings.warn("Power iteration has not converged up to specified tolerance")

    return v, last_err
