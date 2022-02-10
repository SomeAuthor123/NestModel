
import warnings
import scipy
import numpy as np
import networkx as nx
import graph_tool.all as gt

def get_stochastic_matrix(G, sparse=False):
    is_nx = isinstance(G, (nx.Graph,nx.DiGraph))
    if sparse or not is_nx:
        if is_nx:
            order = np.fromiter(G, np.int32, count=len(G.nodes))
            assert np.all(np.diff(order) >= 0), "The node labels are not sorted!"
            # get adjacency matrix from networkx
            M = nx.to_scipy_sparse_matrix(G)

        else:
            # get adjacency from gt
            M = gt.adjacency(G)
        #print(M.dtype)

        # get the normalisation constant (might contain dead ends)
        norm = np.array(np.sum(M, axis=1)).flatten()

        # make dead ends into spider traps
        M = M + scipy.sparse.diags(np.array(norm==0,dtype=np.double))
        #print(M.dtype)
        norm[norm==0]=1 # spider traps have been removed

        # Normalise the stochastic matrix
        norm=1/norm
        D = scipy.sparse.diags(norm)
        M = D * M

        return M
    else:
        M = nx.to_numpy_array(G)
        norm = np.sum(M, axis=1).flatten() # get out degree
        M += np.diag(norm==0) # add spider traps

        norm[norm==0]=1 # spider traps have been removed


        M =  (M.T / norm).T # normalise matrix
        return np.array(M)



def pagerank(G, alpha = 0.85, S = None, max_iter = 1000, eps = 1e-14, sparse=False):
    """
    G: Graph
    beta: teleportation parameter
    S: teleport set for biased random walks
    max_iter: maximum number of iterations.
    eps: convergence parameter
    -> break iteration if L1-norm of difference between old and new pagerank vectors are smaller than eps
    """

    M = get_stochastic_matrix(G, sparse)

    n = M.shape[0]

    if S is None:
        e = np.ones(n)
        s = n
    else:
        e = np.zeros(n)
        e[S] = 1
        s = np.sum(e)
    e = e/s
    v = np.zeros(n)
    v_new = np.full(shape=n, fill_value =1.0/n)
    i = 0

    while np.linalg.norm(v-v_new,1) > eps and i < max_iter:
        v = v_new
        v_new = alpha * v @ M   + (1.0 - alpha) * e
        i += 1

    if i >= max_iter:
         warnings.warn("Power iteration has not converged up to specified tolerance")

    return v_new, M


def check_convergence(v, M, alpha):
    """ Small helper function that checks for convergence of pagerank"""
    n=len(v)
    e = (1.0 - alpha)/n
    return np.sum(np.abs(alpha * v @ M   + e - v))


def to_arr2(G, v):
    arr = np.zeros(len(G.nodes),dtype=np.double)
    for i,val in zip(range(len(G.nodes)),v.values()):
        arr[i]=val
    return arr


def all_pagerank(G, version ,alpha, max_iter, epsilon, return_err=False):
    """Computes the pagerank either using a custom implementation or the implementation from graph packages"""
    M = None
    if version=="mine":
        pagerank_vec, M = pagerank(G, alpha=alpha, max_iter=max_iter, eps=epsilon, sparse=True)
    elif version=="theirs":
        is_nx = isinstance(G, (nx.Graph,nx.DiGraph))
        if is_nx:
            pagerank_vec = to_arr2(G, nx.pagerank(G, max_iter=max_iter, alpha=alpha, tol=epsilon))
        else:
            pagerank_vec = np.array(gt.pagerank(G, epsilon=epsilon, max_iter=max_iter, damping=alpha).get_array())
    else:
        raise ValueError("Invalid version")
    if return_err:
        if M is None:
            M = get_stochastic_matrix(G, sparse=True)
        return pagerank_vec, check_convergence(pagerank_vec, M, alpha)
    else:
        return pagerank_vec