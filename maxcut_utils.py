from typing import Tuple, Iterable, Union

import cvxpy as cvx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from numpy.random import RandomState
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        fres =func(*args, **kwargs)
        dt = time.time() - t0
        return fres, dt
    return wrapper

@timeit
def goemans_williamson(graph: nx.Graph) -> Tuple[np.ndarray, float, float]:
    """
    The Goemans-Williamson algorithm for solving the maxcut problem.
    Ref:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
    Returns:
        np.ndarray: Graph coloring (+/-1 for each node)
        float:      The GW score for this cut.
        float:      The GW bound from the SDP relaxation
    """
    # Kudos: Originally implementation by Nick Rubin, with refactoring and
    # cleanup by Jonathon Ward and Gavin E. Crooks
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # Setup and solve the GW semidefinite programming problem
    psd_mat = cvx.Variable(laplacian.shape, PSD=True)
    obj = cvx.Maximize(cvx.trace(laplacian @ psd_mat))
    constraints = [cvx.diag(psd_mat) == 1]  # unit norm
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT)

    evals, evects = np.linalg.eigh(psd_mat.value)
    sdp_vectors = evects.T[evals > float(1.0E-6)].T

    # Bound from the SDP relaxation
    bound = np.trace(laplacian @ psd_mat.value)

    random_vector = np.random.randn(sdp_vectors.shape[1])
    random_vector /= np.linalg.norm(random_vector)
    colors = np.sign([vec @ random_vector for vec in sdp_vectors])
    score = colors @ laplacian @ colors.T

    return colors, score, bound


def graph_gen_regular(n, d, random_seed=0):
    """Generates a random d-regular graph with edges of unit weights"""
    g = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    return g

def graph_randomize(g, random_seed=0, scale=1.0):
    """Randomize weights of a grpah @g to uniform destribution with scale @scale
    """
    adjacency = nx.adjacency_matrix(g).todense().astype('float64')
    prng = RandomState(random_seed)
    random_m = prng.rand(*adjacency.shape)
    adjacency = np.multiply(adjacency, random_m)
    adjacency *= scale
    adjacency = (adjacency + adjacency.T) / 2
    g = nx.from_numpy_matrix(adjacency, create_using=nx.Graph)
    return g

def show_graph(graph: nx.Graph, node_map=None, labels=None, show_weights=False):
    layout = nx.spring_layout(graph)
    nx.draw(graph, layout, node_color=node_map, labels=labels)
    if show_weights is True:
        edge_labels = nx.get_edge_attributes(graph, "weight")
        print(edge_labels)
        nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels)
    plt.show()

    
if __name__ == '__main__':
    import numpy as np
    import networkx as nx
    from utils import run_gnn_training, get_gnn
    import torch
    import random
    import dgl

    seed_value = 2
    random.seed(seed_value)        # seed python RNG
    np.random.seed(seed_value)     # seed global NumPy RNG
    torch.manual_seed(seed_value)  # seed torch RNG

    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TORCH_DTYPE = torch.float32

    N = 40 # set graph size
    K = 5 # set range of graph degree
    dim_embedding = 100 # set gnn input embeding
    hidden_dim = int(dim_embedding/2)  # et gnn hidden embeding
    # NN learning hypers #
    number_epochs = int(1000)
    learning_rate = 1e-4
    PROB_THRESHOLD = 0.5

    # Early stopping to allow NN to train to near-completion
    tol = 1e-4          # loss must change by more than tol, or trigger
    patience = 100    # number early stopping triggers before breaking loop
    
    res = {}

    g = graph_gen_regular(N, K, seed_value)
    g = graph_randomize(g, seed_value)
    
    (partition_gw, score_gw, up_score_gw), dt_gw = goemans_williamson(g)

    print(partition_gw, score_gw, up_score_gw, dt_gw)
    
    
    graph_dgl = dgl.from_networkx(nx_graph=g)
    graph_dgl = graph_dgl.to(TORCH_DEVICE)
    laplacian = - np.array(nx.laplacian_matrix(g).todense())
    q_torch = torch.tensor(laplacian, dtype=TORCH_DTYPE, device=TORCH_DEVICE)

    # Establish pytorch GNN + optimizer
    opt_params = {'lr': learning_rate}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.5,
        'number_classes': 1,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs': number_epochs,
        'tolerance': tol,
        'patience': patience
    }

    net, embed, optimizer = get_gnn(N, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)


    dt_gnn, epoch, final_bitstring, best_bitstring = run_gnn_training(
        q_torch, graph_dgl, net, embed, optimizer, gnn_hypers['number_epochs'],
        gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'])

    partition_gnn_final = final_bitstring.detach().cpu().numpy().astype('float32').reshape((final_bitstring.numel()))
    partition_gnn_best = best_bitstring.detach().cpu().numpy().astype('float32').reshape((best_bitstring.numel()))
    
    score_final = -1.0 * partition_gnn_final @ laplacian @ partition_gnn_final.T
    score_best = -1.0 * partition_gnn_best @ laplacian @ partition_gnn_best.T


    