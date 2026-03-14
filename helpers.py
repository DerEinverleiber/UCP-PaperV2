import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.linalg import inv


def inverse_reduced_graph_laplacian(graph: csr_matrix, reference_idx) -> csr_matrix:
    graph = graph.toarray()

    graph_laplacian = laplacian(graph)
    reduced_graph_laplacian = np.delete(graph_laplacian, reference_idx, axis=0)
    reduced_graph_laplacian = np.delete(reduced_graph_laplacian, reference_idx, axis=1)
    return inv(reduced_graph_laplacian)