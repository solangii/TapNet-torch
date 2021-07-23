import numpy as np
from numpy.linalg import svd

from numpy.linalg import svd as svd_gpu


def rank(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, tol=1e-13):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    if len(A.shape) == 2:
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
    elif len(A.shape) == 3:
        nnz = (s >= tol).sum(axis=-1)
        nnz = max(nnz)
        ns = np.transpose(vh[:, nnz:, :].conj(), axes=[0, 2, 1])
    return ns

"""

def nullspace_gpu(A, tol=1e-13):
    A = cp.atleast_2d(A)
    u, s, vh = svd_gpu(A)
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
"""


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def experiment_name_generator(config):
    #few_shot_setting = f"{str(config.n_way)}way_{str(config.n_shot)}shot_{str(config.n_query)}query"
    #training_params = f"{str(config.glocal_layers)}Layers_{str(config.graph_node_dim)}dim"
    #name = f"{few_shot_setting}_{training_params}"
    return None