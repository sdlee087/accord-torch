import torch
import numpy as np
from scipy import sparse

def scipy_csr_to_torch_coo(csr, dtype=torch.float64, device = 'cpu'):
    coo = csr.tocoo()
    return torch.sparse_coo_tensor(np.array([coo.row, coo.col]), coo.data, size = torch.Size(csr.shape), dtype = dtype, device = device)

def torch_coo_to_scipy_csr(coo):
    indices = coo.indices().numpy()
    values = coo.values().numpy()
    shape = coo.shape
    return sparse.csr_matrix((values, (indices[0], indices[1])), shape = shape)

def non_negative(x):
    return x if x >= 0 else 0