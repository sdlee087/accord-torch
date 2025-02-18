import torch
import numpy as np
from scipy import sparse

def scipy_csr_to_torch_coo(csr, dtype=torch.float64, device = 'cpu'):
    coo = csr.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype = dtype)
    values = torch.tensor(coo.data, dtype = dtype)
    return torch.sparse_coo_tensor(indices, values, torch.Size(csr.shape)).to(device)

def torch_coo_to_scipy_csr(coo):
    indices = coo.indices().numpy()
    values = coo.values().numpy()
    shape = coo.shape
    return sparse.csr_matrix((values, (indices[0], indices[1])), shape = shape)