import numpy as np
from scipy import sparse
import torch

from .util import *

def pyaccord(X, lamb, cfg, logger, part = (0,0), omega_old = None, label = 0, device = 'cpu'):
    tau_init = float(cfg["tau_init"])
    outer_iter = int(cfg["max_outer"])
    inner_iter = int(cfg["max_inner"])
    eps = float(cfg["eps"])
    beta = float(cfg["beta"])
    log_interval = int(cfg["log_interval"])
    flt = torch.float64 if cfg["float64"] else torch.float32

    n, p = X.shape

    # compute for all entries in default
    d_off = 0
    b_size = p

    if part[1] > 0: 
        b_size = part[1] - part[0]
        d_off = part[0]
        if d_off + b_size > p:
            b_size = p - d_off

    XT = X.transpose(0,1)

    if omega_old is None:
        if cfg["resume"] is not None:
            omega_old = torch.from_numpy(sparse.load_npz("%s_%s_%s.npz" % (cfg["resume"], round(lamb*100), label)).todense()).to(dtype = flt, device = device)
        else:  
            omega_old = torch.zeros(b_size, p,dtype=flt).to(device)
            omega_old.diagonal(d_off).copy_(torch.ones(b_size, dtype=flt, device = device))

    omega = torch.clone(omega_old)

    logger.info("Setup Complete")
    logger.info("Iterate size: (%d, %d)" % omega.size())
    logger.info("L1 penalty: %.2f" % lamb)
    logger.info("Starting from: %d" % d_off)
    logger.info("Omega_nnz: %d" % torch.sum(omega != 0.0))

    # main loop
    Y = torch.matmul(omega_old, XT)
    g_old = 0.5 * torch.norm(Y, p='fro')**2 / n

    for i in range(outer_iter):
        tau = tau_init
        grad = torch.matmul(Y, X)/n

        for j in range(inner_iter):
            # update omega with current tau
            o_tilde = - tau*grad + omega_old
            omega_d = (o_tilde.diagonal(d_off) + torch.sqrt((o_tilde.diagonal(d_off))**2 + 4.0*tau)) * 0.5
            omega = torch.nn.functional.softshrink(o_tilde, lambd=(lamb * tau))
            omega.diagonal(d_off).copy_(omega_d)

            # check progress
            Y = torch.matmul(omega, XT)
            g_new = 0.5 * torch.norm(Y, p='fro')**2 / n

            D = omega - omega_old
            Q = g_old + torch.sum(D * grad) + torch.norm(D, p='fro')**2 / (2.0 * tau)
            nnz = torch.sum(omega != 0.0)
            if g_new < Q:
                #logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, g_new=g_new, Q=Q, nnz=nnz))
                break
            tau *= beta

        if torch.max(abs(D)).item() <= eps:
            g_old = 0.5 * torch.norm(Y, p='fro')**2 / n
            f = g_old -torch.log(omega_d).sum() + lamb*(torch.linalg.vector_norm(omega, ord = 1) - torch.sum(omega_d))
            logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, f={f:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, f=f, g_new=g_new, Q=Q, nnz=nnz))
            break
        
        g_old = 0.5 * torch.norm(Y, p='fro')**2 / n
        if i % log_interval == 0:
            f = g_old -torch.log(omega_d).sum() + lamb*(torch.linalg.vector_norm(omega, ord = 1) - torch.sum(omega_d))
            logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, f={f:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, f=f, g_new=g_new, Q=Q, nnz=nnz))

        if i < outer_iter - 1:
            omega_old, omega = omega, omega_old

    # Save results
    logger.info("Saving Results")
    sparse.save_npz("%s_%s_%s" % (cfg["out_file"], round(lamb*100), label), sparse.csr_matrix(omega.cpu().numpy()))
    logger.info("Saving Complete!")
    return omega

def pyaccord_sp(X, lamb, cfg, logger, part = (0,0), omega_old = None, label = 0, device = 'cpu'):
    tau_init = float(cfg["tau_init"])
    outer_iter = int(cfg["max_outer"])
    inner_iter = int(cfg["max_inner"])
    eps = float(cfg["eps"])
    beta = float(cfg["beta"])
    log_interval = int(cfg["log_interval"])
    flt = torch.float64 if cfg["float64"] else torch.float32

    n, p = X.shape

    # compute for all entries in default
    d_off = 0
    b_size = p

    if part[1] > 0: 
        b_size = part[1] - part[0]
        d_off = part[0]
        if d_off + b_size > p:
            b_size = p - d_off

    XT = X.transpose(0,1)

    if omega_old is None:
        if cfg["resume"] is not None:
            omega_old = scipy_csr_to_torch_coo(sparse.load_npz("%s_%s_%s.npz" % (cfg["resume"], round(lamb*100), label)), dtype = flt, device = device)
        else:  
            omega_old = torch.sparse_coo_tensor(torch.tensor([range(b_size), range(d_off,d_off + b_size)], dtype = torch.int64), torch.ones(b_size, dtype=flt), [b_size, p], device = device)
        
    omega = torch.clone(omega_old)
    shape = omega.shape

    logger.info("Setup Complete")
    logger.info("Iterate size: (%d, %d)" % omega.size())
    logger.info("L1 penalty: %.2f" % lamb)
    logger.info("Starting from: %d" % d_off)
    logger.info("Omega_nnz: %d" % len(omega_old.coalesce().values()))

    # main loop
    Y = torch.matmul(omega_old, XT)
    g_old = 0.5 * torch.norm(Y, p='fro')**2 / n
    f_old = float('inf')

    for i in range(outer_iter):
        tau = tau_init
        grad = torch.matmul(Y, X)/n

        for j in range(inner_iter):
            # update omega with current tau
            o_tilde = - tau*grad + omega_old
            omega_d = (o_tilde.diagonal(d_off) + torch.sqrt((o_tilde.diagonal(d_off))**2 + 4.0*tau))*0.5
            o_tilde.diagonal(d_off).copy_(0.)

            pos_mask = o_tilde >= lamb * tau
            o_tilde[pos_mask] -= lamb*tau

            neg_mask = o_tilde <= -lamb * tau
            o_tilde[neg_mask] += lamb*tau

            pos_mask |= neg_mask
            pos_mask.diagonal(d_off).copy_(True)
            o_tilde.diagonal(d_off).copy_(omega_d)

            indices = pos_mask.nonzero(as_tuple = False).t()
            omega = torch.sparse_coo_tensor(indices, o_tilde[pos_mask], shape, device = device).coalesce()

            # check progress
            Y = torch.matmul(omega, XT)
            g_new = 0.5 * torch.norm(Y, p='fro')**2 / n

            D = (omega - omega_old).coalesce()
            #D = (omega - omega_old).to_sparse_csr()
            Q = g_old + torch.sum((D * grad).values()) + torch.norm(D.values(), p='fro')**2 / (2.0 * tau)
            nnz = len(indices[0])
            if g_new < Q:
                #logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, g_new=g_new, Q=Q, nnz=nnz))
                break
            tau *= beta

        # check for progression in the iterate
        if torch.max(abs(D.values())).item() <= eps:
            g_old = 0.5 * torch.norm(Y, p='fro')**2 / n
            f = g_old -torch.log(omega_d).sum() + lamb*(torch.linalg.vector_norm(omega.coalesce().values(), ord = 1) - torch.sum(omega_d))
            logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, f={f:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, f=f, g_new=g_new, Q=Q, nnz=nnz))

            logger.info("Saving Results")
            sparse.save_npz("%s_%s_%s" % (cfg["out_file"], round(lamb*100), label), torch_coo_to_scipy_csr(omega.cpu().coalesce()))
            logger.info("Saving Complete!")
            return omega, True
        
        g_old = 0.5 * torch.norm(Y, p='fro')**2 / n
        # check for progression in the objective function
        if i % log_interval == 0:
            f_new = g_old -torch.log(omega_d).sum() + lamb*(torch.linalg.vector_norm(omega.coalesce().values(), ord = 1) - torch.sum(omega_d))
            logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, f={f:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, f=f_new, g_new=g_new, Q=Q, nnz=nnz))
            if ((f_old - f_new)/torch.abs(f_new)).item() <= eps:
                logger.info("Saving Results")
                sparse.save_npz("%s_%s_%s" % (cfg["out_file"], round(lamb*100), label), torch_coo_to_scipy_csr(omega.cpu().coalesce()))
                logger.info("Saving Complete!")
                return omega, True
            f_old = f_new

        if i < outer_iter - 1:
            omega_old, omega = omega, omega_old

    # Save results
    logger.info("Maximum loop reached")
    logger.info("Saving Results")
    sparse.save_npz("%s_%s_%s" % (cfg["out_file"], round(lamb*100), label), torch_coo_to_scipy_csr(omega.cpu().coalesce()))
    logger.info("Saving Complete!")
    return omega, False

def diag_intersection_sum(omega_d, a, b, c, d):
    diag_ind = torch.arange(a,b, device = omega_d.device)
    diag_mask = (diag_ind >= c) & (diag_ind < d)
    values = omega_d[diag_mask]
    return torch.sum(values) if values.numel() > 0 else 0.0

def pyaccord_sp_block(X, lamb, cfg, logger, part = (0,0), omega_old = None, label = 0, device = 'cpu'):
    tau_init = float(cfg["tau_init"])
    outer_iter = int(cfg["max_outer"])
    inner_iter = int(cfg["max_inner"])
    eps = float(cfg["eps"])
    beta = float(cfg["beta"])
    log_interval = int(cfg["log_interval"])
    flt = torch.float64 if cfg["float64"] else torch.float32

    n, p = X.shape
    split = cfg["split"]
    block_n = len(lamb)
    split = torch.tensor([0] + split + [p], device = device)
    assert block_n + 1 == len(split)

    # compute for all entries in default
    d_off = 0
    b_size = p

    if part[1] > 0: 
        b_size = part[1] - part[0]
        d_off = part[0]
        if d_off + b_size > p:
            b_size = p - d_off

    XT = X.transpose(0,1)

    if omega_old is None:
        if cfg["resume"] is not None:
            omega_old = scipy_csr_to_torch_coo(sparse.load_npz("%s_%s_%s_%s.npz" % (cfg["resume"], round(lamb[0]*100), round(lamb[1]*100), label)), dtype = flt, device = device)
        else:  
            omega_old = torch.sparse_coo_tensor(torch.tensor([range(b_size), range(d_off,d_off + b_size)], dtype = torch.int64), torch.ones(b_size, dtype=flt), [b_size, p], device = device)
        
    omega = torch.clone(omega_old)
    shape = omega.shape

    logger.info("Setup Complete")
    logger.info("Iterate size: (%d, %d)" % omega.size())
    logger.info(f"L1 penalty: {lamb}")
    logger.info("Starting from: %d" % d_off)
    logger.info("Omega_nnz: %d" % len(omega_old.coalesce().values()))

    # main loop
    Y = torch.matmul(omega_old, XT)
    g_old = 0.5 * torch.norm(Y, p='fro')**2 / n
    f_old = float('inf')

    for i in range(outer_iter):
        tau = tau_init
        grad = torch.matmul(Y, X)/n

        for j in range(inner_iter):
            # update omega with current tau
            o_tilde = - tau*grad + omega_old
            omega_d = (o_tilde.diagonal(d_off) + torch.sqrt((o_tilde.diagonal(d_off))**2 + 4.0*tau))*0.5
            o_tilde.diagonal(d_off).copy_(0.)
            block_indices = torch.bucketize(torch.arange(p, device = device), split, right=True) - 1
            thresholds = torch.tensor(lamb, device = device)[block_indices] * tau

            pos_mask = o_tilde >= thresholds
            neg_mask = o_tilde <= -thresholds

            o_tilde[pos_mask] -= torch.cat([thresholds[pos_mask[i]] for i in range(b_size)])
            o_tilde[neg_mask] += torch.cat([thresholds[neg_mask[i]] for i in range(b_size)])

            # pos_mask = torch.full(shape, False, dtype=torch.bool)
            # neg_mask = torch.full(shape, False, dtype=torch.bool)

            # for k in range(block_n):
            #     slice_k = slice(split[k], split[k+1])
            #     pos_mask[:, slice_k] = o_tilde[:, slice_k] >= lamb[k] * tau
            #     o_tilde[:, slice_k][pos_mask[:, slice_k]] -= lamb[k] * tau

            #     neg_mask[:, slice_k] = o_tilde[:, slice_k] <= -lamb[k] * tau
            #     o_tilde[:, slice_k][neg_mask[:, slice_k]] += lamb[k] * tau

            pos_mask |= neg_mask
            pos_mask.diagonal(d_off).copy_(True)
            o_tilde.diagonal(d_off).copy_(omega_d)

            indices = pos_mask.nonzero(as_tuple = False).t()
            omega = torch.sparse_coo_tensor(indices, o_tilde[pos_mask], shape, device = device).coalesce()

            # check progress
            Y = torch.matmul(omega, XT)
            g_new = 0.5 * torch.norm(Y, p='fro')**2 / n

            D = (omega - omega_old).coalesce()
            #D = (omega - omega_old).to_sparse_csr()
            Q = g_old + torch.sum((D * grad).values()) + torch.norm(D.values(), p='fro')**2 / (2.0 * tau)
            nnz = len(indices[0])
            if g_new < Q:
                #logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, g_new=g_new, Q=Q, nnz=nnz))
                break
            tau *= beta

        # check for progression in the iterate
        if torch.max(abs(D.values())).item() <= eps:
            g_old = 0.5 * torch.norm(Y, p='fro')**2 / n
            f = g_old -torch.log(omega_d).sum()
            for k in range(block_n):
                indices = omega.indices()
                mask = (indices[1] >= split[k]) & (indices[1] < split[k+1])
                f += lamb[k] * (torch.linalg.vector_norm(omega.values()[mask], ord = 1) - diag_intersection_sum(omega_d, d_off, d_off + b_size, split[k], split[k+1])) 
            logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, f={f:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, f=f, g_new=g_new, Q=Q, nnz=nnz))

            logger.info("Saving Results")
            sparse.save_npz("%s_%s_%s_%s" % (cfg["out_file"], round(lamb[0]*100), round(lamb[1]*100), label), torch_coo_to_scipy_csr(omega.cpu().coalesce()))
            logger.info("Saving Complete!")
            return omega, True
        
        g_old = 0.5 * torch.norm(Y, p='fro')**2 / n
        if i % log_interval == 0:
            f_new = g_old -torch.log(omega_d).sum() 
            for k in range(block_n):
                indices = omega.indices()
                mask = (indices[1] >= split[k]) & (indices[1] < split[k+1])
                f_new += lamb[k] * (torch.linalg.vector_norm(omega.values()[mask], ord = 1) - diag_intersection_sum(omega_d, d_off, d_off + b_size, split[k], split[k+1])) 
            logger.info("Round {i:03d}.{j:02d}: tau = {tau:10.4f}, f={f:10.4f}, g = {g_new:10.4f}, Q = {Q:10.4f}, nnz = {nnz:d}".format(i=i, j=j, tau=tau, f=f_new, g_new=g_new, Q=Q, nnz=nnz))
            if ((f_old - f_new)/torch.abs(f_new)).item() <= eps:
                logger.info("Saving Results")
                sparse.save_npz("%s_%s_%s_%s" % (cfg["out_file"], round(lamb[0]*100), round(lamb[1]*100), label), torch_coo_to_scipy_csr(omega.cpu().coalesce()))
                logger.info("Saving Complete!")
                return omega, True
            f_old = f_new

        if i < outer_iter - 1:
            omega_old, omega = omega, omega_old

    # Save results
    logger.info("Maximum loop reached")
    logger.info("Saving Results")
    sparse.save_npz("%s_%s_%s_%s" % (cfg["out_file"], int(lamb[0]*100), int(lamb[1]*100), label), torch_coo_to_scipy_csr(omega.cpu().coalesce()))
    logger.info("Saving Complete!")
    return omega, False