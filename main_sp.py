import numpy as np
from scipy import sparse
import torch
import torch.multiprocessing as mp
import argparse, yaml
from datetime import datetime

from src.pyaccord import pyaccord_sp
from src.logger import setup_logger
from src.util import scipy_csr_to_torch_coo

DEFAULT_CONFIG={
    "l1": [0.1],
    "eps": 1e-7,
    "tau_init": 0.5,
    "max_outer": 100,
    "max_inner": 10,
    "row_divide": 0,
    "label_start": 0,
    "log_interval": 1,
    "resume": None,
    "resume_from_whole": None,
    "total_process": 1,
    "CUDA": False,
    "float64": False
}

parser = argparse.ArgumentParser()
parser.add_argument("--config")
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.safe_load(f)

cfg = {**DEFAULT_CONFIG, **config}
flt = torch.float64 if cfg["float64"] else torch.float32
X = np.load(cfg["data_file"])
X = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)

def partition_range(start, stop, num_partitions):
    """
    Partitions a range(start, stop) into num_partitions roughly equal parts.
    """
    step = (stop - start) // num_partitions
    remainder = (stop - start) % num_partitions
    partitions = []
    current = start
    
    for i in range(num_partitions):
        extra = 1 if i < remainder else 0  # Distribute remainder across the first few partitions
        next_value = current + step + extra
        partitions.append((current, next_value))
        current = next_value
    
    return partitions

def run_process(queue, semaphore, row_id, lamb, parts):
    with semaphore:
        proc_id = queue.get()
        logger = setup_logger(cfg, proc_id)
        logger.info("Start Time: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        device = torch.device(f"cuda:{proc_id}" if cfg["CUDA"] and torch.cuda.is_available() else "cpu")

        omega_old = None
        if cfg["resume_from_whole"] is not None:
            whole_omega = sparse.load_npz(cfg["resume_from_whole"])
            omega_old = scipy_csr_to_torch_coo(whole_omega[parts[row_id][0]:parts[row_id][1],:], dtype=flt, device = device)

        logger.info(f"Process {row_id} Start.")
        pyaccord_sp(torch.from_numpy(X).type(flt).to(device), lamb, cfg, logger, part = parts[row_id], omega_old = omega_old, label = row_id + cfg["label_start"], device = device)
        logger.info(f"Process {row_id} Complete.")
        queue.put(proc_id)

if __name__ == "__main__":
    lambs = cfg["l1"]
    for lamb in lambs:
        if cfg["row_divide"] == 0:
            logger = setup_logger(cfg, 0)
            logger.info("Start Time: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            device = torch.device(f"cuda" if cfg["CUDA"] and torch.cuda.is_available() else "cpu")
            omega_old = None
            if cfg["resume_from_whole"] is not None:
                whole_omega = sparse.load_npz(cfg["resume_from_whole"])
                omega_old = scipy_csr_to_torch_coo(whole_omega, dtype=flt, device = device)
            pyaccord_sp(torch.from_numpy(X).type(flt).to(device), lamb, cfg, logger, omega_old = omega_old, device = device)
        else:
            mp.set_start_method('spawn')
            proc_queue = mp.Queue()
            for i in range(cfg["total_process"]):
                proc_queue.put(i)
            processes = []
            semaphore = mp.Semaphore(cfg["total_process"])
            parts = partition_range(cfg["row_min"], cfg["row_max"], cfg["row_divide"])
            for i in range(cfg["row_divide"]):
                p = mp.Process(target=run_process, args = (proc_queue, semaphore, i, lamb, parts))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        print("All processes completed.")
