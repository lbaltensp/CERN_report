import sys
import os

# Add root directory to sys.path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# Adding dll_path so that i can import h5py
from utils.add_dll_path import add_ivi_dll_path
add_ivi_dll_path()
import h5py

from mpi4py import MPI
import numpy as np
import argparse

# ─── PARSE ARGS ─────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=1000,
               help="number of ints per creator")
p.add_argument("--pairs", type=int, default=3,
               help="number of creator–writer pairs")
p.add_argument("--cb_config_list", type=str, default="*:1",
               help="number of aggregators")
p.add_argument("--file", default="parallel_test.h5",
               help="HDF5 filename")
args = p.parse_args()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
num_of_core_pairs = args.pairs
k                 = args.k
filename          = args.file
cb_config_list  = args.cb_config_list

# ─── MPI SETUP ──────────────────────────────────────────────────────────────────
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size == 2 * num_of_core_pairs

# ─── CORE ASSIGNMENT ────────────────────────────────────────────────────────────

import psutil
import time

core_map = {
    4:  [63, 62, 61, 60],                                 # Last 4 cores → CCD7
    8:  [63, 62, 61, 60, 59, 58, 57, 56],                 # All of CCD7
    12: [63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52], # CCD7 + part of CCD6
}

# Determine current core map based on total ranks
total_ranks = size
fixed_cores = core_map[total_ranks]

# Assign the current rank a fixed core
core_id = fixed_cores[rank]
proc = psutil.Process()
proc.cpu_affinity([core_id]) 
proc.nice(psutil.REALTIME_PRIORITY_CLASS)
actual = proc.cpu_affinity()   # Read back from OS
print(f"[Rank {rank}] pinned to core {core_id}, OS sees affinity = {actual}")

# time.sleep(10)

# ─── CREATOR / WRITER COMM ─────────────────────────────────────────────────────
if rank < num_of_core_pairs:
    # Creator
    data = np.random.randint(0, 1024, size=k, dtype=np.uint16)
    comm.Send(data, dest=rank+num_of_core_pairs, tag=0)
else:
    # Writer
    writer_id = rank - num_of_core_pairs
    data      = np.empty(k, dtype=np.uint16)
    comm.Recv(data, source=writer_id, tag=0)

comm.Barrier()

info = MPI.Info.Create()
# info.Set("cb_nodes", "2")
# info.Set("romio_cb_write", "enable")
# info.Set("romio_cb_read", "enable")  # optional
info.Set("cb_config_list", cb_config_list) 

# ─── TIMING START ──────────────────────────────────────────────────────────────
t0 = MPI.Wtime()

# ─── PARALLEL HDF5 I/O ON COMM_WORLD ────────────────────────────────────────────
with h5py.File(filename, 'w', driver='mpio', comm=comm, info=info) as f:
    # collective create
    dset = f.create_dataset('dataset', (num_of_core_pairs * k,), dtype=np.uint16)               

    if rank >= num_of_core_pairs:
        start = writer_id * k
        end   = start + k
        dset[start:end] = data                                                      
        
# ─── TIMING END & REDUCE ───────────────────────────────────────────────────────
t1 = MPI.Wtime()

dt = t1 - t0
sum_dt = comm.allreduce(dt, op=MPI.SUM)
max_dt = comm.allreduce(dt, op=MPI.MAX)
min_dt = comm.allreduce(dt, op=MPI.MIN)
avg_dt = sum_dt / size
if rank == 0:
    print(f"[COMM_WORLD] I/O time = {avg_dt:.6f} s")

# Get number of aggregators used (cb_nodes) using raw MPI-IO interface
amode = MPI.MODE_RDONLY
fh = MPI.File.Open(comm, filename, amode, info)
info_used = fh.Get_info()

if rank == 0:
    print("[COMM_WORLD] ROMIO hints used:")
    for key in info_used:
        print(f"  {key} = {info_used[key]}")

fh.Close()

# print(f"[Rank {rank}] I/O time = {dt:.6f} s")
# mpiexec -env ROMIO_PRINT_HINTS 1 -n 4 python all_ranks_version.py --k 10000 --pairs 2 --file test.h5
