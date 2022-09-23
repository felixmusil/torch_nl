import torch.profiler
import torch

torch.jit.set_fusion_strategy([("STATIC", 3), ("DYNAMIC", 3)])

import sys
sys.path.insert(0, '../')
import numpy as np

from torch_nl import compute_neighborlist, compute_neighborlist_n2, ase2data
from torch_nl.timer import timeit

from ase.build import molecule, bulk, make_supercell

device = 'cuda'
cutoff = 4
frame = bulk('Si', 'diamond', a=4, cubic=True)

frame = make_supercell(frame, 6*np.eye(3))



pos, cell, pbc, batch, n_atoms = ase2data([frame], device=device)


with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=20, warmup=20, active=2, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/local_scratch/musil/nl_n.prof'),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        ) as prof:
    for _ in range(50):
        mapping, mapping_batch, shifts_idx = compute_neighborlist_n2(cutoff, pos, cell, pbc, batch)
        prof.step()