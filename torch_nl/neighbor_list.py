from typing import Optional
import torch

from .image import compute_images
from .geometry import compute_distances, compute_cell_shifts


def strict_nl(rcut, pos, cell, mapping, batch_mapping,  shifts_idx):
    cell_shifts = compute_cell_shifts(cell, shifts_idx, batch_mapping)
    if cell_shifts is None:
        d2 = (pos[mapping[0]] - pos[mapping[1]]).square().sum(dim=1)
    else:
        d2 = (pos[mapping[0]] - pos[mapping[1]] - cell_shifts).square().sum(dim=1)

    mask = d2 <= rcut*rcut
    mapping = mapping[:, mask]
    mapping_batch = batch_mapping[mask]
    shifts_idx = shifts_idx[mask]
    return mapping, mapping_batch, shifts_idx # , d2[mask].sqrt()


def compute_nl_n2(rcut, pos, cell, pbc, batch, self_interaction):
    n_atoms = torch.bincount(batch)
    mapping, batch_mapping, shifts_idx = compute_images(pos,
                                     cell, pbc, rcut,
                                     n_atoms, self_interaction)
    mapping, mapping_batch, shifts_idx = strict_nl(rcut, pos, cell, mapping, batch_mapping,  shifts_idx)
    return mapping, mapping_batch, shifts_idx
