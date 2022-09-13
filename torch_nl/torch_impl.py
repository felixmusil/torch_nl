from typing import Optional
import torch

from .image import compute_images

def compute_distances(
    pos: torch.Tensor,
    mapping: torch.Tensor,
    cell_shifts: Optional[torch.Tensor] = None,
):
    assert mapping.dim() == 2
    assert mapping.shape[0] == 2

    if cell_shifts is None:
        dr = pos[mapping[1]] - pos[mapping[0]]
    else:
        dr = pos[mapping[1]] - pos[mapping[0]] + cell_shifts

    return dr.norm(p=2, dim=1)

def compute_cell_shifts(cell, shifts_idx, batch_mapping):
    if cell is None:
        cell_shifts = None
    else:
        cell_shifts = torch.einsum("jn,jnm->jm", shifts_idx, cell.view(-1, 3, 3)[batch_mapping])
    return cell_shifts

def compute_strict_nl_n2(rcut, pos, cell, mapping, batch_mapping,  shifts_idx):
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
    mapping, mapping_batch, shifts_idx = compute_strict_nl_n2(rcut, pos, cell, mapping, batch_mapping,  shifts_idx)
    return mapping, mapping_batch, shifts_idx
