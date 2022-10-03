import torch

from .naive_impl import build_naive_neighborhood
from .geometry import compute_cell_shifts
from .linked_cell import build_linked_cell_neighborhood


def strict_nl(
    cutoff: float,
    pos: torch.Tensor,
    cell: torch.Tensor,
    mapping: torch.Tensor,
    batch_mapping: torch.Tensor,
    shifts_idx: torch.Tensor,
):
    """Apply a strict cutoff to the neighbor list defined in mapping.

    Parameters
    ----------
    cutoff : _type_
        _description_
    pos : _type_
        _description_
    cell : _type_
        _description_
    mapping : _type_
        _description_
    batch_mapping : _type_
        _description_
    shifts_idx : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    cell_shifts = compute_cell_shifts(cell, shifts_idx, batch_mapping)
    if cell_shifts is None:
        d2 = (pos[mapping[0]] - pos[mapping[1]]).square().sum(dim=1)
    else:
        d2 = (
            (pos[mapping[0]] - pos[mapping[1]] - cell_shifts)
            .square()
            .sum(dim=1)
        )

    mask = d2 < cutoff * cutoff
    mapping = mapping[:, mask]
    mapping_batch = batch_mapping[mask]
    shifts_idx = shifts_idx[mask]
    return mapping, mapping_batch, shifts_idx


@torch.jit.script
def compute_neighborlist_n2(
    cutoff: float,
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    self_interaction: bool = False,
):
    # with torch.cuda.amp.autocast():
    n_atoms = torch.bincount(batch)
    mapping, batch_mapping, shifts_idx = build_naive_neighborhood(
        pos, cell, pbc, cutoff, n_atoms, self_interaction
    )
    mapping, mapping_batch, shifts_idx = strict_nl(
        cutoff, pos, cell, mapping, batch_mapping, shifts_idx
    )
    return mapping, mapping_batch, shifts_idx


@torch.jit.script
def compute_neighborlist(
    cutoff: float,
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor,
    self_interaction: bool = False,
):
    n_atoms = torch.bincount(batch)
    mapping, batch_mapping, shifts_idx = build_linked_cell_neighborhood(
        pos, cell, pbc, cutoff, n_atoms, self_interaction
    )

    mapping, mapping_batch, shifts_idx = strict_nl(
        cutoff, pos, cell, mapping, batch_mapping, shifts_idx
    )
    return mapping, mapping_batch, shifts_idx
