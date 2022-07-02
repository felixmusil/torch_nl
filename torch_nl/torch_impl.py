from typing import Tuple, Optional
import torch

# @torch.jit.script
def strides_of(v: torch.Tensor) -> torch.Tensor:
    """TODO: add docs"""
    strides = torch.zeros(v.shape[0] + 1, dtype=torch.int64, device=v.device)
    strides[1:] = v
    strides = torch.cumsum(strides, dim=0)
    return strides


# @torch.jit.script
def compute_images(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch: torch.Tensor,
    n_atoms: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TODO: add doc"""

    cell = cell.view((-1, 3, 3)).to(torch.float64)
    pbc = pbc.view((-1, 3))
    reciprocal_cell = torch.linalg.inv(cell).transpose(2, 1)
    # print('reciprocal_cell: ', reciprocal_cell.device)
    inv_distances = reciprocal_cell.norm(2, dim=-1)
    # print('inv_distances: ', inv_distances.device)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats_ = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))
    # print('num_repeats_: ', num_repeats_.device)
    images, batch_images, shifts_expanded, shifts_idx_ = [], [], [], []
    for i_structure in range(num_repeats_.shape[0]):
        num_repeats = num_repeats_[i_structure]
        r1 = torch.arange(
            -num_repeats[0],
            num_repeats[0] + 1,
            device=cell.device,
            dtype=torch.long,
        )
        r2 = torch.arange(
            -num_repeats[1],
            num_repeats[1] + 1,
            device=cell.device,
            dtype=torch.long,
        )
        r3 = torch.arange(
            -num_repeats[2],
            num_repeats[2] + 1,
            device=cell.device,
            dtype=torch.long,
        )
        shifts_idx = torch.cartesian_prod(r1, r2, r3)
        shifts = torch.matmul(shifts_idx.to(cell.dtype), cell[i_structure])
        pos = positions[batch == i_structure]
        shift_expanded = shifts.repeat(1, n_atoms[i_structure]).view((-1, 3))
        pos_expanded = pos.repeat(shifts.shape[0], 1)
        images.append(pos_expanded + shift_expanded)

        batch_images.append(
            i_structure
            * torch.ones(
                images[-1].shape[0], dtype=torch.int64, device=cell.device
            )
        )
        shifts_expanded.append(shift_expanded)
        shifts_idx_.append(
            shifts_idx.repeat(1, n_atoms[i_structure]).view((-1, 3))
        )
    return (
        torch.cat(images, dim=0).to(positions.dtype),
        torch.cat(batch_images, dim=0),
        torch.cat(shifts_expanded, dim=0).to(positions.dtype),
        torch.cat(shifts_idx_, dim=0),
    )

# @torch.jit.script
def get_j_idx(
    edge_index: torch.Tensor, batch_images: torch.Tensor, n_atoms: torch.Tensor
) -> torch.Tensor:
    """TODO: add docs"""
    # get neighbor index reffering to the list of original positions
    n_neighbors = torch.bincount(edge_index[0])
    strides = strides_of(n_atoms)
    n_reapeats = torch.zeros_like(n_atoms)
    for i_structure, (st, nd) in enumerate(zip(strides[:-1], strides[1:])):
        n_reapeats[i_structure] = torch.sum(n_neighbors[st:nd])
    n_atoms = torch.repeat_interleave(n_atoms, n_reapeats, dim=0)

    batch_i = torch.repeat_interleave(strides[:-1], n_reapeats, dim=0)

    n_images = torch.bincount(batch_images)
    strides_images = strides_of(n_images[:-1])
    images_shift = torch.repeat_interleave(strides_images, n_reapeats, dim=0)

    j_idx = torch.remainder(edge_index[1] - images_shift, n_atoms) + batch_i
    return j_idx
