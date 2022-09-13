import torch
from typing import Tuple

def get_fully_connected_mapping(i_ids, shifts_idx, self_interaction) -> torch.Tensor:
    n_atom = i_ids.shape[0]
    n_atom2 = n_atom * n_atom
    n_cell_image = shifts_idx.shape[0]
    j_ids = torch.repeat_interleave(i_ids, n_cell_image)
    mapping = torch.cartesian_prod(i_ids, j_ids)
    shifts_idx = shifts_idx.repeat((n_atom2, 1))
    if not self_interaction:
        mask = torch.ones(mapping.shape[0], dtype=bool, device=i_ids.device)
        ids = n_cell_image*torch.arange(n_atom, device=i_ids.device) \
                    + torch.arange(0, mapping.shape[0], n_atom*n_cell_image, device=i_ids.device)
        mask[ids] = False
        mapping = mapping[mask, :]
        shifts_idx = shifts_idx[mask]
    return mapping, shifts_idx

def compute_images(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction:bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """TODO: add doc"""
    device = positions.device
    dtype = positions.dtype

    cell = cell.view((-1, 3, 3))
    pbc = pbc.view((-1, 3))

    has_pbc = pbc.prod(dim=1, dtype=bool)
    reciprocal_cell = torch.zeros_like(cell)
    reciprocal_cell[has_pbc,:,:] = torch.linalg.inv(cell[has_pbc,:,:]).transpose(2, 1)
    inv_distances = reciprocal_cell.norm(2, dim=-1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    num_repeats_ = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))

    stride = torch.zeros(n_atoms.shape[0]+1,dtype=torch.long)
    stride[1:] = torch.cumsum(n_atoms, dim=0,dtype=torch.long)
    ids = torch.arange(positions.shape[0], device=device, dtype=torch.long)

    mapping, batch_mapping, shifts_idx_ = [], [], []
    for i_structure in range(n_atoms.shape[0]):
        num_repeats = num_repeats_[i_structure]
        reps = []
        for ii in range(3):
            r1 = torch.arange(
                -num_repeats[ii],
                num_repeats[ii] + 1,
                device=device,
                dtype=dtype,
            )
            _, indices = torch.sort(torch.abs(r1))
            reps.append(r1[indices])
        shifts_idx = torch.cartesian_prod(*reps)
        i_ids = ids[stride[i_structure]:stride[i_structure+1]]

        s_mapping, shifts_idx = get_fully_connected_mapping(i_ids, shifts_idx, self_interaction)
        mapping.append(s_mapping)
        batch_mapping.append(
            i_structure
            * torch.ones(
                s_mapping.shape[0], dtype=torch.long, device=device
            )
        )
        shifts_idx_.append(shifts_idx)
    return (
        torch.cat(mapping, dim=0).t(),
        torch.cat(batch_mapping, dim=0),
        torch.cat(shifts_idx_, dim=0),
    )


