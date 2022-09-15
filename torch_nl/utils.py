import torch
import numpy as np


def ase2data(frames):
    n_atoms = [0]
    pos = []
    cell = []
    pbc = []
    for ff in frames:
        n_atoms.append(len(ff))
        pos.append(torch.from_numpy(ff.get_positions()))
        cell.append(torch.from_numpy(ff.get_cell().array))
        pbc.append(torch.from_numpy(ff.get_pbc()))
    pos = torch.cat(pos)
    cell = torch.cat(cell)
    pbc = torch.cat(pbc)
    stride = torch.from_numpy(np.cumsum(n_atoms))
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    for ii, (st, nd) in enumerate(zip(stride[:-1], stride[1:])):
        batch[st:nd] = ii
    n_atoms = torch.Tensor(n_atoms[1:]).to(dtype=torch.long)
    return pos, cell, pbc, batch, n_atoms


def strides_of(v: torch.Tensor) -> torch.Tensor:
    v = v.flatten()
    stride = v.new_empty(v.shape[0] + 1)
    stride[0] = 0
    torch.cumsum(v, dim=0, dtype=stride.dtype, out=stride[1:])
    return stride


def get_number_of_cell_repeats(cutoff, cell, pbc):
    cell = cell.view((-1, 3, 3)).to(torch.float64)
    pbc = pbc.view((-1, 3))

    has_pbc = pbc.prod(dim=1, dtype=bool)
    reciprocal_cell = torch.zeros_like(cell)
    reciprocal_cell[has_pbc, :, :] = torch.linalg.inv(
        cell[has_pbc, :, :]
    ).transpose(2, 1)
    print(reciprocal_cell)
    inv_distances = reciprocal_cell.norm(2, dim=-1)
    print(inv_distances, cutoff, inv_distances* cutoff)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    # num_repeats[0] += 1
    print(num_repeats)
    num_repeats_ = torch.where(pbc, num_repeats, torch.zeros_like(num_repeats))
    return num_repeats_


def get_cell_shift_idx(num_repeats, device, dtype):
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
    return shifts_idx
