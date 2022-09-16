from typing import Tuple
import torch

from .utils import get_number_of_cell_repeats, get_cell_shift_idx, strides_of
from .geometry import compute_cell_shifts


@torch.jit.script
def ravel_3d(idx_3d: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Convert 3d indices meant for an array of sizes `shape` into linear
    indices.

    Parameters
    ----------
    idx_3d : [-1, 3]
        _description_
    shape : [3]
        _description_

    Returns
    -------
    torch.Tensor
        linear indices
    """
    idx_linear = idx_3d[:, 2] + shape[2] * (
        idx_3d[:, 1] + shape[1] * idx_3d[:, 0]
    )
    return idx_linear


@torch.jit.script
def unravel_3d(idx_linear: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    """Convert linear indices meant for an array of sizes `shape` into 3d indices.

    Parameters
    ----------
    idx_linear : torch.Tensor [-1]

    shape : torch.Tensor [3]


    Returns
    -------
    torch.Tensor [-1, 3]

    """
    idx_3d = idx_linear.new_empty((idx_linear.shape[0], 3))
    idx_3d[:, 2] = torch.remainder(idx_linear, shape[2])
    idx_3d[:, 1] = torch.remainder(
        torch.div(idx_linear, shape[2], rounding_mode="floor"), shape[1]
    )
    idx_3d[:, 0] = torch.div(
        idx_linear, shape[1] * shape[2], rounding_mode="floor"
    )
    return idx_3d


@torch.jit.script
def get_linear_bin_idx(
    cell: torch.Tensor, pos: torch.Tensor, nbins_s: torch.Tensor
) -> torch.Tensor:
    """Find the linear bin index of each input pos given a box defined by its cell vectors and a number of bins, contained in the box, for each directions of the box.

    Parameters
    ----------
    cell : torch.Tensor [3, 3]
        cell vectors
    pos : torch.Tensor [-1, 3]
        set of positions
    nbins_s : torch.Tensor [3]
        number of bins in each directions

    Returns
    -------
    torch.Tensor
        linear bin index
    """
    scaled_pos = torch.linalg.solve(cell.t(), pos.t()).t()
    bin_index_s = torch.floor(scaled_pos * nbins_s).to(torch.long)
    bin_index_l = ravel_3d(bin_index_s, nbins_s)
    return bin_index_l


@torch.jit.script
def linked_cell(
    pos: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
    num_repeats: torch.Tensor,
    self_interaction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Determine the atomic neighborhood of the atoms of a given structure for a particular cutoff using the linked cell algorithm.

    Parameters
    ----------
    pos : torch.Tensor [n_atom, 3]
        atomic positions in the unit cell (positions outside the cell boundaries will result in an undifined behaviour)
    cell : torch.Tensor [3, 3]
        unit cell vectors in the format V=[v_0, v_1, v_2]
    cutoff : float
        length used to determine neighborhood
    num_repeats : torch.Tensor [3]
        number of unit cell repetitions in each directions required to account for PBC
    self_interaction : bool, optional
        to keep the original atoms as their own neighbor, by default False

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        neigh_atom : [2, n_neighbors]
            indices of the original atoms (neigh_atom[0]) with their neighbor index (neigh_atom[1]). The indices are meant to access the provided position array
        neigh_shift_idx : [n_neighbors, 3]
            cell shift indices to be used in reconstructing the neighbor atom positions.
    """
    device = pos.device
    dtype = pos.dtype
    n_atom = pos.shape[0]
    # find all the integer shifts of the unit cell given the cutoff and periodicity
    shifts_idx = get_cell_shift_idx(num_repeats, dtype)
    n_cell_image = shifts_idx.shape[0]
    shifts_idx = torch.repeat_interleave(shifts_idx, n_atom, dim=0)
    batch_image = torch.zeros((shifts_idx.shape[0]), dtype=torch.long)
    cell_shifts = compute_cell_shifts(
        cell.view(-1, 3, 3), shifts_idx, batch_image
    )

    i_ids = torch.arange(n_atom, device=device, dtype=torch.long)
    i_ids = i_ids.repeat(n_cell_image)
    # compute the positions of the replicated unit cell (including the original)
    # they are organized such that: 1st n_atom are the non-shifted atom, 2nd n_atom are moved by the same translation, ...
    images = pos[i_ids] + cell_shifts
    # create a rectangular box at [0,0,0] that encompases all the atoms (hence shifting the atoms so that they lie inside the box)
    b_min = images.min(dim=0).values
    b_max = images.max(dim=0).values
    images -= b_min - 1e-5
    i_pos = pos - b_min + 1e-5
    box_length = b_max - b_min + 1e-3
    # divide the box into square bins of size cutoff in 3d
    nbins_s = torch.maximum(torch.ceil(box_length / cutoff), pos.new_ones(3))
    # adapt the box lenghts so that it encompasses
    box_vec = torch.diag_embed(nbins_s * cutoff)
    nbins_s = nbins_s.to(torch.long)
    nbins = torch.prod(nbins_s)
    # determine which bins the original atoms and the images belong to following a linear indexing of the 3d bins
    bin_index_i = get_linear_bin_idx(box_vec, i_pos, nbins_s)
    bin_index_j = get_linear_bin_idx(box_vec, images, nbins_s)
    # reorder the atoms acording to their bins id
    _, atom_i = torch.sort(bin_index_i)
    sb_idx_j, atom_j = torch.sort(bin_index_j)
    n_atom_i_per_bin = torch.bincount(bin_index_i, minlength=nbins)
    n_atom_j_per_bin = torch.bincount(sb_idx_j, minlength=nbins)
    s_atom_j_bin_stride = strides_of(n_atom_j_per_bin)
    s_atom_i_bin_stride = strides_of(n_atom_i_per_bin)
    # find which bins the original atoms belong to
    i_bins_l = torch.unique(bin_index_i)
    i_bins_s = unravel_3d(i_bins_l, nbins_s)
    # find their neighbors. Since they have a side length of cutoff only 27 bins are in the neighborhood
    dd = torch.tensor([0, 1, -1], dtype=torch.long)
    bin_shifts = torch.cartesian_prod(dd, dd, dd).repeat((i_bins_s.shape[0], 1))
    neigh_bins_s = torch.repeat_interleave(i_bins_s, 27, dim=0) + bin_shifts
    neigh_bins_l = ravel_3d(neigh_bins_s, nbins_s)
    # remove the bins that are outside of the search range, i.e. beyond the borders of the box in the case of non-periodic directions
    mask = torch.logical_and(neigh_bins_l >= 0, neigh_bins_l < nbins)
    neigh_i_bins_l = torch.repeat_interleave(i_bins_l, 27, dim=0)[mask]
    neigh_j_bins_l = neigh_bins_l[mask]
    neigh_bins_l = torch.cat(
        [neigh_i_bins_l.view(1, -1), neigh_j_bins_l.view(1, -1)], dim=0
    )
    # linear list of bin indices containing original atoms and neighbor atoms
    neigh_bins_l = torch.unique(neigh_bins_l, dim=1)
    neigh_atom = []
    for ii in range(neigh_bins_l.shape[1]):
        bin_ids = neigh_bins_l[:, ii]
        st, nd = (
            s_atom_i_bin_stride[bin_ids[0]],
            s_atom_i_bin_stride[bin_ids[0] + 1],
        )
        # index of the central atoms
        i_atoms = atom_i[st:nd]

        st, nd = (
            s_atom_j_bin_stride[bin_ids[1]],
            s_atom_j_bin_stride[bin_ids[1] + 1],
        )
        # index of the neighbor atoms
        j_atoms = atom_j[st:nd]
        neigh_atom.append(torch.cartesian_prod(i_atoms, j_atoms).t())

    neigh_atom = torch.cat(neigh_atom, dim=1)
    if not self_interaction:
        # neighbor atoms are still indexed from 0 to n_atom*n_cell_image
        neigh_atom = neigh_atom[:, neigh_atom[0] != neigh_atom[1]]
    # sort neighbor list so that the i_atom indices increase
    sorted_ids = torch.argsort(neigh_atom[0])
    neigh_atom = neigh_atom[:, sorted_ids]
    # get the cell shift indices for each neighbor atom
    neigh_shift_idx = shifts_idx[neigh_atom[1]]
    # make sure the j_atom indices access the original positions
    neigh_atom[1] = torch.remainder(neigh_atom[1], n_atom)

    return neigh_atom, neigh_shift_idx


@torch.jit.script
def build_linked_cell_neighborhood(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    n_atoms: torch.Tensor,
    self_interaction: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the neighborlist of a given set of atomic structures using the linked cell algorithm.

    Parameters
    ----------
    positions : torch.Tensor [-1, 3]
        set of atomic positions for each structures
    cell : torch.Tensor [3*n_structure, 3]
        set of unit cell vectors for each structures
    pbc : torch.Tensor [n_structures, 3] bool
        periodic boundary conditions to apply
    cutoff : float
        length used to determine neighborhood
    n_atoms : torch.Tensor
        number of atoms in each structures
    self_interaction : bool
        to keep the original atoms as their own neighbor

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        mapping : [2, n_neighbors]
            indices of the neighbor list for the given positions array, mapping[0/1] correspond respectively to the central/neighbor atom (or node in the graph terminology)
        batch_mapping : [n_neighbors]
            indices mapping the neighbor atom to each structures
        cell_shifts_idx : [n_neighbors, 3]
            cell shift indices to be used in reconstructing the neighbor atom positions.
    """

    n_structure = n_atoms.shape[0]
    device = positions.device
    cell = cell.view((-1, 3, 3))
    pbc = pbc.view((-1, 3))
    # compute the number of cell replica necessary so that all the unit cell's atom have a complete neighborhood (no MIC assumed here)
    num_repeats = get_number_of_cell_repeats(cutoff, cell, pbc)

    stride = strides_of(n_atoms)
    ids = torch.arange(positions.shape[0], device=device, dtype=torch.long)
    print(num_repeats, pbc)
    mapping, batch_mapping, cell_shifts_idx = [], [], []
    for i_structure in range(n_structure):
        # select the atoms of structure i
        i_ids = ids[stride[i_structure] : stride[i_structure + 1]]
        # compute the neighborhood with the linked cell algorithm
        neigh_atom, neigh_shift_idx = linked_cell(
            positions[i_ids],
            cell[i_structure],
            cutoff,
            num_repeats[i_structure],
            self_interaction,
        )

        batch_mapping.append(
            i_structure
            * torch.ones(neigh_atom.shape[1], dtype=torch.long, device=device)
        )
        # shift the mapping indices so that they can access positions
        mapping.append(neigh_atom + stride[i_structure])
        cell_shifts_idx.append(neigh_shift_idx)
    # print(mapping)
    return (
        torch.cat(mapping, dim=1),
        torch.cat(batch_mapping, dim=0),
        torch.cat(cell_shifts_idx, dim=0),
    )
