from typing import Tuple
import torch
import numpy as np

from ase.neighborlist import neighbor_list
from ase import Atoms


def ase_neighbor_list(
    positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, rcut: float, self_interaction: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function for converting a list of neighbor edges from
    an input AtomicData instance, to an ASE neighborlist as
    output by ase.neighborlist.neighbor_list. Return documenation
    taken from https://wiki.fysik.dtu.dk/ase/_modules/ase/neighborlist.html#neighbor_list.

    Parameters
    ----------

    rcut:
        Upper distance cover for determining neighbors
    self_interaction:
        If True, self edges will be added.

    Returns
    -------
    torch.Tensor:
        first atom indices, of shape (n_atoms)
    torch.Tensor:
        second atom index, of shape (n_atoms)
    torch.Tensor:
        Dot product of the periodic shift vectors with the system unit cell vectors
    """

    frame = Atoms(
        positions=positions.numpy(),
        cell=cell.numpy(),
        pbc=pbc.numpy(),
        numbers=np.ones(positions.shape[0]),
    )

    idx_i, idx_j, idx_S, dist = neighbor_list(
        "ijSd", frame, cutoff=rcut, self_interaction=self_interaction
    )
    return (
        torch.from_numpy(idx_i),
        torch.from_numpy(idx_j),
        torch.from_numpy(np.dot(idx_S, frame.cell)),
        torch.from_numpy(dist),
    )
