import pytest
from ase.build import bulk, molecule
import numpy as np
from ase.neighborlist import neighbor_list
from ase import Atoms
import torch

from .neighbor_list import (
    compute_neighborlist_n2,
    compute_cell_shifts,
    compute_neighborlist,
)
from .utils import ase2data
from .geometry import compute_distances

# triclinic atomic structure
CaCrP2O7_mvc_11955_symmetrized = {
    "positions": [
        [3.68954016, 5.03568186, 4.64369552],
        [5.12301681, 2.13482791, 2.66220405],
        [1.99411973, 0.94691001, 1.25068234],
        [6.81843724, 6.22359976, 6.05521724],
        [2.63005662, 4.16863452, 0.86090529],
        [6.18250036, 3.00187525, 6.44499428],
        [2.11497733, 1.98032773, 4.53610884],
        [6.69757964, 5.19018203, 2.76979073],
        [1.39215545, 2.94386142, 5.60917746],
        [7.42040152, 4.22664834, 1.69672212],
        [2.43224207, 5.4571615, 6.70305327],
        [6.3803149, 1.71334827, 0.6028463],
        [1.11265639, 1.50166318, 3.48760997],
        [7.69990058, 5.66884659, 3.8182896],
        [3.56971588, 5.20836551, 1.43673437],
        [5.2428411, 1.96214426, 5.8691652],
        [3.12282634, 2.72812741, 1.05450432],
        [5.68973063, 4.44238236, 6.25139525],
        [3.24868468, 2.83997522, 3.99842386],
        [5.56387229, 4.33053455, 3.30747571],
        [2.60835346, 0.74421609, 5.3236629],
        [6.20420351, 6.42629368, 1.98223667],
    ],
    "cell": [
        [6.19330899, 0.0, 0.0],
        [2.4074486111396207, 6.149627748674982, 0.0],
        [0.2117993724186579, 1.0208820183960539, 7.305899571570074],
    ],
    "numbers": [
        20,
        20,
        24,
        24,
        15,
        15,
        15,
        15,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
    ],
    "pbc": [True, True, True],
}


def bulk_metal():
    frames = [
        bulk("Si", "diamond", a=6, cubic=True),
        bulk("Si", "diamond", a=6),
        bulk("Cu", "fcc", a=3.6),
        bulk("Si", "bct", a=6, c=3),
        # test very skewed unit cell
        bulk("Bi", "rhombohedral", a=6, alpha=20),
        bulk("Bi", "rhombohedral", a=6, alpha=10),
        bulk("Bi", "rhombohedral", a=6, alpha=5),
        bulk("SiCu", "rocksalt", a=6),
        bulk("SiFCu", "fluorite", a=6),
        Atoms(**CaCrP2O7_mvc_11955_symmetrized),
    ]
    return frames


def atomic_structures():
    frames = (
        [
            molecule("CH3CH2NH2"),
            molecule("H2O"),
            molecule("methylenecyclopropane"),
        ]
        + bulk_metal()
        + [
            molecule("OCHCHO"),
            molecule("C3H9C"),
        ]
    )
    return frames


@pytest.mark.parametrize(
    "frames, cutoff, self_interaction",
    [
        (atomic_structures(), rc, self_interaction)
        for rc in [1, 3, 5, 7]
        for self_interaction in [True, False]
    ],
)
def test_neighborlist_n2(frames, cutoff, self_interaction):
    """Check that torch_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors."""
    pos, cell, pbc, batch, n_atoms = ase2data(frames)

    dds = []
    mapping, batch_mapping, shifts_idx = compute_neighborlist_n2(
        cutoff, pos, cell, pbc, batch, self_interaction
    )
    cell_shifts = compute_cell_shifts(cell, shifts_idx, batch_mapping)
    dds = compute_distances(pos, mapping, cell_shifts)
    dds = np.sort(dds.numpy())

    dd_ref = []
    for frame in frames:
        idx_i, idx_j, idx_S, dist = neighbor_list(
            "ijSd", frame, cutoff=cutoff, self_interaction=self_interaction
        )
        dd_ref.extend(dist)
    dd_ref = np.sort(dd_ref)

    np.testing.assert_allclose(dd_ref, dds)


@pytest.mark.parametrize(
    "frames, cutoff, self_interaction",
    [
        (atomic_structures(), rc, self_interaction)
        # for rc in [3] #[1, 3, 5, 7]
        # for self_interaction in [False]
        for rc in [1, 3, 5, 7]
        for self_interaction in [False, True]
    ],
)
def test_neighborlist_linked_cell(frames, cutoff, self_interaction):
    """Check that torch_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors."""
    pos, cell, pbc, batch, n_atoms = ase2data(frames)

    dds = []
    mapping, batch_mapping, shifts_idx = compute_neighborlist(
        cutoff, pos, cell, pbc, batch, self_interaction
    )
    cell_shifts = compute_cell_shifts(cell, shifts_idx, batch_mapping)
    dds = compute_distances(pos, mapping, cell_shifts)
    dds = np.sort(dds.numpy())

    dd_ref = []
    for frame in frames:
        idx_i, idx_j, idx_S, dist = neighbor_list(
            "ijSd", frame, cutoff=cutoff, self_interaction=self_interaction
        )
        dd_ref.extend(dist)
    # nice for understanding if something goes wrong
    idx_S = torch.from_numpy(idx_S).to(torch.float64)

    print("idx_i", idx_i)
    print("idx_j", idx_j)
    missing_entries = []
    for ineigh in range(idx_i.shape[0]):
        mask = torch.logical_and(
            idx_i[ineigh] == mapping[0], idx_j[ineigh] == mapping[1]
        )

        if torch.any(torch.all(idx_S[ineigh] == shifts_idx[mask], dim=1)):
            pass
        else:
            missing_entries.append(
                (idx_i[ineigh], idx_j[ineigh], idx_S[ineigh])
            )
            print(missing_entries[-1])
            print(
                compute_cell_shifts(
                    cell,
                    idx_S[ineigh].view((1, -1)),
                    torch.tensor([0], dtype=torch.long),
                )
            )

    dd_ref = np.sort(dd_ref)
    print(dd_ref[-20:])
    print(dds[-20:])
    np.testing.assert_allclose(dd_ref, dds)
