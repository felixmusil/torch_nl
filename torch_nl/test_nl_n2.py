import pytest
import torch
from ase.build import bulk, molecule
import numpy as np
from ase.neighborlist import neighbor_list

from .neighbor_list import compute_nl_n2, compute_cell_shifts, compute_distances
from .utils import ase2data


def bulk_metal():
    frames = [
        bulk("Si", "diamond", a=6, cubic=True),
        bulk("Cu", "fcc", a=3.6),
        bulk("Si", "diamond", a=6),
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
        for rc in range(2, 6, 2)
        for self_interaction in [True, False]
    ],
)
def test_neighborlist(frames, cutoff, self_interaction):
    """Check that torch_neighbor_list gives the same NL as ASE by comparing
    the resulting sorted list of distances between neighbors."""
    pos, cell, pbc, batch, n_atoms = ase2data(frames)

    dds = []
    mapping, batch_mapping, shifts_idx = compute_nl_n2(
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
