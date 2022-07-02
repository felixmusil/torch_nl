import torch

def wrap_positions(positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Wrap positions to unit cell.

    Returns positions changed by a multiple of the unit cell vectors to
    fit inside the space spanned by these vectors.

    Parameters
    ----------
    data:
        torch_geometric.Data instance
    eps: float
        Small number to prevent slightly negative coordinates from being
        wrapped.
    """

    center = torch.tensor((0.5, 0.5, 0.5)).view(1, 3)
    assert (
        data.n_atoms.shape[0] == 1
    ), f"There should be only one structure, found: {data.n_atoms.shape[0]}"

    pbc = data.pbc.view(1, 3)
    shift = center - 0.5 - eps

    # Don't change coordinates when pbc is False
    shift[torch.logical_not(pbc)] = 0.0

    cell = data.cell
    positions = data.pos

    fractional = torch.linalg.solve(cell.t(), positions.t()).t() - shift

    for i, periodic in enumerate(pbc.view(-1)):
        if periodic:
            fractional[:, i] = torch.remainder(fractional[:, i], 1.0)
            fractional[:, i] += shift[0, i]

    pos = torch.matmul(fractional, cell)
    return pos