# torch_nl

Provide a pytorch implementation of a naive (`compute_neighborlist_n2`) and a linked cell (`compute_neighborlist`) neighbor list that are compatible with TorchScript.

Their correctness is tested against ASE's implementation.

Note that contrary to ASE, the atoms positions are assumed to be wrapped inside the unit cell.