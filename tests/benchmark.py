import torch
import ase
import sys
import numpy as np
import scipy
from ase.build import molecule, bulk, make_supercell
from ase.neighborlist import neighbor_list
import pandas as pd
from tqdm import tqdm
# import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0,'../')
from torch_nl import compute_neighborlist, compute_neighborlist_n2, ase2data
from torch_nl.timer import timeit
torch.set_num_threads(4)
cutoff = 4
tags = [
    # "torch_nl O(n^2) CPU",
    "torch_nl O(n^2) GPU",
    "torch_nl O(n) CPU",
    # "torch_nl O(n) GPU"
]

frame = bulk('Si', 'diamond', a=4, cubic=True)
aa = torch.arange(1, 6)
Ps = torch.cartesian_prod(aa,aa,aa)
Ps = Ps[torch.sort(Ps.sum(dim=1)).indices].to(torch.long).numpy()
frames = []
n_atoms = []
for P in Ps:
    frames.append(make_supercell(frame, np.diag(P)))
    n_atoms.append(len(frames[-1]))
n_atoms = np.array(n_atoms)
print("Starting")
tag = "ASE"
datas = []
for frame in tqdm(frames):
    timing = timeit(neighbor_list, ['ijS', frame, cutoff], tag=tag, warmup=1, nit=50)
    data = timing.dumps()
    i,j,S = neighbor_list('ijS', frame, cutoff)
    n_neighbor = np.bincount(i).mean()
    data.update(n_atom=len(frame), n_neighbor_per_atom_avg=int(n_neighbor))
    data.pop('samples')
    datas.append(data)


for tag in tqdm(tags):
    if "CPU" in tag:
        device = 'cpu'
    elif "GPU" in tag:
        device = 'cuda'

    if 'O(n^2)' in tag:
        nl_func = compute_neighborlist_n2
    elif 'O(n)' in tag:
        nl_func = compute_neighborlist

    for frame in tqdm(frames):
        pos, cell, pbc, batch, n_atoms = ase2data([frame], device=device)
        with torch.cuda.amp.autocast():
            timing = timeit(nl_func, [cutoff, pos, cell, pbc, batch], tag=tag, warmup=10, nit=50)
        data = timing.dumps()
        data.pop('samples')
        mapping, mapping_batch, shifts_idx = nl_func(cutoff, pos, cell, pbc, batch)
        n_neighbor = np.bincount(mapping[0].cpu().numpy()).mean()
        data.update(n_atom=len(frame), n_neighbor_per_atom_avg=int(n_neighbor))
        datas.append(data)

df = pd.DataFrame(datas)

# sns.lmplot(data=df, x='n_atom', y='mean', hue='tag',fit_reg=False)

# plt.savefig('./test_0.png', dpi=300, bbox_inches='tight')
# plt.show()
print("END")
# %%
