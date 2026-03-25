# %%
import numpy as np
import torch
from torch_geometric.data import Data

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    AseAtomsAdaptor = None


OFFSET_LIST = [
    [0, 0, 1], [0, 1, 0], [1, 0, 0],
    [0, 1, 1], [1, 0, 1], [1, 1, 0],
]


class AtomsToGraphs:
    """Convert periodic atomic structures to graphs."""

    def __init__(self, radius=6, max_neigh=50, use_offset_list: bool = False):
        """
        use_offset_list:
          - False: return raw neighbor_list edges (NO self-edge removal, NO self-image edges)
          - True:  remove accidental self-edges and add OFFSET_LIST self-image edges
        """
        if AseAtomsAdaptor is None:
            raise ImportError("pymatgen (AseAtomsAdaptor) is required for neighbor list.")

        self.radius = float(radius)
        self.max_neigh = int(max_neigh)
        self.use_offset_list = bool(use_offset_list)

    def _get_neighbors_pymatgen(self, atoms):
        """Performs nearest neighbor search and returns edge_index and cell_offsets."""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        # cap neighbors per center atom
        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            if idx_i.size == 0:
                continue
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])

        if len(_nonmax_idx) == 0:
            _c_index = np.array([], dtype=np.int64)
            _n_index = np.array([], dtype=np.int64)
            _offsets = np.empty((0, 3), dtype=np.int64)
        else:
            _nonmax_idx = np.concatenate(_nonmax_idx)
            _c_index = _c_index[_nonmax_idx]
            _n_index = _n_index[_nonmax_idx]
            _offsets = _offsets[_nonmax_idx]

        # Raw edges from neighbor list
        edge_index = torch.from_numpy(np.vstack((_n_index, _c_index))).long()  # (2, E)
        cell_offsets = torch.from_numpy(_offsets).long()                       # (E, 3)

        # Option: do NOT use OFFSET_LIST -> return raw
        if not self.use_offset_list:
            return edge_index, cell_offsets

        # Otherwise: remove accidental self-connecting edges
        if edge_index.numel() > 0:
            non_self_mask = edge_index[0, :] != edge_index[1, :]
            edge_index = edge_index[:, non_self_mask]
            cell_offsets = cell_offsets[non_self_mask]

        # Add self-image edges using OFFSET_LIST
        offsets_t = torch.tensor(OFFSET_LIST, dtype=torch.long)  # (S, 3)
        S = offsets_t.size(0)
        num_atoms = len(atoms)

        self_nodes = torch.arange(num_atoms, dtype=torch.long).repeat_interleave(S, dim=0)
        self_connect_edge_index = torch.stack([self_nodes, self_nodes], dim=0)  # (2, num_atoms*S)
        self_connect_cell_offsets = offsets_t.repeat(num_atoms, 1)              # (num_atoms*S, 3)

        edge_index = torch.cat([edge_index, self_connect_edge_index], dim=-1)
        cell_offsets = torch.cat([cell_offsets, self_connect_cell_offsets], dim=0)

        return edge_index, cell_offsets

    def convert_single(self, atoms_u):
        pos_u = torch.tensor(atoms_u.get_positions(), dtype=torch.float32)
        cell_u = torch.tensor(np.asarray(atoms_u.get_cell()), dtype=torch.float32)
        edge_index, cell_offsets = self._get_neighbors_pymatgen(atoms_u)

        atomic_numbers = torch.tensor(atoms_u.get_atomic_numbers(), dtype=torch.long)
        natoms = int(pos_u.shape[0])
        pbc = torch.tensor(np.asarray(atoms_u.pbc), dtype=torch.bool)

        return Data(
            cell_u=cell_u.view(1, 3, 3),
            pos_u=pos_u,
            x=atomic_numbers,
            cell_offsets=cell_offsets,
            edge_index=edge_index,
            natoms=natoms,
            pbc=pbc,
            neighbors=edge_index.size(-1),
        )

    def convert_pairs(self, atoms_u, atoms_r):
        pos_u = torch.tensor(atoms_u.get_positions(), dtype=torch.float32)
        cell_u = torch.tensor(np.asarray(atoms_u.get_cell()), dtype=torch.float32)
        edge_index, cell_offsets = self._get_neighbors_pymatgen(atoms_u)

        pos_r = torch.tensor(atoms_r.get_positions(), dtype=torch.float32)
        cell_r = torch.tensor(np.asarray(atoms_r.get_cell()), dtype=torch.float32)

        atomic_numbers = torch.tensor(atoms_u.get_atomic_numbers(), dtype=torch.long)
        natoms = int(pos_u.shape[0])
        pbc = torch.tensor(np.asarray(atoms_u.pbc), dtype=torch.bool)

        return Data(
            cell_u=cell_u.view(1, 3, 3),
            pos_u=pos_u,
            cell_r=cell_r.view(1, 3, 3),
            pos_r=pos_r,
            x=atomic_numbers,
            cell_offsets=cell_offsets,
            edge_index=edge_index,
            natoms=natoms,
            pbc=pbc,
        )

# %%
