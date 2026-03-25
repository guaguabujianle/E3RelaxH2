# %%
import math
import copy
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from utils.graph_utils import AtomEmbedding, RadialBasis
from utils.graph_utils import vector_norm
from model.block import MessagePassing, SelfInteractionBlock, StructureUpdating, LatticeBlock

class E3RelaxH2(nn.Module):
    def __init__(
        self,
        hidden_channels=512,
        num_layers=3,
        num_rbf=128,
        cutoff=6.,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        num_elements=83,
        return_cell=True
    ):
        super(E3RelaxH2, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_config = rbf
        self.envelope_config = envelope
        self.num_elements = num_elements
        self.return_cell = return_cell

        #### Learnable parameters #############################################

        self.atom_emb = AtomEmbedding(hidden_channels, num_elements)

        self.lattice_scalar_emb = nn.Embedding(3, hidden_channels)   # Joint embedding for lattice basis vectors

        self.radial_basis = RadialBasis(
            num_radial=num_rbf,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        self.message_passing_layers = nn.ModuleList()
        self.message_update_layers = nn.ModuleList()
        self.structure_update_layers = nn.ModuleList()

        self.lattice_blocks = nn.ModuleList()
        self.lattice_feat_mlps = nn.ModuleList()

        for i in range(num_layers):
            self.message_passing_layers.append(MessagePassing(hidden_channels, num_rbf, max_ell=3)) # Layer for message passing between atoms with SH gating
            self.message_update_layers.append(SelfInteractionBlock(hidden_channels)) # Layer for self-updating 
            self.structure_update_layers.append(StructureUpdating(hidden_channels)) # Layer for updating atomic structure

            self.lattice_blocks.append(LatticeBlock(hidden_channels, hidden_channels, num_rbf))  # Joint lattice update block
            self.lattice_feat_mlps.append(
                nn.Sequential(
                    nn.Linear(num_rbf * 2, num_rbf),
                    nn.SiLU(),
                    nn.Linear(num_rbf, num_rbf),
                )
            )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, data):

        pos = getattr(data, 'pos_u', None)
        cell = getattr(data, 'cell_u', None)
        z = getattr(data, 'x', None)
        if pos is None:
            pos = data.pos
        if cell is None:
            cell = data.cell
        if z is None:
            z = data.atomic_numbers
        z = z.long()
        
        cell_offsets = data.cell_offsets.float()
        edge_index = data.edge_index

        neighbors = data.neighbors
        batch = data.batch

        atom_scalar = self.atom_emb(z)
        atom_vector = torch.zeros(atom_scalar.size(0), 3, atom_scalar.size(1), device=atom_scalar.device)

        num_graphs = batch[-1].item() + 1
 
        # Initialize learnable lattice states for the three basis vectors
        axis_ids = torch.arange(3, device=edge_index.device, dtype=torch.long)
        scalar_base = self.lattice_scalar_emb(axis_ids)  # [3, H]
        lattice_scalar_state = scalar_base.unsqueeze(0).expand(num_graphs, -1, -1).contiguous()

        lattice_vector_state = torch.zeros(
            num_graphs, 3, 3, atom_scalar.size(1), device=atom_scalar.device, dtype=atom_scalar.dtype
        )  # (graph, axis, vector-dim, hidden)
        
        #### Interaction blocks ###############################################
        pos_list = []
        cell_list = []

        for n in range(self.num_layers):
            # Atom coordinates and lattice parameters are updated after each graph convolution
            # Therefore, relative positions and interatomic distances must be recalculated
            j, i = edge_index
            abc_unsqueeze = cell.repeat_interleave(neighbors, dim=0) 
            pos_diff = pos[j] + torch.einsum("a p, a p v -> a v", cell_offsets, abc_unsqueeze) - pos[i]
            edge_dist = vector_norm(pos_diff, dim=-1)
            edge_udiff = -pos_diff / (edge_dist.unsqueeze(-1) + 1e-9)  # avoid divide-by-zero -> NaN
            edge_udiff = torch.nan_to_num(edge_udiff) # Safety against NaNs
            edge_feat = self.radial_basis(edge_dist)  

            pos_mean = global_mean_pool(pos, batch)
            pos_centered = pos - pos_mean[batch]                      # (N, 3)
            cell_batch = cell[batch]                                 # (N, 3, 3)
            cell_norm = vector_norm(cell_batch, dim=-1)              # (N, 3)
            pos_expand = pos_centered.unsqueeze(1).expand(-1, 3, -1) # (N, 3, 3)
            # parallel component
            d_para = (pos_expand * cell_batch).sum(dim=-1) / (cell_norm + 1e-9)   # (N, 3)
            # perpendicular component
            cross = torch.linalg.cross(pos_expand, cell_batch, dim=-1)             # (N, 3, 3)
            d_perp = vector_norm(cross, dim=-1) / (cell_norm + 1e-9)               # (N, 3)
            feat_para = self.radial_basis(d_para.reshape(-1)).view(pos.size(0), 3, -1)
            feat_perp = self.radial_basis(d_perp.reshape(-1)).view(pos.size(0), 3, -1)
            lattice_feat = torch.cat([feat_para, feat_perp], dim=-1)
            lattice_feat = self.lattice_feat_mlps[n](lattice_feat)

            # Keep directional vector message for lattice->atom interaction.
            lattice_diff = cell_batch - pos_expand                                          # (N, 3, 3)
            lattice_dist = vector_norm(lattice_diff, dim=-1)                               # (N, 3)
            lattice_udiff = -lattice_diff / (lattice_dist.unsqueeze(-1) + 1e-9)            # (N, 3, 3)
            lattice_udiff = torch.nan_to_num(lattice_udiff) # Safety against NaNs

            lattice_block = self.lattice_blocks[n]
            atom_scalar, lattice_scalar_state, atom_vector, lattice_vector_state = lattice_block.distribute_to_atoms(
                atom_scalar, lattice_scalar_state, atom_vector, lattice_vector_state, lattice_feat, lattice_udiff, batch
            )

            # Message passing
            delta_scalar, delta_vector = self.message_passing_layers[n](
                atom_scalar, atom_vector, edge_index, edge_feat, edge_udiff
            )
            atom_scalar = (atom_scalar + delta_scalar) * self.inv_sqrt_2
            atom_vector = atom_vector + delta_vector

            # Self-updating
            delta_scalar, delta_vector = self.message_update_layers[n](atom_scalar, atom_vector)
            atom_scalar = (atom_scalar + delta_scalar) * self.inv_sqrt_2
            atom_vector = atom_vector + delta_vector

            # Structure updating
            pos_delta = self.structure_update_layers[n](atom_vector)
            pos = pos + pos_delta

            # Lattice nodes collect messages from all atoms jointly
            lattice_scalar_state, lattice_vector_state = lattice_block.gather_from_atoms(
                atom_scalar, lattice_scalar_state, atom_vector, lattice_vector_state, batch
            )
            cell_delta = lattice_block.decode_lattice_delta(lattice_vector_state)
            cell = cell + cell_delta

            pos_list.append(pos)
            cell_list.append(cell)
        
        if self.return_cell:
            return pos_list, cell_list
        else:
            return pos_list

    # Avoid deepcopy failures (EMA) by recreating the module from config
    def __deepcopy__(self, memo):
        cls = self.__class__
        copied = cls(
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            num_rbf=self.num_rbf,
            cutoff=self.cutoff,
            rbf=copy.deepcopy(self.rbf_config, memo),
            envelope=copy.deepcopy(self.envelope_config, memo),
            num_elements=self.num_elements,
            return_cell=self.return_cell,  # <- keep this!
        )
        copied.load_state_dict(copy.deepcopy(self.state_dict(), memo))
        return copied


# %%
