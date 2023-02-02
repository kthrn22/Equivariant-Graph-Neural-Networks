import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, tensor

node_feat = torch.from_numpy(np.random.rand(5, 10)).to(torch.float32)
node_pos = torch.from_numpy(np.random.rand(5, 3)).to(torch.float32)
edge_index = np.array([[0, 1], [1, 0],
                        [0, 2], [2, 0],
                        [0, 3], [3, 0],
                        [0, 4], [4, 0],
                        [1, 2], [2, 1],
                        [1, 4], [4, 1],
                        [2, 3], [3, 2],
                        [2, 4], [4, 2], 
                        [3, 4], [4, 3],]).T
edge_index = torch.from_numpy(edge_index).to(torch.long)

atomic_num = torch.randint(0, 95, (5,))

class EmbeddingBlock(nn.Module):
    def __init__(self, embedding_dim: int):
        super(EmbeddingBlock, self).__init__()
        self.atomic_num_embedding = nn.Embedding(95, embedding_dim, padding_idx = 0)

    def forward(self, atomic_num) -> Tensor:
        r"""
        Initialize scalar representations

        Parameters:
            atomic_num (torch.tensor):
                Shape [N]

        Returns:
            scalar_feat (torch.Tensor):
                Shape [N, embedding_dim]
        """
        scalar_feat = self.atomic_num_embedding(atomic_num)
        return scalar_feat

class ScaledSiLU(nn.Module):
    def __init__(self):
        super(ScaledSiLU, self).__init__()
        self.scale_factor = 1 / 0.6
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x) * self.scale_factor

class RadialBasisFunction(nn.Module):
    def __init__(self, num_radial_basis: int, cut_off: float, trainable: bool = False):
        super(RadialBasisFunction, self).__init__()
        self.num_radial_basis = num_radial_basis
        self.cut_off = cut_off

        expanded_distance = nn.Parameter(torch.Tensor(num_radial_basis))
        with torch.no_grad():
            torch.arange(1, num_radial_basis + 1, out = expanded_distance).mul_(torch.pi)
        
        if trainable:
            expanded_distance.requires_grad_()
        else:
            self.register_buffer("expanded_distance", expanded_distance)

        self.expanded_distance = expanded_distance

    def forward(self, distance: Tensor) -> Tensor:
        r"""
        Construct radial basis for interatomic distance

        Parameters:
            distance (torch.Tensor):
                Interatomic distance. Shape [E]

        Returns:
            expanded_distance (torch.Tensor):
                Shape [E, num_radial_basis]
        """
        distance_scaled = distance / self.cut_off

        # shape [E, num_radial_basis]
        expanded_distance = torch.sin(self.expanded_distance * distance_scaled.unsqueeze(-1)) / distance.unsqueeze(-1)

        return expanded_distance

class CosineCutoff(nn.Module):
    def __init__(self, cut_off: float):
        super(CosineCutoff, self).__init__()
        self.cut_off = cut_off

    def forward(self, x):
        return 0.5 * (1 + torch.cos(x * torch.pi / self.cut_off)) * (x < self.cut_off).to(x.dtype)

class MessageBlock(nn.Module):
    def __init__(self, embedding_dim, num_radial_basis, cut_off):
        super(MessageBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.scalar_feat_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            ScaledSiLU(),
            nn.Linear(embedding_dim, embedding_dim * 3)   
        )
        self.radial_basis = RadialBasisFunction(num_radial_basis, cut_off)
        self.rbf_proj = nn.Linear(num_radial_basis, embedding_dim * 3)
        self.cosine_cut_off = CosineCutoff(cut_off)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scalar_feat_proj[0].weight)
        self.scalar_feat_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.scalar_feat_proj[2].weight)
        self.scalar_feat_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)        

    def forward(self, vectorial_feat: Tensor, scalar_feat: Tensor, node_pos: Tensor, edge_index: Tensor):
        r"""
        Parameters:
            vectorial_feat (torch.Tensor):
                Vectorial representations. Shape [N, embedding_dim, 3]
            scalar_feat (torch.Tensor):
                Scalar representations. Shape [N, embedding_dim]
            edge_index (torch.Tensor):
                Shape [2, E]
            node_pos (torch.Tensor):
                Atom's 3D coordinates. Shape [N, 3]

        Returns:
            vectorial_message (torch.Tensor):
                Shape [N, embedding_dim, 3]
            scalar_message (torch.Tensor):
                Shape [N, embedding]
        """
        num_nodes, num_edges = node_pos.shape[0], edge_index.shape[-1]
        # shape [N, embedding_dim] -> [N, embedding_dim * 3]
        scalar_feat = self.scalar_feat_proj(scalar_feat)

        source, target = edge_index
        # r_ij = r_i - r_j. Shape [E, 3]
        relative_distance = node_pos[target] - node_pos[source]
        # Shape [E]
        distance = torch.sum(relative_distance ** 2, dim = -1) ** 0.5
        # Shape [E, num_radial_basis]
        expanded_distance = self.radial_basis(distance)
        # Shape [E, num_radial_basis] -> [E, embedding_dim * 3]
        filter = self.rbf_proj(expanded_distance)
        filter = self.cosine_cut_off(filter)

        # Shape [E, embedding_dim * 3]
        message = scalar_feat[source] * filter
        # Shape [E, embedding_dim * 3] -> [E, embedding_dim, 3]
        message = message.view(-1, self.embedding_dim, 3)#.permute(2, 0, 1)
        # Shape [E, embedding_dim, 1]
        scalar_message, equivariant_vectorial_message, invariant_vectorial_message = torch.split(message, [1, 1, 1], dim = -1)
        
        # Shape [E, embedding_dim]
        scalar_message = scalar_message.squeeze(-1)
        equivariant_vectorial_message = equivariant_vectorial_message.squeeze(-1)
        invariant_vectorial_message = invariant_vectorial_message.squeeze(-1)

        # aggregrate message
        target_index_lifted = torch.broadcast_to(target.unsqueeze(-1), (num_edges, self.embedding_dim))
        # shape [N, embedding_dim]
        scalar_message = torch.zeros(num_nodes, self.embedding_dim).scatter_add_(0, target_index_lifted, scalar_message)

        # shape [E, 3] -> [E, embedding_dim, 3]
        relative_distance = torch.broadcast_to((relative_distance / distance.unsqueeze(-1)).unsqueeze(1), (num_edges, self.embedding_dim, 3))
        # shape [E, embedding_dim] -> [E, embedding_dim, 3]
        invariant_vectorial_message = invariant_vectorial_message.unsqueeze(-1) * relative_distance
        equivariant_vectorial_message = equivariant_vectorial_message.unsqueeze(-1) * vectorial_feat[source]
        vectorial_message = invariant_vectorial_message + equivariant_vectorial_message
        
        target_index_lifted = torch.broadcast_to(target_index_lifted.unsqueeze(-1), (num_edges, self.embedding_dim, 3))
        # shape [N, embedding_dim, 3]
        vectorial_message = torch.zeros(num_nodes, self.embedding_dim, 3).scatter_add_(0, target_index_lifted, vectorial_message)

        return vectorial_message, scalar_message

class UpdateBlock(nn.Module):
    def __init__(self, embedding_dim):
        super(UpdateBlock, self).__init__()

        self.scalar_feat_proj = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                ScaledSiLU(),
                nn.Linear(embedding_dim, 3 * embedding_dim)
        )

        self.vectorial_feat_proj = nn.Linear(embedding_dim, 2 * embedding_dim, bias = False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scalar_feat_proj[0].weight)
        self.scalar_feat_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.scalar_feat_proj[2].weight)
        self.scalar_feat_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vectorial_feat_proj.weight)

    def forward(self, vectorial_feat: Tensor, scalar_feat: Tensor):
        r"""
        Parameters:
            vectorial_feat (torch.Tensor):
                Vectorial representations. Shape [N, embedding_dim, 3]
            scalar_feat (torch.Tensor):
                Scalar representations. Shape [N, embedding_dim]

        Returns:
            vectorial_update (torch.Tensor):
                Shape [N, embedding_dim, 3]
            scalar_update (torch.Tensor):
                Shape [N, embedding_dim]
        """

        num_nodes = vectorial_feat.shape[0]
        # shape [N, embedding_dim, 3] -> [N, 2 * embedding_dim, 3]
        vectorial_feat = self.vectorial_feat_proj(vectorial_feat.permute(0, 2, 1)).permute(0, 2, 1)
        # shape [N, 2 * embedding_dim, 3] -> [N, embedding_dim, 2, 3] -> split [N, embedding_dim, 1, 3] [N, embedding_dim, 1, 3] 
        U, V = torch.split(vectorial_feat.view(num_nodes, -1, 2, 3), [1, 1], dim = 2)
        # shape [N, embedding_dim, 1, 3] -> [N, embedding_dim, 3]
        U, V = U.squeeze(2), V.squeeze(2)

        # shape [N, 2 * embedding_dim] -> [N, 3 * embedding_dim]
        a = self.scalar_feat_proj(torch.cat([scalar_feat, torch.sum(V, dim = -1)], dim = -1))
        
        # shape [N, 3 * embedding_dim] -> [N, embedding_dim, 3] -> split into 3 tensors [N, embedding_dim]
        a = a.view(num_nodes, -1, 3)
        a_vv, a_sv, a_ss = torch.split(a, [1, 1, 1], dim = -1)
        a_vv, a_sv, a_ss = a_vv.squeeze(-1), a_sv.squeeze(-1), a_ss.squeeze(-1)

        # [N, embedding_dim]
        scalar_product = torch.sum(U * V, dim = -1)
        # [N, embedding_dim]
        scalar_update = a_ss + a_sv * scalar_product

        # [N, embedding_dim, 3]
        vectorial_update = U * a_vv.unsqueeze(-1)

        return vectorial_update, scalar_update

class GatedEquivariantBlock(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super(GatedEquivariantBlock, self).__init__()
        self.vectorial_feat_proj_1 = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.vectorial_feat_proj_2 = nn.Linear(embedding_dim, out_dim, bias = False)

        self.scalar_feat_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            ScaledSiLU(),
            nn.Linear(embedding_dim, out_dim * 2)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vectorial_feat_proj_1.weight)
        nn.init.xavier_uniform_(self.vectorial_feat_proj_2.weight)
        nn.init.xavier_uniform_(self.scalar_feat_proj[0].weight)
        self.scalar_feat_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.scalar_feat_proj[2].weight)
        self.scalar_feat_proj[2].bias.data.fill_(0)

    def forward(self, vectorial_feat: Tensor, scalar_feat: Tensor):
        r"""
        Returns force vectors

        Parameters:
            vectorial_feat (torch.Tensor):
                Vectorial representations. Shape [N, embedding_dim, 3]
            scalar_feat (torch.Tensor):
                Scalar representations. Shape [N, embedding_dim]

        Returns:
            vectorial_feat (torch.Tensor):
                Shape [N, out_dim, 3]
            scalar_feat (torch.Tensor):
                Shape [N, out_dim]
        """
        num_nodes = vectorial_feat.shape[0]
        # shape [N, embedding_dim, 3] -> [N, embedding_dim, 3]
        vectorial_feat_1 = self.vectorial_feat_proj_1(vectorial_feat.permute(0, 2, 1)).permute(0, 2, 1)
        # shape [N, embedding_dim, 3] -> [N, out_dim, 3]
        vectorial_feat_2 = self.vectorial_feat_proj_2(vectorial_feat.permute(0, 2, 1)).permute(0, 2, 1)

        # shape [N, embedding_dim] -> [N, embedding_dim * 2] -> [N, out_dim * 2]
        scalar_feat = self.scalar_feat_proj(torch.cat([scalar_feat, torch.sum(vectorial_feat_1, dim = -1)], dim = -1))

        # shape [N, out_dim * 2] -> [N, out_dim, 2] -> split into 2 tensors [N, out_dim, 1], [N, out_dim, 1]
        scalar_feat = scalar_feat.view(num_nodes, -1, 2)
        scalar_feat, vectorial_feat_1 = torch.split(scalar_feat, [1, 1], dim = -1)
        # [N, out_dim, 1] -> [N, out_dim]
        scalar_feat, vectorial_feat_1 = scalar_feat.squeeze(-1), vectorial_feat_1.squeeze(-1)

        # shape [N, out_dim, 3]
        vectorial_feat = vectorial_feat_2 * vectorial_feat_1.unsqueeze(-1)

        return vectorial_feat, scalar_feat

class OutputBlock(nn.Module):
    def __init__(self, embedding_dim):
        super(OutputBlock, self).__init__()

        self.out = nn.ModuleList([
            GatedEquivariantBlock(embedding_dim, embedding_dim // 2),
            GatedEquivariantBlock(embedding_dim // 2, 1)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for block in self.out:
            block.reset_parameters()

    def forward(self, vectortial_feat: Tensor, scalar_feat: Tensor) -> Tensor:
        r"""
        Returns force vectors

        Parameters:
            vectorial_feat (torch.Tensor):
                Vectorial representations. Shape [N, embedding_dim, 3]
            scalar_feat (torch.Tensor):
                Scalar representations. Shape [N, embedding_dim]

        Returns:
            vectors (torch.Tensor):
                Force vectors. Shape [N, 3]
        """
        for gated_equivariant_block in self.out:
            vectortial_feat, scalar_feat = gated_equivariant_block(vectortial_feat, scalar_feat)
        
        # [N, 1, 3] -> [N, 3]
        vectors = vectortial_feat.squeeze(1)
        
        return vectors
        
class PaiNN(nn.Module):
    def __init__(self, embedding_dim: int, num_blocks: int, num_radial_basis: int, cut_off: float, out_dim: int,
            return_potential_energy: bool = True, return_force_vectors: bool = True):
        super(PaiNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding_block = EmbeddingBlock(embedding_dim)

        self.message_blocks = nn.ModuleList([
            MessageBlock(embedding_dim, num_radial_basis, cut_off) for _ in range(num_blocks)
        ])

        self.update_blocks = nn.ModuleList([
            UpdateBlock(embedding_dim) for _ in range(num_blocks)
        ])

        self.out_scalar_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            ScaledSiLU(),
            nn.Linear(embedding_dim // 2, out_dim)
        )

        if return_force_vectors: 
            self.out_vector = OutputBlock(embedding_dim)

        self.return_potential_energy = return_potential_energy
        self.return_force_vectors = return_force_vectors

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.out_scalar_proj[0].weight)
        self.out_scalar_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_scalar_proj[2].weight)
        self.out_scalar_proj[2].bias.data.fill_(0)

        if self.return_force_vectors:
            self.out_vector.reset_parameters()

        for message_block, update_block in zip(self.message_blocks, self.update_blocks):
            message_block.reset_parameters()
            update_block.reset_parameters()

    def forward(self, atomic_num: Tensor, node_pos: Tensor, edge_index: Tensor) -> Tensor:
        r"""
        Parameters:
            atomic_num (torch.Tensor):
                Atomic number of each atom in the molecular graph. Shape [N]
            node_pos (torch.Tensor):
                Atoms' 3D coordinates. Shape [N, 3]
            edge_index (torch.Tensor):
                Shape [2, E]
        """
        num_nodes = atomic_num.shape[0]
        # Initialize vectorial representations and scalar representations
        # Shape [N, embedding_dim, 3], [N, embedding_dim]
        vectorial_feat, scalar_feat = torch.zeros(num_nodes, self.embedding_dim, 3), self.embedding_block(atomic_num)
        
        _, scalar_message = self.message_blocks[0](vectorial_feat, scalar_feat, node_pos, edge_index)
        scalar_feat = scalar_feat + scalar_message
        vectorial_update, scalar_update = self.update_blocks[0](vectorial_feat, scalar_feat)

        for message_block, update_block in zip(self.message_blocks[1:], self.update_blocks[1:]):
            vectorial_message, scalar_message = message_block(vectorial_feat, scalar_feat, node_pos, edge_index)
            vectorial_feat = vectorial_feat + vectorial_message
            scalar_feat = scalar_feat + scalar_message

            vectorial_update, scalar_update = update_block(vectorial_feat, scalar_feat)
            vectorial_feat = vectorial_feat + vectorial_message
            scalar_feat = scalar_feat + scalar_message

        # [N, out_dim]
        scalar = self.out_scalar_proj(scalar_feat)

        if self.return_potential_energy:
            potential_energy = torch.sum(scalar.squeeze())

        if self.return_force_vectors:
            force_vectors = self.out_vector(vectorial_feat, scalar_feat)        

        if self.return_potential_energy and self.return_force_vectors:
            return potential_energy, force_vectors
            
        if self.return_potential_energy:
            return potential_energy
        
        if self.return_force_vectors:
            return force_vectors

        return scalar

p = PaiNN(10, 5, 20, 0.5, 1)

e, f = p.forward(atomic_num, node_pos, edge_index)