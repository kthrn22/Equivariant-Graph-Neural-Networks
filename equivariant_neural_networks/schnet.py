import torch 
import torch.nn as nn
import numpy as np
from torch import Tensor


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()

    def forward(self, x):
        return nn.functional.softplus(x) - torch.log(torch.tensor(2)).to(x.dtype)

class RadialBasisFunction(nn.Module):
    def __init__(self):
        super(RadialBasisFunction, self).__init__()

    def forward(self, x, gamma, mu):
        r"""
        Parameters:
            x (torch.tensor):
            gamma (float):
            mu (torch.tensor)
        """
        return torch.exp(-gamma * ((x - mu) ** 2))

class EmbeddingBlock(nn.Module):
    def __init__(self, embedding_dim: int):
        super(EmbeddingBlock, self).__init__()

        self.atomic_num_embedding = nn.Embedding(95, embedding_dim, padding_idx = 0)

    def forward(self, atomic_num: Tensor):
        r"""
        Initialize atom's representation

        Parameters:
            atomic_num (torch.Tensor):
                Shape [N]

        Returns:
            node_feat (torch.Tensor):
                Shape [N, embedding_dim]
        """

        return self.atomic_num_embedding(atomic_num)

class FilterGeneratingNetworks(nn.Module):
    def __init__(self, num_filters):
        super(FilterGeneratingNetworks, self).__init__()
        r"""
        Args:
            num_filters (int):
                number of filters
        """
        self.num_filters = num_filters
        self.rbf = RadialBasisFunction()

    def forward(self, node_pos, edge_index, lower_bound, upper_bound, gamma):
        r"""
        Parameters:
            lower_bound (float):
                lower bound for mu values
            upper_bound (float):
                upper bound for mu values
            gamma (float)
            node_pos (torch.tensor):
                3D coordinates of nodes. Shape [N, 3]
            edge_index (torch.tensor)
                Shape [2, E]

        Returns:
            expanded_distance (torch.tensor):
                Shape [E, num_filters]
        """
        source, target = edge_index
        # shape [E]
        distance = torch.sum((node_pos[source] - node_pos[target]) ** 2, dim = -1) ** 0.5
        # shape [E] -> [E, num_filters]
        distance_lifted = torch.broadcast_to(distance.unsqueeze(-1), (edge_index.shape[-1], self.num_filters))

        # shape [num_filters]
        mu = torch.linspace(lower_bound, upper_bound, self.num_filters)
        # shape [num_filters] -> [E, num_filters]
        mu_lifted = torch.broadcast_to(mu.unsqueeze(0), (edge_index.shape[-1], self.num_filters))

        expanded_distance = self.rbf(distance_lifted, gamma, mu_lifted)

        return expanded_distance    
    
class CFConv(nn.Module):
    def __init__(self, num_filters, hidden_dim, out_dim):
        super(CFConv, self).__init__()
        r"""
        Args:
            num_filters (int):
                number of filters filter-generating networks
            hidden_dim (int):
                hidden dim for linear layer
            out_dim (int):
                final output dimension for CFConv
        """
        self.num_filters = num_filters
        self.out_dim = out_dim
        self.filter_generating = FilterGeneratingNetworks(num_filters)
        self.linear_1 = nn.Linear(num_filters, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.activation = ShiftedSoftplus()

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_2.bias.data.fill_(0)

    def forward(self, in_node_feat, node_pos, edge_index, lower_bound, upper_bound, gamma):
        r"""
        Apply Continuous-filter convolution on input node features

        Parameters:
            in_node_feat (torch.tensor):
                Input node features. Shape [N, feat_dim]
            node_pos (torch.tensor):
                3D coordinates of nodes. Shape [N, 3]
            edge_index (torch.tensor):
                Shape [2, E]
            upper_bound (float):
                Upper bound for mu values
            lower_bound (float):
                Lower bound for mu values
            gamma (float):
                gamma value for Radial Basis Function

        Returns: 
            out_node_feat (torch.tensor):
                Output node features. Shape [N, out_dim]
        """
        assert in_node_feat.shape[-1] == self.out_dim
        # shape [E, num_filters]
        expanded_distance = self.filter_generating(node_pos, edge_index, lower_bound, upper_bound, gamma)
        # shape [E, num_filters] -> [E, hidden_dim]
        expanded_distance = self.linear_1(expanded_distance)
        
        expanded_distance = self.activation(expanded_distance)
        # shape [E, hidden_dim] -> [E, out_dim]
        expanded_distance = self.linear_2(expanded_distance)
        expanded_distance = self.activation(expanded_distance)

        assert expanded_distance.shape[-1] == in_node_feat.shape[-1]

        source_index, target_index = edge_index
        # compute message for each edge. shape [E, out_dim]
        message = in_node_feat[target_index] * expanded_distance

        # aggregrate message
        # shape [N, out_dim]
        out_node_feat = torch.zeros(in_node_feat.shape[0], self.out_dim).to(message.dtype)
        # lift target index. shape [E] -> [E, out_dim]
        target_index_lifted = torch.broadcast_to(target_index.unsqueeze(-1), (edge_index.shape[-1], self.out_dim))
        out_node_feat = out_node_feat.scatter_add_(0, target_index_lifted, message)

        return out_node_feat

class InteractionBlock(nn.Module):
    def __init__(self, in_dim, num_filters, hidden_dim):
        super(InteractionBlock, self).__init__()
        
        self.atom_wise_layer_1 = nn.Linear(in_dim, in_dim)
        self.cfconv = CFConv(num_filters, hidden_dim, in_dim)
        self.atom_wise_layer_2 = nn.Linear(in_dim, in_dim)
        self.atom_wise_layer_3 = nn.Linear(in_dim, in_dim)
        self.activation = ShiftedSoftplus()

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.atom_wise_layer_1.weight)
        self.atom_wise_layer_1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_wise_layer_2.weight)
        self.atom_wise_layer_2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.atom_wise_layer_3.weight)
        self.atom_wise_layer_3.bias.data.fill_(0)
        

    def forward(self, in_node_feat, node_pos, edge_index, lower_bound, upper_bound, gamma):
        r"""
        Parameters:

        Returns:
        """
            
        # [N, in_dim] -> [N, in_dim]
        node_feat = self.atom_wise_layer_1(in_node_feat)

        node_feat = self.cfconv(node_feat, node_pos, edge_index, lower_bound, upper_bound, gamma)
        node_feat = self.atom_wise_layer_2(node_feat)
        node_feat = self.activation(node_feat)
        out_node_feat = self.atom_wise_layer_3(node_feat)
        # residual connection
        out_node_feat = out_node_feat + in_node_feat

        return out_node_feat

class SchNet(nn.Module):
    def __init__(self, num_interaction_block: int, hidden_dim: int, num_filters):
        super(SchNet, self).__init__()

        self.embedding_block = EmbeddingBlock(hidden_dim)

        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(hidden_dim, num_filters, hidden_dim) for _ in range(num_interaction_block)
        ])

        self.linear_1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activation = ShiftedSoftplus()
        self.linear_2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, atomic_num: Tensor, node_pos: Tensor, edge_index: Tensor,
            lower_bound: float = 0.0, upper_bound: float = 30.0, gamma: float = 10.0):
        r"""
        Parameters:
            atomic_num (torch.Tensor):
                Atomic number of each atom. Shape [N]
            node_pos (torch.Tensor):
                3D Coordinates of each atom. Shape [N, 3]
            edge_index (torch.Tensor):
                Shape [2, E]
            lower_bound (float):
                Lower bound for centers' value in radial basis function
            upper_bound (float):
                Upper bound for centers' value in radial basis function
            gamma (float):
                Gamma value for radial basis function
        """

        # Shape [N, hidden_dim]
        in_node_feat = self.embedding_block(atomic_num)

        for interaction_block in self.interaction_blocks:
            in_node_feat = interaction_block(in_node_feat, node_pos, edge_index, lower_bound, upper_bound, gamma)

        # Shape [N, hidden_dim] -> [N, hidden_dim // 2]
        in_node_feat = self.linear_1(in_node_feat)
        in_node_feat = self.activation(in_node_feat)

        # Shape [N, hidden_dim] -> [N, 1]
        in_node_feat = self.linear_2(in_node_feat)

        energy = torch.sum(in_node_feat.squeeze())

        return energy