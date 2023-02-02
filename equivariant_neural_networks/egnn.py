import torch
import numpy as np
import torch.nn as nn
from torch import Tensor

class Swish(nn.Module):
    def __init__(self, beta: float):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.beta = beta

    def forward(self, x):
        return x * self.sigmoid(self.beta * x)
        
class EquivariantGraphConvolutionalLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, swish_beta: float, velocity: bool = False):
        super(EquivariantGraphConvolutionalLayer, self).__init__()
        activation = Swish(swish_beta)

        self.edge_function = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden_dim),
            activation, 
            nn.Linear(hidden_dim, hidden_dim),
            activation
        )

        self.coordinate_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )

        self.node_function = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, in_dim)
        )

        if velocity:
            self.velocity_function = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, 1)
            )

        self.velocity = velocity

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_function[0].weight)
        self.edge_function[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.edge_function[2].weight)
        self.edge_function[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.coordinate_function[0].weight)
        self.coordinate_function[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.coordinate_function[2].weight)
        self.coordinate_function[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.node_function[0].weight)
        self.node_function[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.node_function[2].weight)
        self.node_function[2].bias.data.fill_(0)
        if self.velocity:
            nn.init.xavier_uniform_(self.velocity_function[0].weight)
            self.velocity_function[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.velocity_function[2].weight)
            self.velocity_function[2].bias.data.fill_(0)


    def forward(self, node_feat: Tensor, degree: Tensor, coordinate: Tensor, edge_index: Tensor, velocity_vector: Tensor = None):
        r"""
        Parameters: 
            node_feat (torch.Tensor):
                Node features. Shape [N, n, in_dim]
            degree (torch.Tensor):
                Shape [N]
            coordinate (torch.Tensor):
                Shape [N, n]
            edge_index (torch.tensor):
                Shape [2, E]
        """
        if self.velocity:
            assert velocity_vector is not None

        num_nodes, num_edges = node_feat.shape[0], edge_index.shape[-1]
        num_dimensions = coordinate.shape[-1]
        # j, i (i -> j)
        source, target = edge_index
        # x_i - x_j. Shape [E, n]
        relative_difference = coordinate[target] - coordinate[source]
        # Shape [E]
        distance = torch.sum(relative_difference ** 2, dim = -1)
        # Shape [E] -> unsqueeze [E, 1] -> broadcast to [E, n, 1]
        distance_lifted = torch.broadcast_to(distance.unsqueeze(-1).unsqueeze(1), (num_edges, num_dimensions, 1))

        # Compute message
        # Shape [E, n, in_dim]
        source_feat, target_feat = node_feat[source], node_feat[target]
        # Shape [E, n, 2 * in_dim + 1]
        message = torch.cat([target_feat, source_feat, distance_lifted], dim = -1)
        # Shape [E, n, 2 * in_dim + 1] -> [E, n, hidden_dim]
        message = self.edge_function(message)

        ## Update coordinate
        # Shape [E] -> [E, n]
        target_index_lifted = torch.broadcast_to(target.unsqueeze(-1), (num_edges, num_dimensions))
        # Shape [E, n]
        coordinate_message = relative_difference * self.coordinate_function(message).squeeze()
        # Aggregate message
        updated_coordinate = torch.zeros(num_nodes, num_dimensions).scatter_add_(0, target_index_lifted, coordinate_message)
        inv_degree = (1 / degree).unsqueeze(-1)
        # Shape [N, n]
        coordinate = coordinate + inv_degree * updated_coordinate

        ## Update velocity
        if self.velocity is not None:
            # Shape [N, n]
            velocity_vector = velocity_vector * self.velocity_function(node_feat).squeeze()
            coordinate = coordinate + velocity_vector

        ## Update node feature
        hidden_dim = message.shape[-1]
        # Shape [E] -> [E, n, hidden_dim]
        target_index_lifted = torch.broadcast_to(target.unsqueeze(-1).unsqueeze(-1), (num_edges, num_dimensions, hidden_dim))
        # Aggregrate message
        # Shape [N, n, hidden_dim]
        aggregrated_message = torch.zeros(num_nodes, num_dimensions, hidden_dim).scatter_add_(0, target_index_lifted, message)
        # Shape [N, n, in_dim + hidden_dim] -> [N, n, in_dim]
        updated_node_feat = self.node_function(torch.cat([node_feat, aggregrated_message], dim = -1))
        node_feat = node_feat + updated_node_feat
 
        if self.velocity is not None:
            return coordinate, node_feat, velocity_vector

        return coordinate, node_feat