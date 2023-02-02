import sys
sys.path.append("./")
sys.path.append("./utils")

import numpy as np
import torch
import torch.nn as nn
import sympy as sym
import math

from torch import Tensor
from typing import Callable, Optional

from utils.dimenet_utils import real_sph_harm, bessel_basis

def convert_edge_index_to_adjacency_list(edge_index: Tensor or np.array, num_nodes: int):
    source, target = edge_index
    adjacency_list = [[] for _ in range(num_nodes)]

    for edge_id, (neighbor, vertex) in enumerate(zip(source, target)):
        vertex, neighbor = int(vertex), int(neighbor)
        adjacency_list[vertex].append((neighbor, edge_id))

    return adjacency_list

def angle_index_from_adjacency_list(adjacency_list: list, num_nodes: int) -> list:
    r"""
    Returns angle (a(kj, ji) and kj's edge index)
    i<-j<-k (j is i's neighbor and k is j's neighbor)
    """
    angle_index = []

    for i in range(num_nodes):
        for j, ji_edge_index in adjacency_list[i]:
            for k, kj_edge_index in adjacency_list[j]:
                if k == i:
                    continue
                
                angle_index.append([kj_edge_index, ji_edge_index])

    return angle_index   

class Envelope(nn.Module):
    def __init__(self, exponent: int):
        super(Envelope, self).__init__()

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = - self.p * (self.p + 1) / 2

    def forward(self, d: Tensor) -> Tensor:
        exponent = self.p - 1
        d_p = d.pow(exponent)
        d_p_1 = d_p * d
        d_p_2 = d_p_1 * d

        return (1 / d + self.a * d_p + self.b * d_p_1 + self.c * d_p_2) * (d < 1.0).to(d.dtype)

class RadialBesselBasis(nn.Module):
    def __init__(self, num_radial_basis: int, cut_off: float, envelope_exponent: int):
        super(RadialBesselBasis, self).__init__()
        self.num_radial_basis = num_radial_basis
        self.cut_off = cut_off
        self.envelope_function = Envelope(envelope_exponent)
        
        # shape [num_radial_basis]
        self.wave_numbers = nn.Parameter(torch.Tensor(num_radial_basis))
        
        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Initialize wave numbers to n * pi
        """
        with torch.no_grad():
            torch.arange(1, self.num_radial_basis + 1, out = self.wave_numbers).mul_(torch.pi)
        self.wave_numbers.requires_grad_()

    def forward(self, distance: Tensor) -> Tensor:
        r"""
        Compute Radial Basis Function representation of interatomic distance

        Parameters:
            distance (torch.Tensor):
                Interatomic distance. Shape [E]

        Returns:
            distance_representation (torch.Tensor):
                Shape [E, num_radial_basis]
                
        """
        distance_scaled = distance / self.cut_off # d / c
        distance_scaled = distance_scaled.unsqueeze(-1)
        # shape [E, num_radial_basis]
        distance_representation = self.envelope_function(distance_scaled) * torch.sin(distance_scaled * self.wave_numbers)

        return distance_representation

class SphericalBesselBasis(nn.Module):
    def __init__(self, num_radial_basis: int, num_spherical_basis: int, cut_off: float, envelope_exponent: int):
        super(SphericalBesselBasis, self).__init__()

        self.num_radial_basis = num_radial_basis
        self.num_spherical_basis = num_spherical_basis

        self.cut_off = cut_off
        self.envelope_function = Envelope(envelope_exponent)

        bessel_formulas = bessel_basis(num_spherical_basis, num_radial_basis)
        spherical_harmonics_formulas = real_sph_harm(num_spherical_basis)

        self.bessel_functions = []
        self.spherical_harmonics = []

        # distance d & angle alpha
        d, alpha = sym.symbols("x theta")
        modules = {'sin': torch.sin, 'cos': torch.cos}

        for l in range(num_spherical_basis):
            
            if l == 0:
                first_y = sym.lambdify([alpha], spherical_harmonics_formulas[l][0], modules)(0)
                self.spherical_harmonics.append(lambda d: torch.zeros_like(d) + first_y)
            else:
                y = sym.lambdify([alpha], spherical_harmonics_formulas[l][0], modules)
                self.spherical_harmonics.append(y)
            
            for n in range(num_radial_basis):
                j = sym.lambdify([d], bessel_formulas[l][n], modules)
                self.bessel_functions.append(j)

    def forward(self, distance: Tensor, angle: Tensor, angle_index: Tensor) -> Tensor:
        r"""
        Compute angle representation using spherical Bessel functions and spherical harmonics

        Parameters:
            distance (torch.Tensor):
                Interatomic distance. Shape [E]
            angle (torch.Tensor):
                Angle between 2 bonds. Shape [A] (A = number of angles)
            angle_index (torch.Tensor):
                Shape [2, A]
        Returns: 
            angle_representation (torch.Tensor):
                Shape [A, num_spherical_basis, num_radial_basis]
        """
        kj_index, ji_index = angle_index
        distance_scaled = distance / self.cut_off
        
        # shape [A, num_spherical_basis]
        cbf = torch.stack([y(angle) for y in self.spherical_harmonics], dim = -1)

        d_kj = distance_scaled[kj_index]
        # shape [A, num_spherical_basis * num_radial_basis]
        rbf = self.envelope_function(d_kj).unsqueeze(-1) * torch.stack([j(d_kj) for j in self.bessel_functions], dim = -1)
        # shape [A, num_spherical_basis * num_radial_basis] -> [A, num_spherical_basis, num_radial_basis]
        rbf = rbf.view(-1, self.num_spherical_basis, self.num_radial_basis)

        # shape [A, num_spherical_basis, num_radial_basis]
        angle_representation = rbf * cbf.unsqueeze(-1)

        return angle_representation

class EmbeddingBlock(nn.Module):
    def __init__(self, num_radial_basis: int, hidden_dim: int, activation = None):
        super(EmbeddingBlock, self).__init__()

        self.atomic_num_embedding = nn.Embedding(95, hidden_dim, padding_idx = 0)
        self.linear_distance = nn.Linear(num_radial_basis, hidden_dim)
        self.linear = nn.Linear(3 * hidden_dim, hidden_dim)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.atomic_num_embedding.weight, -math.sqrt(3), math.sqrt(3))
        nn.init.orthogonal_(self.linear_distance.weight)
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, atomic_number: Tensor, distance_representation: Tensor, edge_index: Tensor) -> Tensor:
        r"""
        Initialize message embeddings 

        Parameters:
            atomic_number (torch.Tensor):
                Atoms' atomic number. Shape [N]
            distance_representation (torch.Tensor):
                Radial Basis function representation of interatomic distance. Shape [E, num_radial]
            edge_index (torch.Tensor):
                Shape [2, E]
                        
        Returns:
            message (torch.Tensor):
                Message embeddings. Shape [E, hidden_dim]
        """
        # Shape [E, num_radial] -> [E, hidden_dim]
        distance_representation = self.linear_distance(distance_representation)
        
        source, target = edge_index
        
        # Shape [E, hidden_dim]
        h_i = self.atomic_num_embedding(atomic_number[target])
        h_j = self.atomic_num_embedding(atomic_number[source])

        # Shape [E, 3 * hidden_dim]
        message_ji = torch.cat([h_j, h_i, distance_representation], dim = -1)   
        # Shape [E, 3 * hidden_dim] -> [E, hidden_dim]
        message = self.linear(message_ji)
        if self.activation is not None:
            message = self.activation(message)

        return message

class ResidualLayer(nn.Module):
    def __init__(self, in_dim: int, use_bias = True, activation = None):
        super(ResidualLayer, self).__init__()

        self.linear_1 = nn.Linear(in_dim, in_dim, bias = use_bias)
        self.linear_2 = nn.Linear(in_dim, in_dim, bias = use_bias)
        self.activation = activation

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear_1.weight)
        if self.linear_1.bias is not None:
            self.linear_1.bias.data.fill_(0)

        nn.init.orthogonal_(self.linear_2.weight)
        if self.linear_2.bias is not None:
            self.linear_2.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        x_0 = x
        
        x = self.linear_1(x)
        if self.activation is not None:
            x = self.activation(x)

        x = self.linear_2(x)
        if self.activation is not None:
            x = self.activation(x)

        return x_0 + x

class OutputBlock(nn.Module):
    def __init__(self, num_radial_basis: int, hidden_dim: int, out_dim: int, num_linear_layers: int, activation = None):
        super(OutputBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.linear_distance = nn.Linear(num_radial_basis, hidden_dim, bias = False)
        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias = True) for _ in range(num_linear_layers)])
        self.linear_out = nn.Linear(hidden_dim, out_dim, bias = False)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear_distance.weight)
        
        for linear_layer in self.linear_layers:
            nn.init.orthogonal_(linear_layer.weight)
            linear_layer.bias.data.fill_(0)

        nn.init.orthogonal_(self.linear_out.weight)

    def forward(self, distance_representation: Tensor, message: Tensor, edge_index: Tensor, num_nodes: int):
        r"""
        Transform message embeddings with distance representation and compute atom-wise output

        Parameters: 
            distance_representation (torch.Tensor):
                Radial Basis function representation of interatomic distance. Shape [E, num_radial_basis]
            message (torch.Tensor):
                Message embeddings. Shape [E, hidden_dim] 
            edge_index (torch.Tensor):
                Shape [2, E]
            num_nodes (int)

        Returns: 
            out_node_embedding (torch.Tensor):
                Shape [N, out_dim]
        """
        # shape [E, num_radial_basis] -> [E, hidden_dim]
        distance_representation = self.linear_distance(distance_representation)
        # shape [E, hidden_dim]
        transformed_message = distance_representation * message

        # aggregrate message for each atom
        source_index, target_index = edge_index
        # shape [E] -> [E, hidden_dim]
        target_index_lifted = torch.broadcast_to(target_index.unsqueeze(-1), (edge_index.shape[-1], self.hidden_dim))
        # shape [N, hidden_dim]
        node_embedding = torch.zeros(num_nodes, self.hidden_dim).scatter_add_(0, target_index_lifted, transformed_message)

        for linear_layer in self.linear_layers:
            node_embedding = linear_layer(node_embedding)
            if self.activation is not None:
                node_embedding = self.activation(node_embedding)

        # shape [N, hidden_dim] -> [N, out_dim]
        out_node_embedding = self.linear_out(node_embedding)

        return out_node_embedding

class InteractionBlock(nn.Module):
    def __init__(self, num_radial_basis: int, num_spherical_basis: int, hidden_dim: int, bilinear_dim: int, 
                message_in_dim: int, num_layers_before_skip: int, num_layers_after_skip: int, activation = None):
        super(InteractionBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.linear_distance = nn.Linear(num_radial_basis, hidden_dim, bias = False)
        self.linear_angle = nn.Linear(num_spherical_basis * num_radial_basis, bilinear_dim, bias = False)

        self.linear_source_message = nn.Linear(message_in_dim, hidden_dim) # kj
        self.linear_target_message = nn.Linear(message_in_dim, hidden_dim) # ji

        self.weight_bilinear = nn.Parameter(torch.Tensor(hidden_dim, bilinear_dim, hidden_dim))

        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(hidden_dim, use_bias = True, activation = activation) for _ in range(num_layers_before_skip)
        ])

        self.linear_skip = nn.Linear(hidden_dim, message_in_dim)

        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(message_in_dim, use_bias = True, activation = activation) for _ in range(num_layers_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear_distance.weight)
        nn.init.orthogonal_(self.linear_angle.weight)
        
        nn.init.orthogonal_(self.linear_source_message.weight)
        self.linear_source_message.bias.data.fill_(0)
        nn.init.orthogonal_(self.linear_target_message.weight)
        self.linear_target_message.bias.data.fill_(0)

        self.weight_bilinear.data.normal_(mean = 0, std = 2 / self.hidden_dim)

        for layer in self.layers_before_skip:
            layer.reset_parameters()

        nn.init.orthogonal_(self.linear_skip.weight)
        self.linear_skip.bias.data.fill_(0)

        for layer in self.layers_before_skip:
            layer.reset_parameters()

    def forward(self, distance_representation: Tensor, angle_representation: Tensor, message: Tensor, 
                angle_index: Tensor) -> Tensor:
        r"""
        Update message embeddings

        Parameters: 
            distance_representation (torch.Tensor):
                Shape [E, num_radial_basis]
            angle_representation (torch.Tensor):
                Shape [E, num_shperical_basis, num_radial_basis]
            message (torch.Tensor):
                Shape [E, message_in_dim]
            angle_index (torch.Tensor):
                Shape [2, A]
            
        Returns:

        """
        num_edges, num_angles = distance_representation.shape[0], angle_index.shape[-1]
        # shape [E, num_radial_basis] -> [E, hidden_dim]
        distance_representation = self.linear_distance(distance_representation)
        # shape [E, num_spherical_basis, num_radial_basis] -> [E, num_spherical_basis * num_radial_basis] -> [E, bilinear_dim]
        angle_representation = self.linear_angle(angle_representation.view(angle_representation.shape[0], -1))

        source_edge, target_edge = angle_index
        source_message, target_message = message[source_edge], message[target_edge]
        # shape [A, message_in_dim] -> [A, hidden_dim]
        source_message = self.linear_source_message(source_message)
        source_message = source_message * distance_representation[target_edge]
        # shape [A, hidden_dim]
        source_message = torch.einsum('ab,ah,ibh->ai', angle_representation, source_message, self.weight_bilinear)

        # aggregrate source message
        # shape [A, hidden_dim]
        target_index_lifted = torch.broadcast_to(target_edge.unsqueeze(-1), (num_angles, self.hidden_dim))
        # shape [E, hidden_dim]
        aggregrated_message = torch.zeros(num_edges, self.hidden_dim).scatter_add_(0, target_index_lifted, source_message) 
        
        # residual
        x, x_0 = aggregrated_message + self.linear_target_message(message), message

        for layer in self.layers_before_skip:
            x = layer(x)
                
        # shape [E, hidden_dim] -> [E, message_in_dim]
        x = self.linear_skip(x)
        if self.activation is not None:
            x = self.activation(x)

        x = x + x_0
        # shape [E, message_in_dim] -> [E, message_in_dim]
        for layer in self.layers_after_skip:
            x = layer(x)

        updated_message = x
        
        return updated_message

class DimeNet(nn.Module):
    def __init__(self, num_radial_basis: int, num_spherical_basis: int, embedding_dim: int, bilinear_dim: int,
            out_dim: int, cut_off: float, envelope_exponent: int, num_interaction_blocks: int, 
            num_layers_before_skip: int, num_layers_after_skip: int, num_output_linear_layers: int,
            activation = None):
        super(DimeNet, self).__init__()

        self.rbf = RadialBesselBasis(num_radial_basis, cut_off, envelope_exponent)
        self.sbf = SphericalBesselBasis(num_radial_basis, num_spherical_basis, cut_off, envelope_exponent)

        self.embedding_block = EmbeddingBlock(num_radial_basis, embedding_dim, activation)

        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(num_radial_basis, num_spherical_basis, embedding_dim, bilinear_dim, 
            embedding_dim, num_layers_before_skip, num_layers_after_skip, 
            activation) for _ in range(num_interaction_blocks)
        ])

        self.output_blocks = nn.ModuleList([
            OutputBlock(num_radial_basis, embedding_dim, out_dim, 
                num_output_linear_layers, activation) for _ in range(num_interaction_blocks + 1) 
        ])

        self.reset_parameters()        

    def reset_parameters(self):        
        self.rbf.reset_parameters()
        self.embedding_block.reset_parameters()

        for interaction_block in self.interaction_blocks:
            interaction_block.reset_parameters()

        for output_block in self.output_blocks:
            output_block.reset_parameters()

    def forward(self, atomic_number: Tensor, edge_index: Tensor, angle_index: Tensor, distance: Tensor, 
        angle: Tensor) -> Tensor:
        r"""
        Parameters:
            atomic_number (torch.Tensor):
                Atomic number of atoms. Shape [N]
            edge_index (torch.Tensor):
                Shape [2, E]
            angle_index (torch.Tensor):
                Shape [2, A]
            distance (torch.Tensor):
                Shape [E]
            angle (torch.Tensor):
                Shape [A]
        
        Returns:
            output (torch.Tensor)
        """

        num_nodes = atomic_number.shape[0]

        distance_representation = self.rbf(distance)
        angle_representation = self.sbf(distance, angle, angle_index).to(torch.float32)

        message = self.embedding_block(atomic_number, distance_representation, edge_index)
        t = self.output_blocks[0](distance_representation, message, edge_index, num_nodes)

        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            message = interaction_block.forward(distance_representation, angle_representation, message,
                    angle_index)
            t = t + output_block(distance_representation, message, edge_index, num_nodes)

        output = torch.sum(t, dim = 0)

        return output