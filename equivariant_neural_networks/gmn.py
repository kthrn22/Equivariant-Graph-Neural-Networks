import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import List, Dict, Tuple

class GMNLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim :int, 
        out_dim: int, edge_attr_dim: int = 0, activation = nn.ReLU(),
        use_residual_connection: bool = True, learnable: bool = False
        ):
        super(GMNLayer, self).__init__()

        self.edge_function = nn.Sequential(
            nn.Linear(in_dim * 2 + 1 + edge_attr_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.coordinate_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1, bias = False),
        )

        self.node_function = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, out_dim),
        )

        num_basis_stick, num_basis_hinge = 1, 3

        self.stick_mlp = nn.Sequential(
            nn.Linear(num_basis_stick * num_basis_stick, hidden_dim),
            activation,
            nn.Linear(hidden_dim, num_basis_stick)
        )

        self.hinge_mlp = nn.Sequential(
            nn.Linear(num_basis_hinge * num_basis_hinge, hidden_dim),
            activation,
            nn.Linear(hidden_dim, num_basis_hinge)
        )

        if learnable:
            self.stick_mlp_learnable = nn.Sequential(
                nn.Linear(3 * 3, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, 3)
            )

            self.hinge_mlp_learnable = nn.Sequential(
                nn.Linear(3 * 3, hidden_dim),
                activation, 
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, 3)
            )

        self.cartersian_velocity_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )

        self.angle_velocity_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )

        self.center_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, in_dim)
        )

        self.learnable_fk = learnable
        self.use_residual_connection = use_residual_connection

    def equivariant_message_passing(self, Z: Tensor, type: str):
        r"""
        Parameters:
            Z (torch.Tensor):
                Shape [N, 3, num_basis]
        """
        # Shape [N, num_basis, num_basis]
        invariant = torch.bmm(Z.permute(0, 2, 1), Z)
        # Shape [N, num_basis ** 2]
        invariant = invariant.view(-1, invariant.shape[-1] * invariant.shape[-2])
        invariant = nn.functional.normalize(invariant, dim = -1, p = 2)
        
        # Shape [N, num_basis ** 2] -> [N, num_basis]
        if type == "stick":
            if self.learnable_fk: 
                invariant = self.stick_mlp_learnable(invariant)
            else:
                invariant = self.stick_mlp(invariant)

        elif type == "hinge":
            if self.learnable_fk:
                invariant = self.hinge_mlp(invariant)
            else:
                invariant = self.hinge_mlp(invariant)

        else:
            raise NotImplementedError

        # Shape [N, 3, num_basis], [N, num_basis, 1] -> [N, 3, 1]
        equivariant_message = torch.bmm(Z, invariant.unsqueeze(-1))
        # Shape [N, 3, 1] -> [N, 3]
        equivariant_message = equivariant_message.squeeze(-1)

        return equivariant_message

    def rotation_matrix(self, angle, d):
        # From https://github.com/hanjq17/GMN/blob/d26f7fb0062442e8b39cf2821c5c4bd055a1a64f/spatial_graph/models/layer.py
        x, y, z = torch.unbind(d, dim = -1)
        cos, sin = torch.cos(angle), torch.sin(angle)
        rot = torch.stack([
            cos + (1 - cos) * x * x,
            (1 - cos) * x * y - sin * z,
            (1 - cos) * x * z + sin * y,
            (1 - cos) * x * y + sin * z,
            cos + (1 - cos) * y * y,
            (1 - cos) * y * z - sin * x,
            (1 - cos) * x * z - sin * y,
            (1 - cos) * y * z + sin * x,
            cos + (1 - cos) * z * z,
        ], dim = -1)

        return rot.view(-1, 3, 3)

    def update_object_isolated(self, node_feat, node_pos, velocity, force, object_index):
        r"""
        Parameters:
            node_feat (torch.Tensor):
                Node features. Shape [N, in_dim]
            node_pos (torch.Tensor):
                Node 3D Coordinates. Shape [N, 3]
            velocity (torch.Tensor):
                Velocity vectors. Shape [N, 3]
            force (torch.Tensor):
                Force vectors. Shape [N, 3]
            object_index (torch.Tensor):
                Sticks. Shape [K] (K is the number of objects)
        """
        # Shape [K, 3], [K, 3], [K, 3], [K, in_dim]
        x, v, f, h = node_pos[object_index], velocity[object_index], force[object_index], node_feat[object_index]
        # Shape [K, 3]
        v = self.cartersian_velocity_mlp(h) * v + f
        x = x + v

        node_pos[object_index], velocity[object_index] = x, v

        return node_pos, velocity

    def learnable_forward_kinetics(self, node_feat: Tuple, node_pos: Tuple, center_pos: Tensor, velocity: Tuple, 
    force: Tuple, cartesian_acceleration: Tensor, type: str):
        r"""
        Parameters:
            node_feat (Tuple):
                Node features. Shape ([K, in_dim], [K, in_dim])
            node_pos (Tuple):
                Node 3D Coordinates. Shape ([K, 3], [K, 3])
            center_pos (torch.Tenso):
                Objects generalized Cartesian position. Shape [K, 3]
            velocity (Tuple):
                Velocity vectors. Shape ([K, 3], [K, 3])
            force (Tuple):
                Force vectors. Shape ([K, 3], [K, 3])
            cartesian_acceleration (torch.Tensor):
                Generalized Cartesian acceleration vectors of objects. Shape [K, 3]
            type (str):
                Object type: "stick", "hinge"
        """
        # Shape [K, 3]
        x_1, h_1, f_1, v_1 = node_pos[0], node_feat[0], force[0], velocity[0]
        x_2, h_2, f_2, v_2 = node_pos[1], node_feat[1], force[1], velocity[1]
        x_center = center_pos

        # Shape [K, 3]
        equivariant_message_1 = self.equivariant_message_passing(torch.stack([cartesian_acceleration, x_1 - x_center, f_1], dim = -1), type = type)
        equivariant_message_2 = self.equivariant_message_passing(torch.stack([cartesian_acceleration, x_2 - x_center, f_2], dim = -1), type = type)

        v_1 = self.cartersian_velocity_mlp(h_1) * v_1 + equivariant_message_1
        v_2 = self.cartersian_velocity_mlp(h_2) * v_2 + equivariant_message_2

        x_1 = x_1 + v_1
        x_2 = x_2 + v_2

        return x_1, v_1, x_2, v_2

    def update_object_stick(self, node_feat, node_pos, velocity, force, stick_index):
        r"""
        Parameters:
            node_feat (torch.Tensor):
                Node features. Shape [N, in_dim]
            node_pos (torch.Tensor):
                Node 3D Coordinates. Shape [N, 3]
            velocity (torch.Tensor):
                Velocity vectors. Shape [N, 3]
            force (torch.Tensor):
                Force vectors. Shape [N, 3]
            stick_index (torch.Tensor):
                Sticks. Shape [2, K] (K is the number of objects)
        """
        # Shape [K]
        index_1, index_2 = stick_index
        # Shape [K, 3]
        x_1, x_2 = node_pos[index_1], node_pos[index_2]
        f_1, f_2 = force[index_1], force[index_2]
        v_1, v_2 = velocity[index_1], velocity[index_2]
        x_center, f_center, v_center = (x_1 + x_2 ) / 2, (f_1 + f_2) / 2, (v_1 + v_2) / 2

        ## Compute generalized Cartersian acceleration
        # Shape [K, 3] -> unsqueeze [K, 3, 1] -> MP -> [K, 3] 
        equivariant_message_1 = self.equivariant_message_passing(f_1.unsqueeze(-1), type = "stick")
        equivariant_message_2 = self.equivariant_message_passing(f_2.unsqueeze(-1), type = "stick")
        # Shape [K, 3]
        cartesian_acceleration = (equivariant_message_1 + equivariant_message_2) / 2

        if self.learnable_fk:
            x_1, v_1, x_2, v_2 = self.learnable_forward_kinetics((h_1, h_2), (x_1, x_2), x_center, (v_1, v_2), 
                    (f_1, f_2), cartesian_acceleration)
        else:    
            ## Compute generalized angle acceleration
            # Shape [K, 1]
            inertia = torch.sum((x_1 - x_center) ** 2, dim = -1, keepdim = True) + torch.sum((x_2 - x_center) ** 2, dim = -1, keepdim = True)
            # Shape [K, 3]
            total_torque = torch.cross(x_1 - x_center, f_1) + torch.cross(x_2 - x_center, f_2)
            # Shape [k, 3]
            angle_acceleration = total_torque / inertia

            ## Compute angle velocity
            # Shape [K, 3]
            r, v_r = (x_1 - x_2) / 2, (v_1 - v_2) / 2
            angle_velocity = torch.cross(nn.functional.normalize(r, dim = -1, p = 2), v_r) / torch.norm(
                r, dim = -1, p = 2, keepdim = True).clamp_min(1e-5)

            ## Update center Cartesian velocity and center Cartesian position
            # Shape [K, in_dim]
            h_1, h_2 = node_feat[index_1], node_feat[index_2]
            # Shape [K, in_dim] -> [K, in_dim]
            h_center = self.center_mlp(h_1) + self.center_mlp(h_2)
            # Shape [K, 3]
            v_center = self.cartersian_velocity_mlp(h_center) * v_center + cartesian_acceleration
            x_center = x_center + v_center

            ## Update angle velocity
            # Shape [K, 3]
            angle_velocity = self.angle_velocity_mlp(h_center) * angle_velocity + angle_acceleration

            ## Update node_pos (position) and velocity
            # Shape [K, 3, 3]
            rotation = self.rotation_matrix(torch.norm(angle_velocity, dim = -1, p = 2), nn.functional.normalize(angle_velocity, dim = -1, p = 2))
            # Shape [K, 3, 3], [K, 3, 1] -> [K, 3, 1] -> [K, 3]
            r = torch.bmm(rotation, r.unsqueeze(-1)).squeeze(-1)
            # Shape [K, 3]
            x_1, x_2 = x_center + r, x_center - r
            v_1, v_2 = v_center + torch.cross(angle_velocity, r), v_center - torch.cross(angle_velocity, r)

        node_pos[index_1], node_pos[index_2] = x_1, x_2
        velocity[index_1], velocity[index_2] = v_1, v_2

        return node_pos, velocity

    def update_object_hinge(self, node_feat, node_pos, velocity, force, hinge_index):
        r"""
        Parameters:
            node_feat (torch.Tensor):
                Node features. Shape [N, in_dim]
            node_pos (torch.Tensor):
                Node 3D Coordinates. Shape [N, 3]
            velocity (torch.Tensor):
                Velocity vectors. Shape [N, 3]
            force (torch.Tensor):
                Force vectors. Shape [N, 3]
            hinge_index (torch.Tensor):
                Hinges. Shape [3, K] (K is the number of objects)
        """
        index_center, index_1, index_2 = hinge_index
        x_center, x_1, x_2 = node_pos[index_center], node_pos[index_1], node_pos[index_2]
        v_center, v_1, v_2 = velocity[index_center], velocity[index_1], velocity[index_2]
        f_center, f_1, f_2 = force[index_center], force[index_1], force[index_2]

        # Shape [K, 3, 3]
        equivariant_message_center = self.equivariant_message_passing(torch.stack([f_center, (x_center - x_center), (v_center - v_center)], dim = -1), type = "hinge")
        equivariant_message_1 = self.equivariant_message_passing(torch.stack([f_1, (x_1 - x_center), (v_1 - v_center)], dim = -1), type = "hinge")
        equivariant_message_2 = self.equivariant_message_passing(torch.stack([f_2, (x_2 - x_center), (v_2 - v_center)], dim = -1), type = "hinge")

        ## Compute generalized Cartesian acceleration
        # Shape [K, 3]
        cartesian_acceleration = (equivariant_message_center + equivariant_message_1 + equivariant_message_2) / 3

        if self.learnable_fk:
            x_1, v_1, x_2, v_2 = self.learnable_forward_kinetics((h_1, h_2), (x_1, x_2), x_center, (v_1, v_2), 
                    (f_1, f_2), cartesian_acceleration)

            ## Update center Cartesian velocity and Cartesian position
            # Shape [K, in_dim]
            h_1, h_2 = node_feat[index_1], node_feat[index_2]
            h_center = self.center_mlp(h_1 + h_2)
            # Shape [K, 3]
            v_center = self.cartersian_velocity_mlp(h_center) * v_center + cartesian_acceleration
            x_center = x_center + v_center
        else:
            ## Compute angle acceleration 
            # Shape [K, 3]
            angle_acceleration_1 = torch.cross((x_1 - x_center), (f_1 - cartesian_acceleration)) / torch.sum((x_1 - x_center) ** 2, dim = -1, keepdim = True)
            angle_acceleration_2 = torch.cross((x_2 - x_center), (f_2 - cartesian_acceleration)) / torch.sum((x_2 - x_center) ** 2, dim = -1, keepdim = True)

            ## Compute angle velocity
            # Shape [K, 3]
            r_1, r_2 = x_1 - x_center, x_2 - x_center
            v_r1, v_r2 = v_1 - v_center, v_2 - v_center
            angle_velocity_1 = torch.cross(nn.functional.normalize(r_1, dim = -1, p = 2), v_r1) / torch.norm(
                r_1, dim = -1, p = 2, keepdim = True).clamp_min(1e-5)
            angle_velocity_2 = torch.cross(nn.functional.normalize(r_2, dim = -1, p = 2), v_r2) / torch.norm(
                r_2, dim = -1, p = 2, keepdim = True).clamp_min(1e-5)

            ## Update center Cartesian velocity and Cartesian position
            # Shape [K, in_dim]
            h_1, h_2 = node_feat[index_1], node_feat[index_2]
            h_center = self.center_mlp(h_1 + h_2)
            # Shape [K, 3]
            v_center = self.cartersian_velocity_mlp(h_center) * v_center + cartesian_acceleration
            x_center = x_center + v_center

            ## Update angle velocity
            # Shape [K, 3]
            angle_velocity_1 = self.angle_velocity_mlp(h_center) * angle_velocity_1 + angle_acceleration_1
            angle_velocity_2 = self.angle_velocity_mlp(h_center) * angle_velocity_2 + angle_acceleration_2

            ## Update node_pos (position) and velocity
            # Shape [K, 3, 3]
            rotation_1 = self.rotation_matrix(torch.norm(angle_velocity_1, dim = -1, p = 2), nn.functional.normalize(angle_velocity_1, dim = -1, p = 2))
            rotation_2 = self.rotation_matrix(torch.norm(angle_velocity_2, dim = -1, p = 2), nn.functional.normalize(angle_velocity_2, dim = -1, p = 2))
            # Shape [K, 3, 3], [K, 3, 1] -> [K, 3, 1] -> [K, 3]
            r_1 = torch.bmm(rotation_1, r_1.unsqueeze(-1)).squeeze(-1)
            r_2 = torch.bmm(rotation_2, r_2.unsqueeze(-1)).squeeze(-1)
            # Shape [K, 3]
            x_1, x_2 = x_center + r_1, x_center + r_2
            v_1, v_2 = v_center + torch.cross(angle_velocity_1, r_1), v_center + torch.cross(angle_velocity_2, r_2)

        node_pos[index_center], node_pos[index_1], node_pos[index_2] = x_center, x_1, x_2
        velocity[index_center], velocity[index_1], velocity[index_2] = v_center, v_1, v_2

        return node_pos, velocity

    def forward(self, node_feat: Tensor, node_pos: Tensor, 
    velocity: Tensor, edge_index: Tensor, degree: Tensor, object_index, edge_attr = None):
        r"""
        Parameters:
            node_feat (torch.Tensor):
                Node features. Shape [N, in_dim]
            node_pos (torch.Tensor):
                Node 3D coordinates. Shape [N, 3]
            velocity (torch.Tensor):
                Velocity vectors. Shape [N, 3]
            edge_index (torch.Tensor):
                Shape [2, E]
            degree (torch.Tensor):
                Node degree. Shape [N]
            object_index:
                Dictionary to store objects (3 types "isolated", "stick", "hinge")
            edge_attr (torch.Tensor):
                Shape [E, edge_attr_dim]
        """
        num_nodes, num_edges = node_feat.shape[0], edge_index.shape[-1]
        # Shape [E], source: j, target: i
        source, target = edge_index 
        # Shape [E, in_dim]
        source_feat, target_feat = node_feat[source], node_feat[target]
        relative_distance = node_pos[target] - node_pos[source]
        # Shape [E]
        distance = torch.sum(relative_distance ** 2, dim = -1)
        
        ## Compute invariant message for edges
        # Shape [E, 2 * in_dim + 1]
        invariant_edge_message = torch.cat([target_feat, source_feat, distance.unsqueeze(-1)], dim = -1)
        if edge_attr is not None:
            # Shape [E, in_dim * 2 + 1 + edge_attr_dim]
            invariant_edge_message = torch.cat([invariant_edge_message, edge_attr], dim = -1)
        # Shape [E, in_dim * 2 + 1 + edge_attr_dim] -> [E, hidden_dim]
        invariant_edge_message = self.edge_function(invariant_edge_message)
        
        ## Compute force
        # Shape [E, 3] ([E, 3] * [E, 1]) (invariant_edge_message: Shape [E, hidden_dim] -> coordinate function -> [E, 1])
        equivariant_message = relative_distance * self.coordinate_function(invariant_edge_message)
        # Aggregate equivariant message 
        # Shape target [E] -> unsqueeze -> [E, 1] -> broadcast -> [E, 3]
        target_index_lifted = torch.broadcast_to(target.unsqueeze(-1), (num_edges, 3))
        # Shape [N, 3]
        aggregated_equivariant_message = torch.zeros((num_nodes, 3)).scatter_add_(0, target_index_lifted, equivariant_message)
        inv_degree = 1 / degree
        force = aggregated_equivariant_message * inv_degree.unsqueeze(-1)

        ## Update object
        # Shape [N, 3]
        node_pos, velocity = self.update_object_isolated(node_feat, node_pos, velocity, force, object_index["isolated"])
        node_pos, velocity = self.update_object_stick(node_feat, node_pos, velocity, force, object_index["stick"])
        node_pos, velocity = self.update_object_hinge(node_feat, node_pos, velocity, force, object_index["hinge"])

        ## Update node features
        hidden_dim = invariant_edge_message.shape[-1]
        # Aggregate invariant edge messages
        # Shape target [E] -> unsqueeze -> [E, 1] -> broadcast -> [E, hidden_dim]
        target_index_lifted = torch.broadcast_to(target.unsqueeze(-1), (num_edges, hidden_dim))
        # Shape [N, hidden_dim]
        aggregated_invariant_message = torch.zeros((num_nodes, hidden_dim)).scatter_add_(0, target_index_lifted, invariant_edge_message)
        # Shape [N, in_dim + hidden_dim] [N, out_dim]
        updated_node_feat = self.node_function(torch.cat([node_feat, aggregated_invariant_message], dim = -1))
        
        if self.use_residual_connection:
            assert updated_node_feat.shape[-1] == node_feat.shape[-1]
            node_feat = node_feat + updated_node_feat
        else:
            node_feat = updated_node_feat

        return node_feat, node_pos, velocity