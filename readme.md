# Equivariant Graph Neural Networks

Implementation of Geometrically Equivariant Graph Neural Networks (GNNs) in PyTorch.

When working with molecules, understanding atoms' geometric vectors (positions, velocities, etc) is important since they will tell us more about the molecules' properties or functions. How can we integrate these geometric vectors into a GNN, so that the network could better leverage 3D structural information?

**Invariance/ Equivariance:**

Suppose we want to use a neural network to predict a molecular property (dipole moment, ...). Since irr egardless of how the molecule is translated or rotated, the property is still the same, so the network is expected to be invariant to translations and rotations of the molecule.

However, if we want a model to predict the atomic forces for each atom then the model should be equivariant to rotations, since how the molecule is rotated or translated, the atomic forces is rotated or translated accordingly. 

Therefore, based on the task, a network is expected to preserve invariance or equivariance. 

# Table of Contents

* [SchNet](#schnet)

* [DimeNet](#dimenet)

* [PaiNN](#painn)

* [EGNN](#egnn)

* [GMN](#gmn)

#

## SchNet
[Paper](https://arxiv.org/abs/1706.08566)

SchNet obtains rotationally invariant by transform atomic positions into interatomic distances, expand the distance by Gaussian radial basis function, and CFConv block further transforms the distance to compute the filter weight $\mathbf{W}$. As each filter emphasizes certain ranges of interatomic distance, the Interaction Block can update an atom's representation based on its radial environment, .i.e neighborhood. 

```python
class SchNet(num_interaction_blocks: int, hidden_dim: int, num_filters: int)
```

> **Parameters**:
>
> * ```num_interaction_blocks``` (int): Number of Interaction Blocks in the network
>
> * ```hidden_dim``` (int): Size of each atom type embedding
>
> * ```num_filters``` (int): Number of filters for expanding the interatomic distance

**Forward computation**
```python
forward(atomic_num: Tensor, node_pos: Tensor, edge_index: Tensor, lower_bound: float = 0.0, upper_bound: float = 30.0, gamma: float = 10.0):
```

## DimeNet
[Paper](https://arxiv.org/abs/2003.03123)

DimeNet takes both interatomic distance and angles between message embeddings into account: distances and angles are expanded using spherical Bessel functions and 2D spherical Fourier - Bessel basis.  

```python
class DimeNet(num_radial_basis: int, num_spherical_basis: int, embedding_dim: int, bilinear_dim: int, out_dim: int, cut_off: float, envelope_exponent: int, num_interaction_blocks: int, num_layers_before_skip: int, num_layers_after_skip: int, num_output_linear_layers: int, activation = None):
```

> **Parameters**:
>
> * ```num_radial_basis``` (int): Number of radial basis for interatomic distance representations
>
> * ```num_spherical_basis``` (int): Number of basis for angle representations
>
> * ```embedding_dim``` (int): Size of each message embedding
> * ```bilinear_dim```(int): Size of each weight tensor in the bilinear layer
>
> * ```out_dim``` (int): Output size of each node embedding after applying the Output Block
> 
> * ```cut_off``` (float): Cutoff range for an atom's neighborhood
> 
> * ```envelope_exponent``` (float): exponent of the Envelope function
> 
> * ```num_interaction_blocks``` (int): Number of the interactions blocks being used in the network
> 
> * ```num_layers_before_skip``` (int): Number of Residual Layer before applying skip connection in Interaction Block
> 
> * ```num_layers_after_skip``` (int): Number of Residual Layer after applying skip connection in Interaction Block
> 
> * ```num_output_linear_layers``` (int): Number of Linear layers in Output Block 
> 
> * ```activation```: activation function
> 

**Forward computation**
```python
forward(atomic_number: Tensor, edge_index: Tensor, angle_index: Tensor, distance: Tensor: angle: Tensor)
```


## EGNN
[Paper](https://arxiv.org/abs/2102.09844)

EGNN introduces an architecture that is equivariant to translation, rotation, and relection. For each message passing layer, the invariant message is constructed using invariant features (node features, edge attributes) and the scalarization of geometric vectors ( $\mathbf{x}_i, \mathbf{x}_j \rightarrow$ distance $|| \mathbf{x}_i - \mathbf{x}_j ||^2$ ) (since scalars are invariant). Then geometric vectors are updated in the equivariant flavor (weighted sum of all relative differences) while node features are updated by aggregating messages. 

```python
class EquivariantGraphConvolutionalLayer(in_dim: int, hidden_dim: int, swish_beta: float, velocity: bool = False)
```
> **Parameters**:
>
> * ```in_dim``` (int): Dimension of input node features
>
> * ```hidden_dim``` (int): Hidden dimension
>
> * ```swish_beta``` (float): Parameters for Swish activation function


**Forward computation**
```python
forward(node_feat: Tensor, degree: Tensor, coordinate: Tensor, edge_index: Tensor, velocity_vector: Tensor = None)
```


## PaiNN
[Paper](https://arxiv.org/abs/2102.03150)

PaiNN comprises of 2 blocks, message and update, that update equivariant and invariant features iteratively. The Message block compute invariant message using invariant features and rotationally-invariant filters (a linear combination of expanded interatomic distances by applying radial basis functions to interatomic distances), and the equivariant message is computed by a convolution of an invariant filter with equivariant features (which yields equivariance) and a convolution of invariant features with an equivariant filter (which also yields equivariance). The update of invariant features is calculated using scaling functions and the update of equivariant features is calculated using scaling functions with a linear combination of equivariant features. 

```python
class PaiNN(embedding_dim: int, num_blocks: int, num_radial_basis: int, cut_off: float, out_dim: int, return_potential_energy: bool = True, return_force_vectors: bool = True):
```
> **Parameters**:
>
> * ```embedding_dim``` (int): Dimensions of atom type embeddings
>
> * ```num_blocks``` (int): Number of Message/ Update blocks
>
> * ```num_radial_basis``` (int): Number of radial basis
> 
> * ```cut_off``` (float): Cutoff range for an atom's neighborhood
> 
> * ```out_dim``` (int): Output dimensions for scalar features
>
> * ```return_potential_energy``` (bool, optional): If set to ```False```, the network will not return the potential energy (default: ```True```)
>
> * ```return_force_vectors``` (bool, optional): If set to ```False```, the network will not return atomic forces (default: ```True```)

**Forward computation**
```python
forward(atomic_num: Tensor, node_pos: Tensor, edge_index: Tensor):
```


## GMN
[Paper](https://openreview.net/forum?id=SHbhHHfePhP)

GMN extends the message passing paradigm of EGNN to functions with multiple input vectors. Moreover, GMN propose an architecture that preserves geometry constraints by incoporating forward kinematics information of an object (stick or hinge) into the network. 

```python
class GMNLayer(in_dim: int, hidden_dim :int, out_dim: int, edge_attr_dim: int = 0, activation = nn.ReLU(), use_residual_connection: bool = True, learnable: bool = False)
```
> **Parameters**:
>
> * ```in_dim``` (int): Dimension of input node features
>
> * ```hidden_dim``` (int): Hidden dimension
>
> * ```out_dim``` (int): Output dimension for updated node features
> 
> * ```edge_attr_dim``` (int): Dimensions of edge attributes
> 
> * ```use_residual_connection``` (bool, optional): If set to ```False```, the layer will not add a residual connection to update node features (default: ```True```)
>
> * ```learnable``` (bool, optional): If set to ```True```, the layer will apply learnable forward kinematics for objects (default: ```False```)

**Forward computation**
```python
forward(node_feat: Tensor, node_pos: Tensor, velocity: Tensor, edge_index: Tensor, degree: Tensor, object_index, edge_attr = None)
```


# Citations
K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.
Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017)

K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko.
Quantum-chemical insights from deep tensor neural networks.
Nature Communications 8. 13890 (2017)
doi: 10.1038/ncomms13890
```
@inproceedings{gasteiger_dimenet_2020,
  title = {Directional Message Passing for Molecular Graphs},
  author = {Gasteiger, Johannes and Gro{\ss}, Janek and G{\"u}nnemann, Stephan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year = {2020}
}

@inproceedings{gasteiger_dimenetpp_2020,
title = {Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules},
author = {Gasteiger, Johannes and Giri, Shankari and Margraf, Johannes T. and G{\"u}nnemann, Stephan},
booktitle={Machine Learning for Molecules Workshop, NeurIPS},
year = {2020} }
```

[Schütt et al., 2021] Kristof Schütt, Oliver Unke, and Michael Gastegger. Equivariant message passing for the prediction of tensorial properties and molecular spectra. In ICML, 2021.

[Satorras et al., 2021b] Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. arXiv preprint arXiv:2102.09844, 2021.

```
@inproceedings{
huang2022equivariant,
title={Equivariant Graph Mechanics Networks with Constraints},
author={Wenbing Huang and Jiaqi Han and Yu Rong and Tingyang Xu and Fuchun Sun and Junzhou Huang},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=SHbhHHfePhP}
}
```
