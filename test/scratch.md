# EGNN

## Functions

**Edge function** $\phi_e$: Linear -> Swish -> Linear -> Swish
**Coordinate function $\phi_x$**: Linear -> Swish -> Linear  ($\R^{nf} \rightarrow \R^1$)
**Node function $\phi_h$**: Linear -> Swish -> Linear -> Residual

## Invariant message

$\mathbf{m}_{ij} = \phi_e(\mathbf{h}_i, \mathbf{h}_j, ||\mathbf{x}_i - \mathbf{x}_j||^2, \alpha_{ij})$

## Update $\mathbf{x}$
$\mathbf{x}_i^{l + 1} = \mathbf{x}_i^l + C \sum_{j \neq i} (\mathbf{x}_i^l - \mathbf{x}_j^l) \phi_x(\mathbf{m}_{ij})$

## Update $\mathbf{h}$
$\mathbf{m}_i = \sum_{j \neq i} \mathbf{m}_{ij}$

$\mathbf{h}_i^{l + 1} = \phi_h(\mathbf{h}_i^l, \mathbf{m}_i)$

# GMN

## Initial state:
* Node_feat $\mathbf{h}^0$ (N, in_dim), position $\mathbf{x}^0$ (N, 3), velocity $\mathbf{v}^0$ (N, 3)

* Compute object-related stuffs ($K$ objects): generalized position $\mathbf{q}_k^0$, velocity $\mathbf{\dot{q}}_k$, angle velocity $\{\mathbf{\dot{\theta}}_{ki}\}_{i \in O_k}$ 

## GMNLayer:

### Invariant message 
* Edge function $\phi_e$: Linear -> Activation function -> Linear
* $m_{ij}$ = $\phi_e(\mathbf{h_i, h_j, ||x_i - x_j||^2}, a_{ij})$ (edge attribute optional)

### Compute force $\mathbf{f}$
(Just like EGNN, but for force instead of coordinate vectors "record the force as an intermediate variable that will
contribute to the inference of the generalized acceleration in the next step."):
* Coordinate function $\phi_x$: Linear -> Activation -> Linear

* $\mathbf{f}_i^{l+1} = \mathbf{x}_i^{l} + \sum_{j \in N(i)} (\mathbf{x}_i - \mathbf{x}_j) \phi_x(m_{ij})$

### Update node features $\mathbf{h}$
* Node function $\phi_h$: Linear -> Activation -> Linear -> Residual optional
* Aggregrate invariant message: $m_i = \sum_{j \in N(i)} m_{ij}$
* $\mathbf{h}_i^{l + 1}$ = $\phi_h(\mathbf{h}_i, m_i)$

### Object update

**Equivariant functions: $\phi_2$** 
* Sticks MLP: Linear -> act -> Linear -> act -> Linear 
* Hinges MLP: Linear -> act -> Linear -> act -> Linear

**Acceleration $\mathbf{\ddot{q}}$**:

* Sticks: $\sum_{i \in O_k} \phi_2(\mathbf{f}_i)$
* Hinges: $\sum_{i \in O_k} \phi_2(\mathbf{f}_i, \mathbf{x}_{ki}, \mathbf{v}_{ki})$

**Angle acceleration $\mathbf{\ddot{\theta}}_{ki}$**:

* Sticks: 

$$\frac{\sum_{i \in O_k}\mathbf{x}_{ki} \times \mathbf{f}_i}{\sum_{i \in O_k}||\mathbf{x}_{ki}||^2}$$

* Hinges:

$$ \frac{\mathbf{x}_{ki} \times (\mathbf{f}_i - \mathbf{\ddot{q}_k})}{||\mathbf{x}_{ki}||^2} $$

**Update generalized position and velocity $\mathbf{q, \dot{q}}$**:

$$\mathbf{\dot{q}}_k^{l + 1} = \psi(\sum_{i \in O_k} \mathbf{h}_i^{l}) \mathbf{\dot{q}}_k^{l} + \mathbf{\ddot{q}}_k^{l}$$

$$\mathbf{q}_k^{l + 1} = \mathbf{q}_k^{l} + \mathbf{\dot{q}}_k^{l + 1}$$

$$\mathbf{\dot{\theta}}_{ki}^{l + 1} = \psi '(\sum_{i \in O_k} \mathbf{h}_i^{l}) \mathbf{\dot{\theta}}_{ki}^{l} + \mathbf{\ddot{\theta}}_{ki}^{l}$$

**Update $\mathbf{x, v}$**:

$$\mathbf{x}_i^{l + 1} = \mathbf{q}_k^{l + 1} + \text{rot}(\mathbf{\dot{\theta}}_{ki}^{l + 1}) \mathbf{x}_{ki}^{l}$$

$$\mathbf{v}_i^{l + 1} = \mathbf{\dot{q}}_k^{l + 1} + \mathbf{\dot{\theta}}_{ki}^{l + 1} \times \mathbf{x}_{ki}^{l + 1}$$





