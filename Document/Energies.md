# Energies


## Contact

For VF/EE contact, we have barycentric weight $w \in R^4$, area weighted stiffness $k = \kappa a$, direction $n$ of shortest distance $d$ (With positions $x = [x_1^T, x_2^T, x_3^T, x_4^T]^T$).

$$d = || t || = || \sum_i^4 w_i x_i || $$

- For VF : $w_1 = 1, (w_1 + w_2 + w_3) = -1$
- For EE : $(w_1 + w_2) = 1, (w_2 + w_3) = -1$

Considering $w$ is constant, so we can have:

$$
\frac{\partial t}{\partial x} = 
\begin{bmatrix}
w_1 I_3, w_2 I_3, w_3 I_3, w_4 I_3
\end{bmatrix} \in R^{3 \times 12}  \quad \text{and} \quad
\frac{\partial^2 t}{\partial x^2} = 0
$$

> So this is Gauss-Newton, which result in problem in some configurations, but this is enough for most cases.

We can make a **quadratic** formulation of energy $E = \frac{1}{2} k (d-\hat{d})^2$ or a **log-barrier** formulation of energy $E = -(d - \hat{d})^2 \ln (\frac{d}{\hat{d}})$.

Then we have:

$$
\frac{\partial E}{\partial x} = \frac{\partial E}{\partial d}  \frac{\partial d}{\partial t} \frac{\partial t}{\partial x} = \frac{\partial E}{\partial d} \frac{t^T}{d} \frac{\partial t}{\partial x}
 = \frac{\partial E}{\partial d} n^T \frac{\partial t}{\partial x}
$$

$$
\frac{\partial^2 E}{\partial x^2} = \frac{\partial^2 E}{\partial d^2} (n^T \frac{\partial t}{\partial x}) (n^T \frac{\partial t}{\partial x})^T
$$

### Contact Implentation

We set $k_1 = \partial E / \partial d$:
- For quadratic formulation: $k_1 = k (d-\hat{d})$
- For log-barrier formulation: $k_1 = \text{Refers to IPC toolkit lol}$

And set $k_2 = \partial^2 E / \partial d^2$
- For quadratic formulation: $k_2 = k$
- For log-barrier formulation: $k_2 = \text{Refers to IPC toolkit lol}$

For $i$'s vertex in VF/EE pair:

$$ \nabla E_i = k_1 w_i n $$

And:

$$ \nabla E_{ij}^2 = k_2 w_i w_j n n^T $$





## Reduced System of Affine-Body-Dynamics 

A Jacobian matrix $J$ map the relation ship between position $x$ (of vertex) and state $q$ (of body) :

$$ \frac{\partial E}{\partial q} = \frac{\partial E}{\partial x} \frac{\partial x}{\partial q} = \frac{\partial E}{\partial x} J$$

$$ \frac{\partial^2 E}{\partial q_i \partial q_j} 
= (\frac{\partial x}{\partial q_j})^T  \frac{\partial^2 E}{\partial x^2} \frac{\partial x}{\partial q_i} + \cancel{\frac{\partial E}{\partial x} \frac{\partial^2 x}{\partial q_i \partial q_j}} 
= J_j^T \frac{\partial^2 E}{\partial x_i \partial x_j} J_i$$

We simplify the symbolic as: $\textcolor{red}{g} = \nabla E_{x_i}$, and $\textcolor{red}{H} = \nabla^2 E_{x_{i, j}}$:

$$\nabla E_{q_i} = J^T \nabla E_{x_i} = J^T \textcolor{red}{g}$$

$$\nabla E_{q_i, q_j}^2 = J_i^T \nabla^2 E_{x_i, x_j} J_j = J_i^T \textcolor{red}{H} J_j$$

For **Soft Body** (cloth, soft-body, rods...), we use full-space simulation:

$$J_s = I_3$$

For **Rigid (Affine) Body**, we use reduced-space simulation (Where $\overline{x}$ is the position in **model space**):

$$ J_r = 
\begin{bmatrix}
1 & 0 & 0 & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 & & & & & & \\
0 & 1 & 0 & & & & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 & & & \\
0 & 0 & 1 & & & & & & & \overline{x}_1 & \overline{x}_2 & \overline{x}_3 \\
\end{bmatrix} \in R^{3 \times 12} $$

So we can simplify the calculation. For gradient:

$$\nabla E_{q_i} = J^T \textcolor{red}{g} = 
\begin{bmatrix}
{g}
\\ {g}_{0} \overline{x} 
\\ {g}_{1} \overline{x} 
\\ {g}_{2} \overline{x} 
\end{bmatrix} \in R^{12}$$ 

Where $g_{i}$ is the *i*'s element in $g$.

For hessian, $\nabla E_{q_i, q_j}^2 = J_i^T \nabla^2 E J_j$, we have 3 cases:

---

(1) **Rigid-Soft**, $J_i = J_r, J_j = I_3$ :

$$
\nabla^2 E_{q_i, q_j} = J_i^T \textcolor{red}{H} J_j = 
\begin{bmatrix}
H
\\ \overline{x}_i H_{1,:}
\\ \overline{x}_i H_{2,:}
\\ \overline{x}_i H_{3,:}
\end{bmatrix} \in R^{12 \times 3}
$$

Where $H_{i,:}$ is the *i*'th row in $H$.

---

(2) **Soft-Rigid**, $J_i = I_3 , J_j = J_r$ :

$$
\nabla^2 E_{q_i, q_j} = J_i^T \textcolor{red}{H} J_j = 
\begin{bmatrix}
H
& H_{:,1} \overline{x}_j^T
& H_{:,2} \overline{x}_j^T
& H_{:,3} \overline{x}_j^T
\end{bmatrix} \in R^{3 \times 12}
$$

Where $H_{:,j}$ is the *j*'th column in $H$.

---

(3) **Rigid-Rigid** (Only between different bodies), $J_i = J_r , J_j = J_r$: 

$$
\nabla^2 E_{q_i, q_j} = J_i^T \textcolor{red}{H} J_j = 
\begin{bmatrix}
H 
& H_{:,1} \textcolor{red}{\overline{x}_j}^T    
& H_{:,2} \textcolor{red}{\overline{x}_j}^T    
& H_{:,3} \textcolor{red}{\overline{x}_j}^T 
\\
\textcolor{green}{\overline{x}_i} H_{1,:}
& H_{1,1} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{1,2} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{1,3} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T 
\\
\textcolor{green}{\overline{x}_i} H_{2,:}  
& H_{2,1} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{2,2} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{2,3} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T 
\\
\textcolor{green}{\overline{x}_i} H_{3,:}
& H_{3,1} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{3,2} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T    
& H_{3,3} \textcolor{green}{\overline{x}_i} \textcolor{red}{\overline{x}_j}^T  
\end{bmatrix} \in R^{12 \times 12}
$$

---

In summury, convert from hessian $\nabla^2 E_{x_i, x_j}$ to $\nabla^2 E_{q_i, q_j}$:

Condition             | Condition 2                     | Expression
---------             | ----------                      | ----------
$\text{row} = 1,     \text{column} = 1$    | Or if $i$ is soft, $j$ is soft  | $H$
$\text{row} = 1,     \text{column} \neq 1$ | Or if $i$ is soft, $j$ is rigid | $H_{:,c} \overline{x}_j^T$
$\text{row} \neq 1,  \text{column} = 1$    | Or if $i$ is rigid, $j$ is soft | $\overline{x}_i H_{r,:}$
$\text{row} \neq 1,  \text{column} \neq 1$ | Only if $i$ is rigid, $j$ is rigid | $H_{r,c} \overline{x}_i \overline{x}_j^T$

Where $r$ and $c$ in $H_{r,c}$ refers to relative row index and column index in faces ($c = \text{column} - 1, r = \text{row} - 1$)

---

### Contact of Affine-Body

If VF/EE pair contains vertex from **rigid-body**, we need to convert the contribution of gradient and hessian of collision pairs, from full-space into reduced-space:

(1) For VF pair:

For vert $i \in \text{V}$ : If $i$ is from Rigid body, we just need to calculate $\nabla E_{q_i}$ and $\nabla^2 E_{q_i, q_i}$ according to the formulation above, and add to the linear system.

For vert $i \in \text{F}$ : If $i$ is from Rigid body, we need to summurize the contribution of three vertices in face. We set $X$ as the **weighted model position** according to the collision barycentric $w$ :

$$X = \sum_{i \in \text{face}}^3 w_i \overline{x}_i$$

So we get:

$$ 
\begin{aligned}
\sum_{i \in \text{face}}^3 \nabla E_{q_i} 
&= \sum_{i} J_i^T k_1 w_i n \\
&= \sum_{i}
\begin{bmatrix}
k_1 w_i n
\\ k_1 w_i {n}_{0} \overline{x}_i 
\\ k_1 w_i {n}_{1} \overline{x}_i 
\\ k_1 w_i {n}_{2} \overline{x}_i 
\end{bmatrix}
= k_1 
\begin{bmatrix}
   w_i n
\\ {n}_{0} \sum_{i} w_i \overline{x}_i 
\\ {n}_{1} \sum_{i} w_i \overline{x}_i 
\\ {n}_{2} \sum_{i} w_i \overline{x}_i 
\end{bmatrix}
= k_1 
\begin{bmatrix}
   w_i n
\\ {n}_{0} X 
\\ {n}_{1} X 
\\ {n}_{2} X 
\end{bmatrix} 
\end{aligned}
$$

And:

$$ 
\begin{aligned}
\sum_{i,j \in \text{face}}^3 \nabla^2 E_{q_i, q_j} 
&= \sum_{i} \sum_{j} J_i^T k_2 w_i w_j n n^T J_j \\
&= 
\begin{cases}
\sum_{i} \sum_{j} H & \text{row} = 1, \text{column} = 1        
\\
\sum_{i} \sum_{j} H_{:,c} \overline{x}_j^T & \text{row} = 1, \text{column} \neq 1     
\\
\sum_{i} \sum_{j} \overline{x}_i H_{r,:} & \text{row} \neq 1, \text{column} = 1     
\\
\sum_{i} \sum_{j} H_{i,j} \overline{x}_i \overline{x}_j^T & \text{row} \neq 1,  \text{column} \neq 1  
\\
\end{cases} \\
&= 
\begin{cases}
k_2 \sum_{i} \sum_{j} w_i w_j n n^T & \text{row} = 1, \text{column} = 1        
\\
k_2 \sum_{i} \sum_{j} w_i w_j (n n^T)_{:,c} \overline{x}_{j}^T & \text{row} = 1, \text{column} \neq 1     
\\
k_2 \sum_{i} \sum_{j} w_i w_j \overline{x}_i (n n^T)_{r,:} & \text{row} \neq 1, \text{column} = 1     
\\
k_2 \sum_{i} \sum_{j} w_i w_j (n n^T)_{r,c} \overline{x}_i \overline{x}_j^T & \text{row} \neq 1,  \text{column} \neq 1  
\\
\end{cases} \\
&= 
\begin{cases}
k_2 \sum_{i} \sum_{j} w_i w_j n n^T & \text{row} = 1, \text{column} = 1     
\\
k_2 \sum_{i} \sum_{j} w_i w_j n_c n \overline{x}_j^T & \text{row} = 1, \text{column} \neq 1     
\\
k_2 \sum_{i} \sum_{j} w_i w_j \overline{x}_i n_r n^T & \text{row} \neq 1, \text{column} = 1     
\\
k_2 \sum_{i} \sum_{j} w_i w_j n_i n_j \overline{x}_i \overline{x}_j^T & \text{row} \neq 1,  \text{column} \neq 1  
\\
\end{cases} \\
&= 
\begin{cases}
k_2 (\sum_{i} \sum_{j} w_i w_j) n n^T & \text{row} = 1, \text{column} = 1    
\\
k_2 n_c n \sum_{i} w_i (\sum_{j} w_j \overline{x}_j^T)  & \text{row} = 1, \text{column} \neq 1     
\\
k_2 n_r \sum_{j} w_j (\sum_{i} w_i \overline{x}_i) n^T & \text{row} \neq 1, \text{column} = 1     
\\
k_2 n_r n_c \sum_{i} w_i \overline{x}_i  (\sum_{j} w_j \overline{x}_j^T) & \text{row} \neq 1,  \text{column} \neq 1  
\\
\end{cases} \\
&= 
\begin{cases}
k_2 n n^T & \text{row} = 1, \text{column} = 1    
\\
k_2 n_c n (\sum_{i} w_i) X^T  & \text{row} = 1, \text{column} \neq 1     
\\
k_2 n_r (\sum_{j} w_j) X  n^T & \text{row} \neq 1, \text{column} = 1     
\\
k_2 n_r n_c (\sum_{i} w_i \overline{x}_i) X^T & \text{row} \neq 1,  \text{column} \neq 1  
\\
\end{cases} \\
&= 
\begin{cases}
k_2 n n^T & \text{row} = 1, \text{column} = 1    
\\
k_2 n_c n X^T  & \text{row} = 1, \text{column} \neq 1     
\\
k_2 n_r X n^T & \text{row} \neq 1, \text{column} = 1     
\\
k_2 n_r n_c X X^T & \text{row} \neq 1,  \text{column} \neq 1  
\\
\end{cases} \\
\end{aligned}
$$
