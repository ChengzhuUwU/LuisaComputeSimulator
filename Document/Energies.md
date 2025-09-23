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

We can use different type of contact energy, include: 

- A **quadratic** formulation of energy $E = \frac{1}{2} k (d-\hat{d})^2$
- A **log-barrier** formulation of energy $E = -(d - \hat{d})^2 \ln (\frac{d}{\hat{d}})$ 
   - Or use Codimentional-IPC enhanced energy, which modeling the thickness $\epsilon$

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
- For log-barrier formulation: $k_1 = (\hat{d} - d)(2 \ln (\frac{d}{\hat{d}}) - \frac{\hat{d}}{d} + 1 )$

And set $k_2 = \partial^2 E / \partial d^2$
- For quadratic formulation: $k_2 = k$
- For log-barrier formulation: $k_2 = (\frac{\hat{d}}{d} + 2)\frac{\hat{d}}{d} - 2\ln (\frac{d}{\hat d}) -3$

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

So we can simplify the calculation. 

### For gradient

$$\nabla E_{q_i} = J^T \textcolor{red}{g} = 
\begin{bmatrix}
{g}
\\ {g}_{0} \overline{x} 
\\ {g}_{1} \overline{x} 
\\ {g}_{2} \overline{x} 
\end{bmatrix} \in R^{12}$$ 

Where $g_{i}$ is the *i*'s element in $g$.

### For hessian

For hessian $\nabla E_{q_i, q_j}^2 = J_i^T \nabla^2 E J_j$, we have 4 cases:

> $i,j$ are vertices from VF/EE Pair

---

(1) **Soft Vert - Soft Vert**, $J_i = I_3, J_j = I_3$ :

$$
\nabla^2 E_{q_i, q_j} = J_i^T \textcolor{red}{H} J_j = I_3^T \textcolor{red}{H} I_3
= H \in R^{3 \times 3}
$$

This is actullly what we do in full-space simulation.

---

(2) **Soft Vert - Rigid Vert**, $J_i = I_3 , J_j = J_r$ :

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

(3) **Rigid Vert - Soft Vert**, $J_i = J_r, J_j = I_3$ :

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

(4) **Rigid Vert - Rigid Vert**, $J_i = J_r , J_j = J_r$: 

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
Soft Vert - Soft Vert  | | $H$
Soft Vert - Rigid Vert  | | $\begin{bmatrix}H & H_{:,1} \overline{x}_j^T & H_{:,2} \overline{x}_j^T & H_{:,3} \overline{x}_j^T \end{bmatrix}$
Rigid Vert - Soft Vert  | | $\begin{bmatrix}H \\ \overline{x}_i H_{1,:} \\ \overline{x}_i H_{2,:} \\ \overline{x}_i H_{3,:}\end{bmatrix}$
Rigid Vert - Rigid Vert | $\text{row} = 1,     \text{column} = 1$    | $H$
Rigid Vert - Rigid Vert | $\text{row} = 1,     \text{column} \neq 1$ | $H_{:,c} \overline{x}_j^T$
Rigid Vert - Rigid Vert | $\text{row} \neq 1,  \text{column} = 1$    | $\overline{x}_i H_{r,:}$
Rigid Vert - Rigid Vert | $\text{row} \neq 1,  \text{column} \neq 1$ | $H_{r,c} \overline{x}_i \overline{x}_j^T$

Where $r$ and $c$ in $H_{r,c}$ refers to relative **blocked** row index and column index in faces ($c = \text{column} - 1, r = \text{row} - 1$)

> Note that:
>
> - "Soft-Rigid" is equal to the first blocked row    in "Rigid-Rigid" hessian
> 
> - "Rigid-Soft" is equal to the first blocked column in "Rigid-Rigid" hessian

---

## Contact of Affine-Body

If VF/EE pair contains vertex from **rigid-body**, we need to convert their contribution (gradient $\nabla E \in R^{12}$ and hessian $\nabla^2 E \in R^{12\times 12}$) from full-space into reduced-space. 

### For gradient

---

(1) For vert in $\text{V}$ (If $\text{V}$ is from Rigid Body)

We just need to calculate $\nabla E_{q_i}$ according to the formulation above.

$$\nabla E_{q_i} 
= J^T \textcolor{red}{g} 
= J^T k_1 w_1 n
= k_1 \begin{bmatrix}
n
\\ n_{0} \overline{x} 
\\ n_{1} \overline{x} 
\\ n_{2} \overline{x} 
\end{bmatrix} \in R^{12}$$ 

> In VF, weight of vert $w_1 = 1$

---

(2) For vertices in $\text{F}$ (If $\text{F}$ is from Rigid Body)

For vertices $i \in \text{F}$ : We need to summurize the contribution of three vertices in face. 

We set $X$ as the **weighted model position** according to the barycentric coordinate :

$$X = \sum_{i \in \text{face}}^3 |w_i| \overline{x}_i , \quad \text{s.t.} \sum_{i \in \text{face}}^3 w_i = -1$$

> Note that $X = -\sum_{i \in \text{face}}^3 w_i \overline{x}_i $

So we can get:

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
   \sum_{i} w_i n
\\ {n}_{0} \sum_{i} w_i \overline{x}_i 
\\ {n}_{1} \sum_{i} w_i \overline{x}_i 
\\ {n}_{2} \sum_{i} w_i \overline{x}_i 
\end{bmatrix}
= -k_1 
\begin{bmatrix}
   n
\\ {n}_{0} X 
\\ {n}_{1} X 
\\ {n}_{2} X 
\end{bmatrix} 
\end{aligned}
$$

---

Sow we can promote the regularity: both VF (vertex–face) and EE (edge–edge) interactions can be expressed in a unified form.

Each interaction is decomposed into two parts: 

- VF -> vertex $\text{V}$ and face $\text{F}$
- EE -> edge $\text{E}_1$ and edge $\text{E}_2$

For $\text{element} \in [V,F,E_1,E_2]$, we can set: $$ X = \sum_{i \in \text{element}} |w_i| \overline{x}_i $$

The contribution of each part to the rigid body is written as:

$$ 
\sum_{i \in \text{element}} \nabla E_{q_i} = (\sum_{i} w_i )k_1 
\begin{bmatrix}
   n
\\ {n}_{0} X 
\\ {n}_{1} X 
\\ {n}_{2} X 
\end{bmatrix}
$$

Where $\sum_{i} w_i$ is 1 (For $\text{V}, \text{E}_1$) or -1 (For $\text{F}, \text{E}_2$).

### For hessian

---

(1) For vert in $\text{V}$ (If $\text{V}$ is from Rigid Body)

We just need to calculate $\nabla^2 E_{q_i, q_i}$ according to the formulation above, and add to the global hessian matrix. (Calculted as hessian between **Rigid Vert - Rigid Vert**)

$$
\begin{aligned}
\nabla^2 E_{q_i, q_i} 
&= J_i^T k_2 n n^T J_i \\
&= k_2 \begin{bmatrix}
nn^T
& n_1 n \overline{x}_i^T
& n_2 n \overline{x}_i^T
& n_3 n \overline{x}_i^T
\\
n_1 X n^T
& n_1 n_1 \overline{x}_i \overline{x}_i^T    
& n_1 n_2 \overline{x}_i \overline{x}_i^T    
& n_1 n_3 \overline{x}_i \overline{x}_i^T 
\\
n_2 X n^T  
& n_2 n_1 \overline{x}_i \overline{x}_i^T    
& n_2 n_2 \overline{x}_i \overline{x}_i^T    
& n_2 n_3 \overline{x}_i \overline{x}_i^T 
\\
n_3 X n^T
& n_3 n_1 \overline{x}_i \overline{x}_i^T    
& n_3 n_2 \overline{x}_i \overline{x}_i^T    
& n_3 n_3 \overline{x}_i \overline{x}_i^T  
\end{bmatrix}  \in R^{12 \times 12} \\
&= k2 
\begin{bmatrix}
n \\ n_1 \overline{x}_i \\ n_2 \overline{x}_i \\ n_3 \overline{x}_i
\end{bmatrix}
\begin{bmatrix}
n^T & n_1 \overline{x}_i^T & n_2 \overline{x}_i^T & n_3 \overline{x}_i^T
\end{bmatrix} \\
&= k2 
\begin{bmatrix}
n \\ n_1 \overline{x}_i \\ n_2 \overline{x}_i \\ n_3 \overline{x}_i
\end{bmatrix}
\begin{bmatrix}
n \\ n_1 \overline{x}_i \\ n_2 \overline{x}_i \\ n_3 \overline{x}_i
\end{bmatrix}^T
\end{aligned}
$$



---

(2) For vert between $\text{F}$ (If $\text{F}$ is from Rigid Body)

We need to summurize the contribution of three vertices in face (9 blocked 3x3 matrix): (Also calculted as hessian between **Rigid Vert - Rigid Vert**)

$$ 
\begin{aligned}
\sum_{i \in \text{face}}^3 \sum_{j \in \text{face}}^3 \nabla^2 E_{q_i, q_j} 
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

In summary:

$$
\begin{aligned}
\sum_{i \in \text{face}}^3 \sum_{j \in \text{face}}^3 \nabla^2 E_{q_i, q_j}  
&= k_2 \begin{bmatrix}
nn^T
& n_1 n X^T
& n_2 n X^T
& n_3 n X^T
\\
n_1 X n^T
& n_1 n_1 X X^T    
& n_1 n_2 X X^T    
& n_1 n_3 X X^T 
\\
n_2 X n^T  
& n_2 n_1 X X^T    
& n_2 n_2 X X^T    
& n_2 n_3 X X^T 
\\
n_3 X n^T
& n_3 n_1 X X^T    
& n_3 n_2 X X^T    
& n_3 n_3 X X^T  
\end{bmatrix} \in R^{12 \times 12} \\
&= k2 
\begin{bmatrix}
n \\ n_1 X \\ n_2 X \\ n_3 X
\end{bmatrix}
\begin{bmatrix}
n^T & n_1 X^T & n_2 X^T & n_3 X^T
\end{bmatrix} \\
&= k2 
\begin{bmatrix}
n \\ n_1 X \\ n_2 X \\ n_3 X
\end{bmatrix}
\begin{bmatrix}
n \\ n_1 X \\ n_2 X \\ n_3 X
\end{bmatrix}^T
\end{aligned}
$$

Same as the hessian of vertex $\nabla^2 E_{q_1,q_1}$.

---

Sow we can promote the regularity:

For $[\text{element}_1, \text{element}_2] \in [[V,V], [F,F], [E_1,E_1], [E_2,E_2], [V,F], [E_1,E_2]]$, We can set: 

$$ X_1 = \sum_{i \in \text{element}_1} |w_i| \overline{x}_i$$

$$ X_2 = \sum_{i \in \text{element}_2} |w_i| \overline{x}_i$$

If $[\text{element}_1, \text{element}_2]$ are both from rigid body, We can summurize the contribution of each part:

- (1) **Rigid - Rigid**:

$$
\begin{aligned}
\sum_{i \in \text{element}_1} \sum_{j \in \text{element}_2}  \nabla^2 E_{q_i, q_j}  
&= k_2 (\sum_{i} w_i )(\sum_{j} w_j) \begin{bmatrix}
nn^T
& n_1 n X_2^T
& n_2 n X_2^T
& n_3 n X_2^T
\\
n_1 X_1 n^T
& n_1 n_1 X_1 X_2^T    
& n_1 n_2 X_1 X_2^T    
& n_1 n_3 X_1 X_2^T 
\\
n_2 X_1 n^T  
& n_2 n_1 X_1 X_2^T    
& n_2 n_2 X_1 X_2^T    
& n_2 n_3 X_1 X_2^T 
\\
n_3 X_1 n^T
& n_3 n_1 X_1 X_2^T    
& n_3 n_2 X_1 X_2^T    
& n_3 n_3 X_1 X_2^T  
\end{bmatrix} \in R^{12 \times 12} \\
&= k_2 (\sum_{i} w_i )(\sum_{j} w_j)
\begin{bmatrix}
n \\ n_1 X_1 \\ n_2 X_1 \\ n_3 X_1
\end{bmatrix}
\begin{bmatrix}
n \\ n_1 X_2 \\ n_2 X_2 \\ n_3 X_2
\end{bmatrix}^T
\end{aligned}
$$

Also, $\sum_{i} w_i$ is 1 (For $\text{V}, \text{E}_1$) or -1 (For $\text{F}, \text{E}_2$).

If one of the element in $[\text{element}_1, \text{element}_2]$ is from soft body and another is from rigid body:

This situation can only happen between bodies, so $[\text{element}_1, \text{element}_2] \in [[V,F], [E_1,E_2]]$, thus: 

$$ \sum_{i \in \text{element}_1} w_i = 1, \sum_{j \in \text{element}_2} w_j = -1$$

The situation can formed as the first blocked row or column in the hessian above. 

- (2) **Soft - Rigid**:
$$
\begin{aligned}
&  \sum_{i \in \text{element}_1} \sum_{j \in \text{element}_2}  \nabla^2 E_{q_i, q_j} 
\\ =& 
(\sum_{j \in \text{element}_2} w_j) \begin{bmatrix}
nn^T
& n_1 n X_2^T
& n_2 n X_2^T
& n_3 n X_2^T
\end{bmatrix} \in R^{3 \times 12} \\
=& 
-n
\begin{bmatrix}
n \\ n_1 X_2 \\ n_2 X_2 \\ n_3 X_2
\end{bmatrix}^T
\end{aligned}
$$

- (3) **Rigid - Soft**:
$$
\begin{aligned}
& \sum_{i \in \text{element}_1} \sum_{j \in \text{element}_2} \nabla^2 E_{q_i, q_j} 
\\ =& 
(\sum_{i \in \text{element}_1} w_i) \begin{bmatrix}
H
\\ n_1 X_1 n^T
\\ n_2 X_1 n^T
\\ n_3 X_1 n^T
\end{bmatrix} \in R^{12 \times 3} \\
=& 
-\begin{bmatrix}
n \\ n_1 X_1 \\ n_2 X_1 \\ n_3 X_1
\end{bmatrix}
n^T
\end{aligned}
$$
