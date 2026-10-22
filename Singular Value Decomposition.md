Given a matrix $A \in $\mathbb{R}^{n \times m}$, the SVD  is a generalization of the eigendecomposition $A = V\mathcal{A}V^{-1}$ to a transformation from a space to another
$$
A = U\Sigma V^T
$$
where $U$ and $V^{-1}$ are two orthogonal matrices $\in \mathbb{R^{n \times n}}$ the component $u_i$ of $U$ are called left singular vectors, the ones of $V^{-1}$ are called right singular vectors respectively. Both those vectors sets are orthonormal internally. 
$\Sigma$ is a square diagonal matrix, where its values $\sigma_i = \Sigma_{ii}$ are so that $\sigma_1 \ge \sigma_2 \dots \ge \sigma_m$. This values are called singular values.
Geometrically, we can see $A$ as a transformation $V^T$ first done in the input space $\mathbb{R}^n$, then scaling every axes by a factor $\sigma_i$, and apply a last transformation $U$ in the output space $\mathbb{R}^m$