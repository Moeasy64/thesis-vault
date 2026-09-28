The **orthogonal group** $O(d)$ is the set of all [[Orthogonal Matrices]] of dimention $d \times d$ :

$$O(d) = { Q \in \mathbb{R}^{d \times d} \mid Q^\top Q = I }$$

It is called a "group" because it satisfies the algebraic axioms:

- **Closure:** the product of two orthogonal matrices is orthogonal.
- **Identity:** the identity matrix $I$ is in $O(d)$.
- **Inverse:** every $Q \in O(d)$ has inverse $Q^{-1} = Q^\top$, which is also in $O(d)$.
- **Associativity:** matrix multiplication is associative.

Geometrically, $O(d)$ consists of all **rotations and reflections** of $\mathbb{R}^d$ — transformations that preserve distances and angles, but never stretch or shear the space.

## Special case: $O(1)$

A $1 \times 1$ orthogonal matrix is just a scalar $q \in \mathbb{R}$ satisfying $q^2 = 1$, so:

$$O(1) = {-1, +1}$$

This is the smallest non-trivial orthogonal group, and corresponds exactly to the sign flip symmetry of eigenvectors when eigenvalues are distinct.

## Relation to the Stiefel and Grassmann manifolds

$O(d)$ acts on the **Stiefel manifold** $\text{St}(d, n)$ — the set of all orthonormal $d$-frames in $\mathbb{R}^n$ — by right multiplication:

$$V \mapsto VQ, \quad Q \in O(d)$$

The quotient of this action is the **Grassmannian** $\text{Gr}(d, n)$, the space of all $d$-dimensional subspaces of $\mathbb{R}^n$:

$$\text{St}(d, n) / O(d) \cong \text{Gr}(d, n)$$

This means that two bases $V$ and $V'$ of the same subspace are always related by some $Q \in O(d)$, and any $O(d)$-invariant function on $\text{St}(d, n)$ can be viewed as a function on subspaces.