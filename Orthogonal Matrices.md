A square matrix $Q \in \mathbb{R}^{d \times d}$ is **orthogonal** if its columns form an orthonormal basis of $\mathbb{R}^d$, meaning they are all of unit norm and mutually perpendicular. The equivalent algebraic condition is:

$$Q^\top Q = Q Q^\top = I$$

that is, the transpose of $Q$ equals its inverse: $Q^{-1} = Q^\top$.

## Geometric meaning

Orthogonal matrices represent **rotations and reflections** in $\mathbb{R}^d$. They preserve:

- **Norms:** $|Qv|_2 = |v|_2$ for all $v \in \mathbb{R}^d$
- **Inner products:** $\langle Qu, Qv \rangle = \langle u, v \rangle$ for all $u, v \in \mathbb{R}^d$
- **Angles:** the angle between any two vectors is unchanged after multiplication by $Q$

## Examples

**Rotation in $\mathbb{R}^2$** by angle $\theta$:

$$Q = \begin{pmatrix} \cos\theta & -\sin\theta \ \sin\theta & \cos\theta \end{pmatrix}$$

**Reflection** across the $x$-axis in $\mathbb{R}^2$:

$$Q = \begin{pmatrix} 1 & 0 \ 0 & -1 \end{pmatrix}$$

**The scalar case** $d = 1$: the only $1 \times 1$ matrices satisfying $q^2 = 1$ are $q = +1$ and $q = -1$, i.e. the identity and the sign flip.

## Determinant

The determinant of an orthogonal matrix is always $\pm 1$:

- $\det(Q) = +1$: pure rotation (special orthogonal group $SO(d)$)
- $\det(Q) = -1$: rotation composed with a reflection

## Relevance for eigenvectors

If $V_i \in \mathbb{R}^{n \times d}$ is an orthonormal basis for an eigenspace, then $V_i Q$ is another orthonormal basis of the same eigenspace for any orthogonal $Q$. This is because orthogonal matrices preserve the orthonormality of columns and do not change the column space. The ambiguity in choosing a basis for an eigenspace of dimension $d$ is therefore parametrized exactly by $O(d)$.