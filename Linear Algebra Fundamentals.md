Preparatory notes covering the basic linear algebra needed before tackling the [[Laplacian Matrix]] and its [[Laplacian Spectrum|spectrum]] as positional encoding in transformers — building the foundations before the real thesis work.

## Vectors, norm, and unit vectors (versors)

The norm (length) of a vector $v = \langle x, y \rangle$ is:

$$\|v\| = \sqrt{x^2 + y^2}$$

A vector is a **versor** (unit vector) if $\|v\| = 1$.

Examples:
- $V = \langle 0, 1 \rangle$, $\|V\| = \sqrt{0^2+1^2} = 1$ → versor
- $W = \langle \tfrac{\sqrt2}{2}, \tfrac{\sqrt2}{2}\rangle$, $\|W\| = \sqrt{\tfrac12+\tfrac12} = 1$ → versor
- $U = \langle -\tfrac12, \tfrac12\rangle$, $\|U\| = \sqrt{\tfrac14+\tfrac14} = \sqrt{\tfrac12}$ → **not** a versor

To turn any nonzero vector into a versor, divide it by its own norm:

$$\hat U = \frac{U}{\|U\|}$$

## Interpolating between vectors: LERP, NLERP, SLERP

Given two versors $V$ and $W$, linear interpolation (**LERP**) with parameter $t \in [0,1]$ is:

$$\text{mix}(V, W, t) = (1-t)V + tW$$

At $t=0$ the result is entirely $A$ (i.e. $V$), at $t=1$ entirely $B$ (i.e. $W$).

Question: **is the interpolated vector still a versor?** In general, no — worked example with $V = \langle 0,1\rangle$, $W = \langle \tfrac{\sqrt2}{2}, \tfrac{\sqrt2}{2}\rangle$, $t = \tfrac12$:

Since $\|aV\| = |a| \cdot \|V\|$:

$$\tfrac12\|V\| = \tfrac12, \qquad \tfrac12\|W\| = \tfrac12$$

$$\text{mix}\!\left(V,W,\tfrac12\right) = \langle 0, \tfrac12\rangle + \langle \tfrac{\sqrt2}{4}, \tfrac{\sqrt2}{4}\rangle = \left\langle \tfrac{\sqrt2}{4},\ \tfrac{2+\sqrt2}{4}\right\rangle$$

$$\left\|\text{mix}(\dots)\right\| = \sqrt{\tfrac{2}{16} + \tfrac{6+4\sqrt2}{16}} = \sqrt{\tfrac{2+2\sqrt2}{2}} \neq 1$$

So **LERP does not preserve unit norm**: even interpolating between two versors can land off the unit circle. To keep the result a versor, we need either:

- **NLERP** (normalized LERP) — apply LERP, then renormalize the result by dividing by its own norm, or
- **SLERP** (spherical LERP) — interpolate along the arc of the unit circle/sphere instead of along the straight chord between the two endpoints.

## General vector algebra

For $v = (x_v, y_v)$ and $w = (x_w, y_w)$:

$$\|v\| = \sqrt{x_v^2 + y_v^2}$$

$$v \cdot w = x_v x_w + y_v y_w$$

$$v \cdot v = x_v^2 + y_v^2 = \|v\|^2$$

**Cross product** (3D):

$$v \times w = \begin{pmatrix} v_y w_z - v_z w_y \\ v_z w_x - v_x w_z \\ v_x w_y - v_y w_x \end{pmatrix}$$

## Complex numbers and modulus

For $p = a + ib$, the conjugate is $\bar p = a - ib$, and:

$$\|p\|^2 = p \cdot \bar p = (a+ib)(a-ib) = a^2 - aib + aib - (ib)^2 = a^2+b^2$$

## Linear transformations as a change of basis

A linear transformation is expressed in terms of the two basis vectors $\hat\imath$ and $\hat\jmath$ of the space: applying a transformation $L$ to a vector $v = [x,y]$ can be seen as a sum of two scalar multiples, where the scalars are the vector's own components and the vectors being scaled are the *transformed* basis vectors $\hat\imath'$ and $\hat\jmath'$.

Suppose $\hat\imath' = \begin{bmatrix}-1\\2\end{bmatrix}$ and $\hat\jmath' = \begin{bmatrix}2\\1\end{bmatrix}$. Applying the transformation to $v$:

$$x\,\hat\imath' + y\,\hat\jmath' = x\begin{bmatrix}-1\\2\end{bmatrix} + y\begin{bmatrix}2\\1\end{bmatrix} = \begin{bmatrix}-x+2y\\2x+y\end{bmatrix}$$

This transformation can therefore be written as a matrix:

$$\begin{bmatrix}-1 & 2\\ 2 & 1\end{bmatrix}$$

where the first column is where $\hat\imath$ lands in the new reference frame, and the second column is where $\hat\jmath$ lands.

**Applying a transformation is nothing more than a change of the reference system.** We change the basis, and every other vector in the span is generated from it accordingly:

$$\begin{bmatrix}-1&2\\2&1\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}$$

## Determinant

The determinant of a matrix tells us by how much the area delimited by the basis vectors gets scaled.

Example: the transformation $\begin{bmatrix}2&0\\0&3\end{bmatrix}$ scales the unit square (area $1$) into a $2\times3$ rectangle — area $= 2\cdot 3 = 6 = \det$.

- $\det = 0$ when the transformation makes the basis vectors parallel, because we no longer have any area at all (if the starting vector space were 2D, in general it means we have collapsed one or more dimension into one, causing the area to be zero in the starting space).
- The determinant can be **negative**: this happens when the transformation reverses (flips) the orientation of the reference system.
- In 3D, the determinant tells us by how much the *volume* is scaled, and it is negative if the transformation produces a left-handed (non-right-handed) orientation.

> **Mind-blowing idea:** the identity transformation $\begin{bmatrix}1&0\\0&1\end{bmatrix}$ is a diagonal matrix, and this makes perfect sense, because its columns are *exactly* the basis vectors of the space itself: $\hat\imath = \begin{bmatrix}1\\0\end{bmatrix}$ and $\hat\jmath=\begin{bmatrix}0\\1\end{bmatrix}$.

## Rank

The rank of a matrix tells us how many dimensions the image of the transformation occupies.

For a $3\times3$ matrix the rank is between $1$ and $3$: a rank lower than $3$ means some column vector of the matrix is linearly dependent on the others.

This is why a linear system whose matrix does not have maximal rank can have infinitely many solutions — more vectors collapse onto the same point, so the final image loses a dimension.

**NB:** rank lower than maximal $\iff \det = 0$.

When the determinant is $0$ the matrix is **not invertible** — which makes sense, since more than one vector now falls onto the same point, so we can no longer tell where a given output vector originally came from before the transformation.

## Eigenvectors and eigenvalues

Since matrices are just linear transformations (a change of basis → a change of the reference system), the concept of eigenvectors and eigenvalues is simple: an **eigenvector** is a vector that, once the transformation is applied, is only *scaled* by some factor — it never changes direction.

Geometrically, this means its span doesn't change: every vector belonging to the span of an eigenvector is itself an eigenvector, and all eigenvectors in that span are scaled by the same factor, called the **eigenvalue**. In mathematical terms:

$$M\vec v = \lambda \vec v$$

Rotations, being rigid transformations, must have eigenvalue $1$ along the rotation axis — otherwise the movement would not be rigid.

### Finding eigenvalues

$$M\vec v = \lambda \vec v \;\Rightarrow\; M\vec v = \lambda I \vec v \;\Rightarrow\; M\vec v - \lambda I \vec v = \vec 0 \;\Rightarrow\; \vec v\,(M - \lambda I) = \vec 0$$

Either $\vec v$ is the null vector (the trivial case), or:

$$\det(M - \lambda I) = 0, \qquad \lambda I = \begin{bmatrix}\lambda& &0\\ &\ddots& \\0& &\lambda\end{bmatrix}$$

We need to find the values of $\lambda$ that make the determinant of $(M-\lambda I)$ become $0$.

## Linear applications (homomorphisms)

These are functions that behave well with respect to vector addition and scalar multiplication: functions from one vector space to another such that, given two vectors, the image of their sum equals the sum of their images, and the image of a scalar multiple of a vector equals that same multiple of the image.

**NB:** the null vector is never "free" of the spaces involved — a necessary condition for $f: V \to W$ to be a linear map is that $f(\vec 0_V) = \vec 0_W$.

A **linear operator** is a particular kind of linear application where the domain and the image are the *same* vector space. These are called **endomorphisms**.

## Diagonalizable matrices

If the basis of an endomorphism is made up solely of eigenvectors, the endomorphism is called **simple**, and it is defined by a **diagonalizable** matrix, with all eigenvalues real. (Note: multiple bases made of orthonormal eigenvectors can exist for the space of a given endomorphism.)

**NB:** symmetric square matrices — such as, for example, the **Laplacian matrix** — are an interesting case of diagonalizable matrices.

## Eigenspace

An eigenspace is the vector subspace associated to all the eigenvector of a specific eigenvalue and the zero vector.

 It is closed under vector addition and scalar multiplication, making it a valid subspace of the original vector space.

Two different eigenspaces of the same matrix have disjoint union of their vector sets (unless the zero vector). Imagine a vector $v \neq 0$, if it belongs to two distinct eigenspaces of $\lambda_1$ and $\lambda_2$ then we would have $Av =  \lambda v_1$ and $Av =  \lambda v_2$. Meaning that the only possible case is $\lambda_1 = \lambda_2$.

The dimension of the eigenspace is called the geometric multiplicity of the eigenvalue, representing the number of linearly independent eigenvectors associated with $\lambda$.

A matrix is diagonalizable if and only if the sum of the dimensions of its eigenspaces equals the dimension of the vector space (i.e., there exists a basis of eigenvectors).

 Eigenspaces are guaranteed to be orthogonal only when the matrix is **symmetric**

*Continued in [[Introduction to Spectral Graph Theory]].*
