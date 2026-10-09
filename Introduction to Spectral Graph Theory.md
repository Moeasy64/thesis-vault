Follows on from [[Linear Algebra Fundamentals]] — preparatory notes toward using the [[Laplacian Matrix|Laplacian]] and its [[Laplacian Spectrum|spectrum]] as positional encoding in transformers. Extracted from [[Spectral Graph theory Introduction.pdf]] 

**Work in progress — not yet complete.**

## Why these matrices

To study graphs we use three matrices — each with different eigenvectors, and each capable of revealing different information about the graph: completeness, connectivity, regularity, etc.

## Adjacency matrix

Symmetric if the graph is undirected:

$$A_{ij} = \begin{cases}1 & \text{if } v_i \sim v_j\\ 0 & \text{otherwise}\end{cases}$$

## Laplacian matrix

$$L_{ij} = \begin{cases} d_{v_i} & \text{if } v_i = v_j\\ -1 & \text{if } v_i \sim v_j \\ 0 & \text{otherwise}\end{cases}$$

where $d_{v_i}$ is the degree of $v_i$. Obtained as $L = D - A$, where $D$ is the (symmetric, diagonal) degree matrix. See [[Laplacian Matrix]].
N.B: the difference of two symmetrical matrix still gives a symmetrical matrix, so $L$ is still symmetrical.

## Normalized Laplacian

$$\mathcal L_{ij} = \begin{cases} 1 & \text{if } v_i = v_j \\[4pt] -\dfrac{1}{\sqrt{d_{v_i} d_{v_j}}} & \text{if } v_i \sim v_j\\[4pt] 0 & \text{otherwise}\end{cases}$$

It keeps the symmetry. Normally one normalizes so that the *rows* sum to $1$, but here we instead use the so-called **symmetric normalization**, where the rows no longer sum to $1$.

The nice property of the symmetric normalization is that it fixes the eigenvalues into the range $[0,2]$, independently of the graph.

$\mathcal L$ can also be seen as:

$$\mathcal L = D^{-1/2} L D^{-1/2}$$

All three matrices, being symmetric and real, are — besides being endomorphisms (linear operators) — also normal matrices, as a consequence, eigenvalues are real, so we can use the **Rayleigh quotient** to find their eigenvalues.

## Rayleigh quotient

The Rayleigh quotient is a function $\mathbb R^n \setminus \{0\} \to \mathbb R$:

$$R_A(x) = \frac{x^\top A x}{x^\top x}$$

where $x$ is the vector we choose, $x^\top A x$ is an inner product, and $x^\top x$ is the squared norm of $x$.

**NB:** if $x$ is an eigenvector, then:

$$R_A(x) = \frac{x^\top A x}{x^\top x} = \frac{x^\top \lambda x}{x^\top x} = \lambda \cdot \frac{x^\top x}{x^\top x} = \lambda$$

i.e. the quotient returns exactly the associated eigenvalue.

### Using Rayleigh to find extremal eigenvalues

We can use the Rayleigh quotient to find the largest and smallest eigenvalues, and from these derive all the remaining eigenvalues and eigenvectors.

Since the matrix is real and symmetric, all of its eigenvectors form an orthonormal basis of $\mathbb R^n$. Let's see what happens to the denominator, writing $x = \sum_i a_i v_i$ in that eigenbasis:

$$x^\top x = \Big[\sum_i a_i v_i\Big]\cdot\Big[\sum_j a_j v_j\Big] = \sum_{i,j} a_i a_j\,(v_i \cdot v_j) = \sum_i a_i^2$$

because for $i \ne j$, $v_i \perp v_j$ (orthonormal basis) so $v_i \cdot v_j = 0$, while for $i=j$, $v_i\cdot v_j = 1$.

For the numerator:

$$x^\top A x = x^\top \sum_i \lambda_i a_i v_i = \sum_{i,j} a_i \lambda_i a_j\, v_j\cdot v_i = \sum_i a_i^2 \lambda_i$$

again using $v_i \cdot v_j = 0$ for $i\ne j$.

So:

$$R_A(x) = \frac{\sum_i \lambda_i a_i^2}{\sum_i a_i^2}$$

Given an ordering of the eigenvalues such that $\lambda_i \le \lambda_n$ for all $i$, we have $a_i^2\lambda_i \le a_i^2\lambda_n$ for every $i$, hence:

$$\sum_i a_i^2\lambda_i \;\le\; \sum_i a_i^2 \lambda_n = \lambda_n \sum_i a_i^2 \quad\Longrightarrow\quad R_A(x) \le \lambda_n$$

So $R_A(x)$ attains its maximum value $\lambda_n$, achieved when $x = v_n$. Symmetrically:

$$\max_{x\neq 0} R_A(x) = \lambda_n, \qquad \min_{x\neq 0} R_A(x) = \lambda_1$$

This extends to any $v_k$ with $1 \le k \le n$, not just $v_1$ and $v_n$: take the subspace $W = \text{span}(v_1,\dots,v_k)$. Every vector $x \in W$ has the form $\sum_{i=1}^k a_i v_i$, and the Rayleigh quotient restricted to $W$ returns $\lambda_k$:

$$\max_{x \neq 0,\; x \in W} R_A(x) = \lambda_k$$

We take the span from $v_1$ up to $v_k$ because the associated eigenvalues keep an increasing order, so this is just re-applying Rayleigh to the corresponding subspace.

**Summary:** we use the Rayleigh quotient to find eigenvectors and eigenvalues.

## Spectral Properties

### Theorem 1 — non-negative eigenvalues

$$\lambda_i \ge 0 \;\; \forall\, 1\le i\le n \quad\Longrightarrow\quad \gamma_i \ge 0 \;\; \forall\, 1\le i \le n$$

where $\gamma_i$ are the eigenvalues of the normalized Laplacian $\mathcal L$.

**Proof.** Start from Rayleigh and rewrite the numerator using $L$: given an undirected graph $G=(V,E)$ with vertices $v_1,\dots,v_n$, for every $x\in\mathbb R^n$:

$$x^\top L x = x^\top(D-A)x = (x_1\ \cdots\ x_n)\begin{bmatrix} d_1x_1 - \sum_{v_i\sim v_1} x_i \\ \vdots \\ d_nx_n - \sum_{v_i\sim v_n} x_i\end{bmatrix}$$

$$= \sum_{i=1}^n x_i\Big(d_ix_i - \sum_{v_i\sim v_j} x_j\Big) = \sum_{i=1}^n x_i^2 d_i - \sum_{v_i\sim v_j} x_i x_j \;{=}\; \sum_{v_i\sim v_j} (x_i-x_j)^2$$

So the numerator is always positive (a sum of squares), and the same holds for the denominator (the squared norm of $x$). Hence, when $x$ is an eigenvector, the associated eigenvalue is necessarily non-negative.

### Theorem 2 — eigenvalue ranges

Let $G=(V,E)$ be a graph where $d$ is the maximum degree among the vertices of $G$. Then:

- the interval $[-d,\,d]$ contains the eigenvalues of the adjacency matrix $A$,
- the interval $[0,\,2d]$ contains those of the Laplacian $L$,
- the interval $[0,\,2]$ contains those of the normalized Laplacian $\mathcal L$.

> **Question for later:** how do these intervals influence positional encoding? Does the fact that $\mathcal L$'s interval is fixed independently of $d$ make learning easier (or any difference at all)? Are there empirical results where one of the three matrices consistently prevails? Is it always the same matrix that wins, across every task?

### Theorem 3 — bipartiteness

Given a graph $G=(V,E)$, the following statements are equivalent:

1. $G$ is a bipartite graph
2. if $a$ is an eigenvalue of $A$, then $-a$ is also an eigenvalue
3. $\displaystyle\sum_{i=1}^n a_i^r = \text{Tr}(A^r)$ for every odd $r$

> **Question for later:** how does point 2 tie into positional encoding?

---

*(Notes end here — to be continued.)*
