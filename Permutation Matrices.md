We define the set of permutation matrices of dimension $n$ as $\Pi(n) = {P \in {0,1}^{n \times n} | PP^\top}$, which is the set of all orthogonal matrices of length $n \times n$. Geometrically, applying this transformation to a matrix or a vector means shuffling its elements without altering the value.
## Permutation equivariance
There's an important property called **permutation equivariance**.
Given a function $f: \mathbb{R}^n \to \mathbb{R}^n$ operating of vectors $x$ of length $n$, we say that is permutation equivariant if applying a permutation do not alter the result after the application:
$$
f(Px) = Pf(x).
$$
If the function $f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n}$ operates on square matrices $n \times n$ we say that is permutation equivariant if applying a permutation do not alter the result after the application:
$$
f(PXP^\top) = Pf(X)
$$