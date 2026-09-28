Being basis invariant means that every function must be invariant in respect to rotation and reflections (reflections means the sign invariance) for every eigenspace. This is important otherwise the result will depend on the choice of the basis we've done between the all possible basis for that eigenspace.
## Why $O(d)$ ([[Orthogonal Group]]) is relevant in SignNet/BasisNet

When a graph Laplacian has an eigenvalue of multiplicity $d > 1$, the corresponding eigenspace has infinitely many valid orthonormal bases. If $V_i \in \mathbb{R}^{n \times d}$ is one such basis, then $V_i Q$ is an equally valid basis for any $Q \in O(d)$. This is not a pathological edge case — the paper shows empirically that over 64% of molecule graphs in ZINC have at least one eigenspace of dimension $> 1$.

This creates a fundamental problem: a naive neural network $f(V_i)$ will produce different outputs for $V_i$ and $V_i Q$, even though they represent exactly the same geometric object (the same subspace). The output of the network would then depend on an arbitrary numerical choice made by the eigensolver, not on any meaningful property of the graph.

The key insight is that the orthogonal projector $V_i V_i^\top$ is invariant under this ambiguity:

$$V_i Q (V_i Q)^\top = V_i Q Q^\top V_i^\top = V_i V_i^\top$$

so it encodes the subspace itself, independently of the chosen basis. BasisNet exploits this by feeding $V_i V_i^\top$ into the network instead of $V_i$ directly, making the architecture invariant to the full group $O(d_1) \times \cdots \times O(d_l)$ acting on the eigenspaces.

Sign invariance — the simpler symmetry addressed by SignNet — is recovered as the special case $O(1) = {-1, +1}$, which applies when all eigenvalues are distinct ($d_i = 1$ everywhere).