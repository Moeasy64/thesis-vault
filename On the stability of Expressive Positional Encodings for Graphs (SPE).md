It Addresses two fundamental challenges of using Laplacian eigenvectors as positional encodings: 
1. **Non-uniqueness**: there are many different eigendecompositions of the same Laplacian
2. **Instability**:  small perturbations to the Laplacian can result in completely different eigenspaces, leading to unpredictable changes in positional encoding i.e. we would like that similar graphs has similar spectra. The root cause of instability is identified as a "hard partitioning" of eigenspaces. SPE instead introduces a **soft and learnable partition** of eigenspaces in an eigenvalue-dependent manner, simultaneously achieving stability (from the soft partition) and expressivity. This paper is particularly relevant for understanding why the Fiedler value ($\lambda_2$) is unstable in the presence of degenerate or near-degenerate eigenvalues.

Basis invariant methods such as BasisNet are unstable because they apply different neural networks to different eigensubspaces. In a high-level view, they perform a hard partitioning of eigenspaces and treat each chunk separately.

In this work, we present Stable and Expressive Positional Encodings (SPE). The key insight is to perform a **soft and learnable** “partition” of eigensubspaces in a eigenvalue dependent way, hereby achieving both stability (from the soft partition) and expressivity (from dependency on both eigenvalues and eigenvectors).

In order to achieve both stability and expressivity, the paper proves the SPE architecture
* Is at least as expressive as the most expressive state of the art method. Moreover, they prove that the network can count various graph substructures, such as cycles, rings, cliques or paths of length $k$.
* Of course expressivity and stability have a negative correlation, anyway they empirically demonstrated through the dataset DrugOOD, that contains completely different data between the training and test data, their expressivity power besides the SPE stability. They basically softened the negative correlation between the two.
* They proved the stability thanks to the soft partition by showing that, if we take the first $d$ eigenvector the Laplacian is determined by the gap between the $d$th and $d-1$th smallest eigenvalue. This implies that the method is stable regardless of how the used $d$ eigenvectors and eigenvalues change.
## Definition of PE stability
A Positional Encoding method $PE: \mathbb{R}^{n \times d} \times \mathbb{R}^d \to \mathbb{R}^{n \times p}$ where $d$ is the number of eigenvector chosen ($d \leq n$) and $p$ is the final positional encoding embedding dimension, is called *stable* if there exist constants $c$, $C >0$ such that for any Laplacian $L, L'$:
$$
||PE(EVD(P) - P_*PE(EVD(P'))||_F \leq C \cdot|| L-P_*L'P_*^\top||^c_F 
$$
where $P_* = arg min_{P \in \Pi(n)}|| L - PL'P^\top||_F$ is the permutation matrix matching two laplacians.
* $||\cdots ||_F$ is the [[Frobenius Norm]].
* $\Pi(n)$ represents the set of [[Permutation Matrices]] of dimension $n \times n$.  Finding that permutation $P*$ means finding the permutation matrix that minimizes the Frobenius distance between the two matrices. The one that best aligns the two Laplacians, meaning we are renaming the nodes in order to be as similar as possible. We apply the same optimal permutation to our positional encoding matrices. The idea is minimizing the structural difference between the two graphs by applying the permutation on the laplacians (right side), the application of the permutation to the positional encoding is just for aligning the embedding of the laplacian and the PE matrices in order to apply the Frobenius norm correctly.
* $C$ resembles the [[Lipschitz Continuity]] constant. This continuity ensures that an amount of change of the input is bounded to a certain constant, impeding a strong change in the output. In this case, we say that a small change in the laplacians (minimizing the difference between the two matrices first) will cause a change of at most $C$ in the PE embedding matrices of the two graphs. Of course the Lipschitz continuity grants us a mathematical stability, anyway if $C$ is very big, the model will remain practically unstable.
* $c$ is another constant that let the model relax the continuity constraint. By introducing this constant, we are using the Holder continuity instead of the Lipschitz one ($c=1$). The idea is that when we observe the usual "small changes " (values near 0), we can use a value $0<c<1$ to permit the model to reflect a bigger difference than a little fraction when changing the PE embeddings, also introducing a non linear growing.
#### Stability implies permutation equivariance
Let $L=PL'P^\top$ for some $P \in \Pi(n)$, we obtain permutation equivariance by transforming the upper equation:
$$
||PE(EVD(PL'P^\top) - P*PE(EVD(P'))||_F \leq 0
$$
$$
PE(EVD(PL'P^\top) = P*PE(EVD(P'))
$$
### SPE: How to achieve stability
In order to achieve stability, the trick is to use a soft partition of the eigensubspace: previous methods used an hard partitioning of the eigensubspace, this means that the various eigenvalues were collected in clusters; every eigenvector associated to the same cluster (based on its eigenvalue) is treated in the same way, giving the same weight to any of the eigenvector, completely ignoring the differences between the two eigenvalues. This partitioning also causes to have non-continuous functions, avoiding the possibility of using the Lipschitz continuity in primis: if a small perturbation in the input shift an eigenvalue in another cluster of the partition, that associated eigenvector will get a complete different treatment, instead soft partitioning substitute the clustering in favor of a weighted sum of the eigenvalues, granting the continuity and a small change of the weight if the eigenvalues shifts. Because the weights depends to the eigenvalue, if two dinstinct ones tends to converge or be exactly the same, the filters $\phi_i, \phi_j$ will assign the same (or almost) weight, as a consequence, the way their corresponding eigenvectors are processed its very similar. Even if the two eigenvectors wil cause to switch between each other, the impact on the network will be minimal, granting stability.

$$
SPE(\lambda, V) = \rho(V diag(\phi_1(\lambda), V diag(\phi_2(\lambda) \dots V diag(\phi_m(\lambda)))
$$
* Every $\phi_i(\lambda)$ is a stand-alone neural network that outputs a weight. Every NN acts like a filter on the vector containing the $d$ smallest eigenvalues. This filter are continuous, granting the use of the Lipschitz continuity. All this NN must be permutation equivariant (such as a MLP), meaning that $\phi_i(P\lambda) = P \phi_i(\lambda)$. The order of the eigenvector in the matrix does not influence the output.
* $V diag(\phi_i(\lambda))V^\top$ it's very similar to a spectral convolution operator on graphs, but it is just a weighted sum of the eigenvectors based on their associated eigenvalues, but weighted by the filter output $\phi_i(\lambda)$, in fact we are not using the matrix of the node feature embedding, but the eigenvector matrix $V$.
* $\rho$ is yet another neural network also permutation equivariant. The input of $\rho$ is a tensor as seen in the image below.
![[Pasted image 20261025195029.png]]
 The permutation equivariant of $\rho, \phi_i$ grants the PE to be basis invariant. This means that, if there's a swap  for two eigenvectors (because they are equal or for a perturbation), the weights are swapped repsectively, 

**Assumption 3.1.** *The key assumptions for SPE are as follows:*
* $\phi_\ell$ and $\rho$ are permutation equivariant (see definitions after SPE Eq. (2)).
* $\phi_\ell$ is $K_\ell$-Lipschitz continuous: for any $\boldsymbol{\lambda}, \boldsymbol{\lambda}' \in \mathbb{R}^d$, $||\phi_\ell(\boldsymbol{\lambda}) - \phi_\ell(\boldsymbol{\lambda}')||_F \le K_\ell ||\boldsymbol{\lambda} - \boldsymbol{\lambda}'||$.
* $\rho$ is $J$-Lipschitz continuous: for any $[\boldsymbol{A}_1, \boldsymbol{A}_2, ..., \boldsymbol{A}_m] \in \mathbb{R}^{n \times n \times m}$ and $[\boldsymbol{A}'_1, \boldsymbol{A}'_2, ..., \boldsymbol{A}'_m] \in \mathbb{R}^{n \times n \times m}$, $||\rho(\boldsymbol{A}_1, \boldsymbol{A}_2, ..., \boldsymbol{A}_m) - \rho(\boldsymbol{A}'_1, \boldsymbol{A}'_2, ..., \boldsymbol{A}'_m)||_F \le J \sum_{l=1}^m ||\boldsymbol{A}_l - \boldsymbol{A}'_l||_F$.

 Theorem 3.1 (Stability of SPE). Under the previous Assumption, SPE is stable with respect to the input Laplacian: for Laplacians $L,L'$.
 Proposition 3.2 (Basis Universality). SPE can universally approximate any continuous basis in-variant function.
