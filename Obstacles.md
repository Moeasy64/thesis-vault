## Weisfeiler-Leman Test
→ [Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019.](https://arxiv.org/pdf/1810.00826)
message-passing GNN has a **limited expressive power** that is as most expressive as Weisfeiler-Leman test. It is a graph isomorphism test. This test is an algorithm that creates an histogram for a graph, based on the nodes and:
* If the two histograms differs, then the two graphs are not isomorphic.
* If they are the same, the two graphs could be isomorphic.
It is proven that MPNN are upper bounded to 1-WL test: if the WL maps the two graphs to the same histogram, then the two graphs are indistinguishable from each other for the MPNN.

> [!EXAMPLE] Example
> An hexagon graph is indistinguishable from two triangle graphs.


---
## Sign Ambiguity
when using the graph laplacian, its eigenvectors are not unique: if $v$ is an eigenvector, than so is $-v$.
The solver for the eigendecomposition will find $v$ or $-v$ depending on different parameters.

If the network is not invariant to the sign ambiguity, the same graph will be treated as two completely different inputs for the positional encoding, causing a constant (failing) oscillation of the parameters trying to align the predictions of feature constantly fluctuating, impeding convergence during training. 

Moreover, trying to make the model invariant to the sign through augmentation requires to use a lot of the available parameters in hoping that the model will learn it.

---
## Basis Ambiguity and eigenvalue multiplicity
When a graph has a strong symmetry, it is easy that its eigenvalue gets an algebraic multiplicity $k > 1$.
The associated eigenspace has dimension greater than 1, so the solvers will extract an arbitrary base of the space by choosing two orthogonal eigenvector, causing a combinatorial explosion of the input.

The cause will result in a strong overfitting: during training, hopefully the network will learn how to treat the different representations seen in the same way, but will perform badly on a very similar graph with a completely different choice of  basis never seen before.

 Applying basis transformations to input eigenvectors should not change the output of the neural network. This avoids the pathological scenario where different eigendecompositions of the same Laplacian produce different model predictions.
 
---
## Limits of the invariant approaches
Different approaches have tried to limit the problem through augmentation, such as [[rethinking_graph_transformers.pdf]].
Augmentation suggest to flip the sign randomly during the epochs. This does not solve the problem mathematically, because the research space still is $2^k$ with $k$ eigenvectors, exponentially slowing the training.

Other proposed invariant architectures like [[SIGN AND BASIS INVARIANT NETWORKS FOR SPECTRAL GRAPH REPRESENTATION LEARNING.pdf]] 
In this case, we need to add a strong inductive bias to the network, limiting its expression because the possible set of learned functions are limited, also increasing the training time in order to find a suboptimal solution. 
An example is SignNet that limits the learned function to $f(v) + f(-v)$

Although these basis invariant methods have the right symmetries, they do not yet account for the fact that two Laplacians that are similar but distinct may produce completely different eigenspaces.

---
## Instability to perturbation on the graph topology
Small perturbations to the input Laplacian **should only induce a limited change of final positional encodings**. This “small change of Laplacians, small change of positional encodings” actually generalizes the previous concept of basis invariance and proposes a stronger requirement on the networks. But this stability (or continuity) requirement is a great challenge for graphs, because small perturbations can produce completely different eigenvectors if some eigenvalues are close (Wang et al. (2022a), Lemma 3.4). Since the neural networks process eigenvectors, not the Laplacian matrix itself, they run the risk of being highly discontinuous with respect to the input matrix, leading to an inability to generalize to new graph structures and a lack of robustness to any noise in the input graph’s adjacency.

To understand the problem, we need to introduce the **spectral gap**: the absolute difference between eigenvalues. If the graph is complex (ex. social networks) the spectral gap tends to be near zero, causing the eigenspaces to mix and collapse with each other, so the network tends to lose its **continuity**: In Machine Learning, we expect that similar inputs (similar graphs/laplacians like remove a single edge) generate similar representations in the latent space. In graphs, this does not happen, causing the network to lose its **stability**, as a consequence, two similar inputs will be mapped into completely different regions of the latent space, causing the network to generalize badly (lack of robustness). Methods that only focus on sign invariance (Dwivedi & Bresson, 2021; Kreuzer et al., 2021; Lim et al., 2023), for instance, are not guaranteed to satisfy “same Laplacian, same positional encodings” espetially if  multiplicity of eigen-values exists.

## Stability Vs. Expressivity
We can define stability as "same laplacian, same positional encoding".
Basis invariant methods such as BasisNet are unstable because they apply different neural networks to different eigensubspaces (hard partitioning of the eigensubspace).

 Wang et al. (2022a) proposes a provably stable positional encoding. But, to achieve stability, it completely ignores the distinctness of each eigensubspaces and processes the merged eigenspaces homogeneously. Consequently, it loses expressive power and has, e.g., a subpar perfor-mance on molecular graph regression tasks (Rampasek et al., 2022).

See [[On the stability of Expressive Positional Encodings for Graphs (SPE)]] to address all the previous issues.
