
[[Graph Positional and Structural Encoder.pdf]]
## Motivation
As we already know, in a Graph Transformer (GT) the full self-attention between all pairs of nodes discards the inductive biases tied to the graph structure, which message passing models exploit well. In order to re-introduce this information, positional and structural encodings (PSE) are necessary. However, there is no PSE that performs at the top for every task:
* Random walk encodings are better on molecular property prediction
* Laplacian eigenvectors are better on tasks with long-range dependencies 
* Empirically, concatenating different PSEs does not reliably improve performance: it can be worse than the best single PSE (e.g. GPS on ZINC), even if in some settings it helps (Table 3)

GPSE (Graph Positional and Structural Encoder) is a self supervised MPNN (Message Passing Neural Network, GNN architecture) with a transferable and pre trained encoder. which extracts latent representations common to various PSE using the graph structure.
Here the GPSE is not the final model, is just a pre-train step where we create an encoder for the generation of positional and structural feature that can be injected inside a final downstream task. They used the standard GPS architecture trained from zero as downstream because it represented the state of the art at that moment. Anyway, other downstreams have been chosen exactly for demonstrate how this GPSE extracted features are in fact independent from the model.

> [!NOTE] Use this PSE inside the Kreuzer architecture?
> It could be a useful test to run to see if the spectral part is giving other informations, even flat transformers have been tested in this paper.

The core idea is to train a MPNN (our GPSE in the pipeline) only on feature resembling the graph structure.
![[Pasted image 20261014155907.png]]As shown in the image on the step C, vertexes and edges (and the noise vector) are the only information in input to GPSE, edge and node features are passed directly the the downstream model. Here $G$ is a graph taken in input during the self-supervised learning of GPSE, $G'$ is an arbitrary graph taken in input during inference, and in fact the GPSE weights are frozen. 

**Step A: building the input and forward pass.**
A query graph $G$ is taken from the pre-training dataset (MolPCBA). Only its structure (nodes and edges) is kept: no atom or bond types. Since a MPNN needs an initial value for every node, each node $i$ receives a random vector $x_i \sim \mathcal{N}(0, I) \in \mathbb{R}^{20}$. This is pure noise, not an embedding: it is not learned and carries no meaning. Its role is to break the symmetry between nodes and to push the network beyond[^1] 1-WL expressivity (Required by specific PSEs).  The message passing runs on the graph and exchanges messages along its edges. The noise is projected into the hidden space only once, at the input:

$$
h_i^{(0)} = \mathrm{ReLU}\left(x_i W_{\mathrm{inp}}\right), \qquad W_{\mathrm{inp}} \in \mathbb{R}^{20 \times 512}
$$

Then the 20 GatedGCN layers (with residual connections, gating and the virtual node) all work at constant dimension 512 and produce $h_i^{(L)} \in \mathbb{R}^{512}$ for every node. For a graph with $n$ nodes the output is a matrix in $\mathbb{R}^{n \times 512}$.

**Step B: self-supervised training.**
In parallel, the 6 target PSEs are computed from $G$ with classical algorithms (eigendecomposition of the Laplacian, powers of the random walk matrix, cycle counting, ...). These are the labels, obtained for free. Each PSE has its own independent MLP head that tries to reconstruct it from $h^{(L)}$:

$$
\hat{y}_{i,k} = \mathrm{ReLU}\left(h_i^{(L)} W_{k,1}\right) W_{k,2}, \qquad W_{k,1} \in \mathbb{R}^{512 \times 512},\ W_{k,2} \in \mathbb{R}^{512 \times 1}
$$

Node-level PSEs (LapPE, ElstaticPE, RWSE, HKdiagSE) are predicted node by node. Graph-level PSEs (EigVal, CycleSE) first apply a sum-pooling over the nodes. The loss compares prediction and target with $\ell_1$ (magnitude) plus cosine similarity (direction), after normalizing every target to zero mean and unit variance:

$$
\mathcal{L} = \sum_{k=1}^{K}\sum_{i=1}^{M}\left[\sum_{j=1}^{|V(G_i)|}\left|y^{(i)}_{j,k}-\hat{y}^{(i)}_{j,k}\right| + \left(1-\sum_{j=1}^{|V(G_i)|}\tilde{y}^{(i)}_{j,k}\,\tilde{\hat{y}}^{(i)}_{j,k}\right)\right]
$$

where $\tilde{y}$ and $\tilde{\hat{y}}$ are the $\ell_2$-normalized versions of $y$ and $\hat{y}$. Backpropagation updates the 20 layers and the heads. **To lower the loss, the hidden space must encode the positional and structural information of each node.**

**Step C: inference and downstream.**
Once GPSE is trained, the MLP heads are discarded and the 20 layers are **frozen**. From here on, GPSE is a pure feature extractor:

1. For each graph $G'$ of the downstream dataset (any dataset, not only molecules), new noise is sampled and passed through the frozen GPSE, obtaining $H \in \mathbb{R}^{n \times 512}$. This is precomputed **offline** for the whole dataset, before any downstream training.
2. Here the downstream part begins, and everything from this point on is trained from scratch with the task loss. First, a **PE encoder** (linear or 2-layer MLP, with batch normalization and dropout; it belongs to the downstream, not to GPSE) reduces the 512 channels to a small PE dimension, which depends on the dataset (e.g. 32 on ZINC, 128 on PCQM4Mv2, 20 on MolHIV).
3. The result is **concatenated to the node features** (node only, never edges). Edge features reach the downstream model unchanged.
4. The downstream model (GPS in the paper) is trained on the task, e.g. regression of one value per molecule on ZINC. Only the PE encoder and the downstream weights are updated.

## GPSE
### Self-supervision
6 types of PSEs are learned. Positional encodings describe the relative position of each node in the graph, structural encodings describe the connectivity patterns around a node or of the whole graph. 
**Positional (node-level):** 
* Laplacian eigenvectors, LapPE (spectral analysis) 
* Electrostatic potential, ElstaticPE (computed from the pseudoinverse of the Laplacian, $L^\dagger$, interpreting nodes as charged particles)
**Structural, local (node-level):** 
* Random walk, RWSE: return probability after $k$ steps, $\mathrm{RWSE}_k = \mathrm{diag}(P^k)$ 
* Heat kernel, HKdiagSE (diagonal of the heat kernel, related to heat diffusion on the graph) Structural, global (graph-level): 
* Laplacian eigenvalues, EigValSE 
* Cycle counting, CycleSE
### Architectural Details
Some PSEs like laplacian eigenvectors (as seen in Kreuzer) require having a global view of the graph. This is true in particular for the low-frequency ones: the first non-trivial eigenvector (Fiedler vector) solves a min-cut problem on the whole graph. GPSE learns the first 4 eigenvectors. An encoder able to capture long-range dependencies is therefore crucial, and for this a MPNN with 20 layers is used. However, depth easily leads to [[over-smoothing]] (node features become nearly uniform). To alleviate it, gating and residual connections are added.

To contrast over-squashing, a **Virtual Node (VN)** is added. This node is connected to all the others, which ensures faster information exchange between distant nodes and faster convergence.

As we said, only nodes and edges are present as input, but a MPNN needs an initial value for every node $i$. So we sample a random vector $x_i \sim \mathcal{N}(0, I) \in \mathbb{R}^{20}$, which flows through the 20 layers. Since this input carries no meaningful information about the nodes, the information that ends up in the hidden representation $h_i \in \mathbb{R}^{512}$ can only come from the graph structure, which is exactly what the network is trained to learn.

### Sign ambiguity and the absolute value of the eigenvectors

The Laplacian is $L = D - M = U \Lambda U^\top$, with eigenpairs ordered as $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$. Let $\hat{U}$ be the matrix of eigenvectors of the non-trivial eigenvalues ($\lambda_i \neq 0$).

An eigenvector is defined only up to its sign: if $u$ is an eigenvector of $L$, so is $-u$, and both are equally valid. For a PSE used as an input feature this is a known issue, and it is usually handled by randomly flipping the signs during training (Dwivedi et al., 2022a) or by [[Sign and Basis Invariant Networks For Spectral Graph Representation Learning]].

GPSE, however, has to **reconstruct** the PSE as a target, not use it as a feature. With a target defined up to sign, the loss would penalize a correct prediction that happens to have the opposite sign, and the tricks above do not solve this. The solution adopted is to define LapPE as the absolute value of the $\ell_2$-normalized eigenvector:

$$
\mathrm{LapPE}_i = \left|\,\mathrm{normalize}\!\left(\hat{U}_{[:,i]}\right)\right|, \qquad \mathrm{normalize}(x) = \frac{x}{\lVert x \rVert_2}
$$

The first 4 eigenvectors are used as targets. This removes the sign ambiguity by construction, at the cost of discarding the sign information.

**Ablation (Table H.6).** The authors pre-train four versions of GPSE (5% of MolPCBA), differing only in how the LapPE target is handled:

1. `GPSE-abs`: absolute value (default)
2. `GPSE-noabs`: raw eigenvector, no absolute value
3. `GPSE-signinvar`: sign-invariant loss, taking the minimum of the losses obtained with both signs
4. `GPSE-SignNet`: target generated by a randomly initialized SignNet

| Downstream dataset                 | abs (default) | no abs     | sign-invariant loss | SignNet |
| ---------------------------------- | ------------- | ---------- | ------------------- | ------- |
| ZINC (MAE $\downarrow$)            | **0.0957**    | 0.1051     | 0.1116              | 0.1035  |
| PCQM4Mv2 (MAE $\downarrow$)        | **0.1216**    | 0.1229     | 0.1243              | 0.1232  |
| Peptides-struct (MAE $\downarrow$) | **0.2516**    | 0.2554     | 0.2594              | 0.2568  |
| Peptides-func (AP $\uparrow$)      | 0.6584        | **0.6687** | 0.6619              | 0.6647  |

> [!NOTE] Important observation
> The absolute value is best on three out of four datasets, but not on Peptides-func, where the raw version is better (differences are within roughly one standard deviation there). The authors conclude it is "similar or better" than the alternatives, and leave better strategies for learning sign-invariant representations of eigenvectors as future work.

**Connection with the thesis.**

* This supports the choice of `abs(Fiedler)` and `abs(eigs)` as features (STEP 3), because it shows that dropping the sign is a reasonable, empirically tested option.
* *Personal note:* the sign of the Fiedler vector carries information the absolute value discards, namely the partition of the graph into two sets of nodes (nodal domains). For the second approach of the thesis (using the Fiedler vector to **order** the vertices and define a canonical sequence) the signed values matter, so abs and signed versions are probably worth comparing explicitly.

### Pros and cons of GPSE

|         | Aspect                                                         | Details (source in the paper)                                                                                                                                                                 |
| ------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pro** | Unifies many PSEs                                              | A single latent representation recovers 6 different PSEs ($R^2 \approx 0.979$ on 5% of MolPCBA, $0.998$ with 90%); no need to choose the PSE by trial and error                               |
| **Pro** | Better than hand-crafted PSEs                                  | GPS+GPSE: 0.0648 MAE on ZINC, better than RWSE, LapPE and their concatenation; about 56% average MAE reduction on ZINC across 5 models (Table 3)                                              |
| **Pro** | Architecture independent                                       | Works as a feature extractor for MPNNs, GPS, SAT and a flat Transformer; the downstream model is trained from scratch with no constraints                                                     |
| **Pro** | Transferable out of distribution                               | Trained only on molecules, yet useful on larger graphs (Peptides), different connectivity (CIFAR10/MNIST superpixels) and node classification on huge graphs (arXiv, Proteins)                |
| **Pro** | Structure only                                                 | It does not use atom or bond features, so it applies to any graph dataset, and it shows less negative transfer than other SSL pre-training approaches on MoleculeNet                          |
| **Pro** | Efficient at inference                                         | Cost $O(Lm)$ (linear in the number of edges), independent of how many PSEs were used in training; faster than LapPE and RWSE, and orders of magnitude faster than computing all PSEs (AllPSE) |
| **Pro** | Beyond 1-WL                                                    | Thanks to the random input features it distinguishes 1-WL-indistinguishable graphs                                                                                                            |
| **Con** | Costly pre-training                                            | About 21 h on MolPCBA (323k graphs) and about 90 h on ChEMBL (971k graphs); the authors state it is prohibitively large for datasets with over one million graphs                             |
| **Con** | Partial expressivity                                           | On CSL and EXP, GIN+GPSE stays below explicit LapPE/RWSE (e.g. CSL test: 42.9% vs 92.5% and 100%); random features are only a "patch" to the 1-WL limit                                       |
| **Con** | Gain depends on the task                                       | Useful when graph structure is informative; on Proteins (where homophily matters more than topology) the benefit is not apparent                                                              |
| **Con** | Pre-trained on molecules                                       | Transfer to other domains is shown on benchmark graphs, but **not on 3D meshes**; whether it works on FAUST or SURREAL is an open question for the thesis                                     |
| **Con** | Loses sign information                                         | LapPE is learned as an absolute value, so the sign of the eigenvectors (e.g. of the Fiedler vector) is not recoverable from the encoding                                                      |
| **Con** | High-dimensional output                                        | $h^{(L)} \in \mathbb{R}^{512}$ must be reduced by a trainable PE encoder; the features are less interpretable than an explicit eigenvector                                                    |
| **Con** | Stochastic input *(personal note, not discussed in the paper)* | The encoding depends on freshly sampled noise at each inference, so it is not deterministic; the paper does not quantify this variability                                                     |

> [!NOTE] Does it work well on meshes?
> Transfer to other domains is shown on benchmark graphs, but **not on 3D meshes**; whether it works on FAUST or SURREAL is an open question for the thesis

[^1]: [^1]: **1-WL (1-dimensional Weisfeiler-Leman)** is a classical algorithm (color refinement) to test whether two graphs are non-isomorphic. All nodes start with the same color, then at each iteration the color of a node is updated from its own color and the multiset of its neighbors' colors:
	
	    $$
	    c^{(t+1)}(v) = \mathrm{HASH}\Big(c^{(t)}(v),\ \{\!\{\, c^{(t)}(u) : u \in \mathcal{N}(v) \,\}\!\}\Big)
	    $$
	
	    If the final color histograms of two graphs differ, the graphs are certainly non-isomorphic; if they coincide, the test is inconclusive. Since this update is the same as message passing, a MPNN cannot distinguish more graphs than 1-WL (Xu et al., 2019; Morris et al., 2019): this is what is meant by "1-WL expressivity". For example, a 6-cycle and two disjoint triangles are indistinguishable (every node has 2 neighbors), although only the latter contains triangles. Random input features, as used in GPSE, give each node a distinct label and allow going beyond this limit (Sato et al., 2021; Abboud et al., 2021).
