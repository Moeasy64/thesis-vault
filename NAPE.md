[[NAPE_Numbering_as_a_Position_Encoding_in_Graphs.pdf]]
## Motivation
NAPE proposes a positional encoding (PE) for graphs that **never computes Laplacian eigenvectors**. Eigenvector-based PEs have two issues:
1. **Non-uniqueness of the encoding**
	- *Sign ambiguity*: if $v$ is an eigenvector of the Laplacian, so is $-v$.
	- *Basis ambiguity*: for an eigenvalue of multiplicity $>1$, any orthonormal basis of the corresponding eigenspace is a valid choice.
	As a consequence, a node's positional representation in the embedding space is not uniquely defined.
2. **Computational cost** of $O(|V|^3)$ (plus memory requirements) to compute the eigenvectors of very large graphs.
## Idea
NAPE builds on the sinusoidal PE of Vaswani et al. ([[Attention Is All You Need.pdf]]), which is related to the Laplacian eigenvectors of a (directed) path graph, where the node order is already given.`

On a general graph there is no natural order ($|V|!$ possible numberings), so NAPE **learns a numbering** of the nodes so that two key objectives are pursued:
1.  Proximity preservation: Nodes connected by edges in the original graph should be positioned close together within the hypercube embedding. 
2.  Degree-based locality: Nodes with identical degrees in the original graph should be mapped to nearby locations within the hypercube.
## Algorithm

The algorithm, at a high level, is:

1. Given an $m$-dimensional hypercube, with $m \geq \lceil \log_2 N \rceil$, each vertex of the cube is a binary label.
2. Each node $i$ gets a binary code $Z_i \in \{0,1\}^m$, i.e. a vertex of the hypercube.
3. The codes are trained (with a contrastive-style loss) so that the Hamming distance $d(x,y) = \sum_{k=1}^{m} |x_k - y_k|$ between them approximates closeness in the graph: neighbouring nodes get close codes.
4. Each code $Z_i$ is converted to an integer $t$ (its "number"), which is fed to the sinusoidal formula to obtain the PE.

In practice, the algorithm takes as input the adjacency matrix $A$, the degree vector $d$, two hyperparameters $\lambda_1, \lambda_2$ and the number of epochs.

### The binary code $Z_i$

The loss is expressed in terms of the binary codes $Z_i$, which are our final objective. However, we cannot differentiate the loss with respect to discrete bits, so the codes are not learned directly. Instead, each node has a free real-valued vector $P_i \in \mathbb{R}^m$ (the rows of a matrix $P \in \mathbb{R}^{N\times m}$, randomly initialised), and the code is obtained through a differentiable map:

$$Z_i = \sigma(\gamma P_i + \epsilon)$$

where $\sigma$ is the sigmoid applied component-wise, and $\gamma$ (steepness) and $\epsilon$ (shift) are learnable parameters. Since $Z_i \in (0,1)^m$ is a continuous relaxation of the binary code, the loss can be backpropagated through $Z_i$ down to $P_i$, which is what the optimiser actually updates at each epoch. The larger $\gamma$ is, the closer each component of $Z_i$ gets to 0 or 1. *(The paper does not explain how the final bits are obtained from $Z_i$; presumably by rounding, to be verified.)*

### The loss function

The loss is defined as a sum of three components, weighted by their hyperparameters:

$$\mathcal{L} = \mathcal{L}_{adj} + \lambda_1\,\mathcal{L}_{deg\_dist} + \lambda_2\,\mathcal{L}_{deg}$$

For graphs where all nodes have similar degree (e.g. transportation networks), $\lambda_1=\lambda_2=0$ and only $\mathcal{L}_{adj}$ is used.

The first two terms share the same contrastive structure. For each node $i$ there is a set of *positives* $(Z_i, Z_\rho)$, to be pulled close, and a set of *negatives* $(Z_i, Z_a)$, to be pushed away:

$$-\log\frac{\exp\big(d(Z_i,Z_\rho)\big)}{\sum_{a\in\text{negatives}}\exp\big(d(Z_i,Z_a)\big)}$$

Contrasting positives against negatives is what gives *contrastive learning* its name. The denominator is essential: without it, the trivial solution $Z_i=Z_j$ for all nodes would minimise every distance (same encoding for every node), so the negatives prevent collapse. *(Note: as written in the paper this form seems to push positives apart; for it to pull them together the distance should enter with a minus sign, $\exp(-d)$. To be checked on the original code.)*

* $\mathcal{L}_{adj}$ (proximity preservation). The positives are the neighbours $\mathcal{N}(i)$ of node $i$ and the negatives are the non-neighbours $\mu(i)$. The inner average over $\mathcal{N}(i)$ keeps hubs from weighing more than peripheral nodes. This is the term that makes the code a *position*: nodes close in the graph get close codes on the hypercube.
* $\mathcal{L}_{deg\_dist}$ (degree-based locality). The positives are the nodes with the same degree as $i$, even if far away in the graph, and the negatives are the nodes with a different degree. It encodes a node's structural role (hub, leaf, isolated node), and it is the only signal available for isolated nodes, which have no neighbours for $\mathcal{L}_{adj}$. It can conflict with $\mathcal{L}_{adj}$, so $\lambda_1$ balances role against topology.
* $\mathcal{L}_{deg}=\|ZW_{deg}-d\|_2$ (degree decodability). A linear regression from the bits of $Z_i$ to the degree $d_i$. It is not relational (it acts on one node at a time) and it anchors the degree to a precise direction of the code space, so that the information is recoverable from $Z_i$.

Once the optimal binary representation is found $Z_i$ is converted to the decimal equivalent $t \in \mathbb{N}$. Now we can apply the standard Attention is All you Need positional encoding, To use this encoding in any GNN algorithm we just need to add the $pos(t)$ vector to the feature matrix of the graph.

> [!NOTE] A big limitation
> As the paper underline, this optimization runs on just a single graph, so the optimization needs to be runned at inference time, every time, before using the downstream model. It does not learn a positional encoding between graphs.

## Scaling the NAPE algorithm

The algorithm described above does not improve on the eigenvector approach as it is: the clustering terms (2) and (3) each have a computational cost of $O(|V|^3)$, the same as computing the eigenvectors, so NAPE would bring no advantage. (Intuitively: for each node $i$ we sum over its positives, and for each positive the denominator sums over up to $|V|$ non-neighbours.) To reduce the cost to $O(|V|^2)$ the paper introduces two techniques:

* redefine what it means for two nodes to be *neighbours*, using the clusters found by random walks;
* replace the *non-neighbours* with a small set of negative samples.

### Neighbours from random walks

Eigenvectors of the Laplacian identify clusters of nodes, and each node belongs to a cluster that can be seen as its local neighbourhood. A random walk, under certain conditions, mostly traverses nodes within the cluster of its starting node. So the neighbours of $i$ are redefined as the nodes that **co-occur** (appear) in a random walk of length $l$ starting at $i$:

$$\mathcal{N}(i) := R^m_l(i),$$

the set of nodes visited in $m$ independent random walks of length $l$ from $i$. Note that these can include non-adjacent nodes (Fig. 3 of the paper), and that the relation is not symmetric: $j \in R^m_l(i)$ does not imply $i \in R^m_l(j)$.

The walks follow the 2nd-order strategy of node2vec, where the parameters $p$ and $q$ weight the transition probabilities and interpolate between a depth-first (DFS) and a breadth-first (BFS) exploration. We set $p<1$ and $q>1$ to obtain a **BFS-like walk**, which stays close to the starting node and therefore follows the clustering behaviour described above ($p=0.25$, $q=2$ in the experiments). The ablation study confirms that BFS walks give a significantly better accuracy than DFS walks.

### Non-neighbours from negative sampling

The main component of $\mathcal{L}_{adj}$ is

$$g(i,p) = \log \frac{\exp\big(d(Z_i,Z_p)\big)}{\sum_{a\in\mu(i)} \exp\big(d(Z_i,Z_a)\big)},$$

where the number of non-neighbours $|\mu(i)|$ varies per node and is $|V|-1$ in the worst case. Following the negative sampling technique of Mikolov et al., it is approximated with a constant number $k$ of nodes randomly drawn from $V$:

$$g(i,p) \approx \log \sigma\big(d(Z_i,Z_p)\big) - \sum_{a=1}^{k} \log \sigma\big(d(Z_i,Z_a)\big), \qquad a \sim P_V,$$

where $P_V$ is a distribution over the nodes, proportional to their degree. In the original formulation the argument of $\sigma$ is a dot product between vector representations; here it is replaced by the Hamming distance, since the $Z_i$ are binary vectors on the vertices of a hypercube. *(As for (2), the sign convention should be checked against the original code.)*
### Degree distribution
For $\mathcal{L}_{deg\_dist}$ the paper builds an auxiliary graph $H(V,\tilde{E})$ from $G(V,E)$, in which only nodes with the **same degree** in $G$ are connected. $H$ is therefore a union of disconnected complete subgraphs. This is especially useful when $G$ has several isolated nodes, since it introduces similarities between them.

The cost of (3) cannot be assumed to be the same as (2): for the isolated nodes, for instance, $\kappa(u)=|V|-2$. Moreover, since every connected subgraph of $H$ is complete, a random walk would give no additional information on the local neighbours. So the sets are scaled down by sampling directly:

* $\deg(i)$: $l$ nodes randomly sampled among those connected to $i$ in $H$;
* $\kappa(i)$: $k$ nodes randomly sampled outside the subgraph of $i$, used as negative samples as in (10). Here the sampling is *not* proportional to the degree distribution.
### Resulting computational cost
With the random walks $R^m_l(i)$ approximating the neighbours and $k$ negative samples approximating the non-neighbours, the cost of the loss $\mathcal{L}$ is

$$O(|V|^2) \quad \text{with the condition} \quad lmk \leq |V|,$$

assuming that the cost of $\mathcal{L}_{deg}$ is negligible with respect to $\mathcal{L}_{adj}$ and $\mathcal{L}_{deg\_dist}$.

*(Notation: in this section $m$ is the number of random walks per node and $l$ their length, not the number of bits of the code, which the experiments denote with $d$.)*
### Frequency tuning
The frequency $\omega_b$ of the sinusoidal encoding affects how robust the learnt position encodings are, and it depends on the number of nodes. For a path graph with $|V|$ nodes, the Laplacian eigenvectors are sinusoids with frequencies

$$\omega_b = \frac{2\pi b}{|V|}.$$

Equating this with the frequency of the original sinusoidal encoding, $\omega_b = 1/\eta^{2b/d}$ with $\eta = 10000$, gives

$$\eta = Q\,|V|^{\beta}, \qquad Q = (2\pi b)^{-d/2b}, \quad \beta = \frac{d}{2b},$$

hence $\eta \propto |V|^\beta$. For this reason, the default constant $10000$ is replaced by a value $\eta$ proportional to the number of nodes of the graph (for example $\eta = 10^5$ for WikiCS and $1.5\cdot10^5$ for PubMed), and it is tuned as a hyperparameter.

Rationale from spectral graph theory: eigenvectors identify **clusters** of nodes. Random walks tend to stay inside a cluster, so they can be used to find clusters without computing any eigenvector.
NAPE generates **almost** unique embeddings, some of them can be duplicated inside a graph.