The paper states that prior works related to Self-supervised learning uses methods based on the reconstruction or disturbing the graph, without considering the **centrality** of the nodes: many nodes can be more or less important than others. This information injection lead to a better graph **representation** (let's just say, embedding, $\mathcal{H_g}$ in this paper) that it is not just based on the structure itself, but also on node importance (globally and locally). Here, the centrality of a node is based on its degree and its eigenvector centrality.

This work has aspects related to other paper seen, such as
* [[SIGN AND BASIS INVARIANT NETWORKS FOR SPECTRAL GRAPH REPRESENTATION LEARNING.pdf]], 
* [[what_improvise_generalization_of_graph.pdf]],
* [[Spectral Geometry Processing with Manifold Harmonics.pdf]]
Here, the eigenvector centrality is explored as well, even though the sign and basis infariance is not addressed directly.
As we saw in SAN, the positional encoding embedding is concatenated (or added) to the node or edge embedding, here the PE information is injected through a self attention mechanism (note that this architecture can't be considered a transformer even though it uses attention in a step of its pipeline).

# Architecture
![[Pasted image 20261021204457.png]]
$f$, $f^t$ are actually the same encoder just seen in different moments!

There are three different modules where the first two has a different duty with respect to the last:
### Representation pre-training:
their duty is to learn the best representation for nodes based on their degree (node-level) and eigenvector centrality (graph-level).
#### node-level importance learning module
exploits degree centrality to find representation of node with equal or similar importance: "this node is important because it has lots of connection, and another node with the same amount of connection should have a similar representation". Predictive learning is used to map each node representation into the corresponding degree.
We need a way to learn the best embedding/representation for this module, so we need a ground truth and a loss function based on it. We can define the ground truth as the actual degree of the node, so
$$
\mathcal{Y}_i^d = d_i = \sum_jA_{ij}
$$
Then a predictor $\mathcal{P}_d$ is trained to predict the degree of each node based on its representation; the loss of its predictor is
$$
\mathcal{L}_d=-\frac{1}{n}\sum_{i=1}^n \mathcal{Y}_i^d log(\mathcal{P}_d(h_i;\theta_d))
$$
It is not clear in the paper what the predictor is, but presumably, a multilayer perceptron. The backpropagation can work fine because: the encoder $f$ takes in input the node features and $A$, and $h_i$ in output, $h_i$ is given in input to $\mathcal{P}$. The loss backpropagates from $\mathcal{P}$ to $f$, changing the representation of the nodes iteration after iteration.
#### graph-level importance learning module
importance between all nodes in the graph based on eigenvector centrality: "this node influence the graph in this way, because of its eigenvector centrality". The [[Eigenvector Centrality]] is used to measure the influence of a node inside the graph, by identifying its neighbours and their importance. 
As noted from the eigendecomposition $\lambda_{max}$ has an associated eigenvector of all real and positive components, that components  $v_1 \dots v_n$ are the score of the eigenvector centrality of every node.
Anyway, the eigendecomposition has its limitations, so we prefer the use of  truncated **Singular Value Decomposition (SVD)**:
1. It is computationally expensive and it is very sensitive to small perturbations in case of ill-matrices
2. Truncate to the first $k$ components holds the dominant informations with a good increase in performances.
3. It is computed once before the training.
using SVD instead of ED is accepted on undirected graphs, because $ED \equiv SVD$ if $A$ is semi-definite positive and symmetric (which is the case most of the time).
The truncated matrix $U_k$ gives us the **structural embedding** $\mathcal{S_G}$: every column $i$ corresponds to the structural embedding of the node $i$
$$
\mathcal{S_G} = \{s_1, \dots, s_n\} = U_k = f_{truncated\_svd}(A)
$$

Then, the information between the structural representation $\mathcal{S_G}$ and the semanthic representation $\mathcal{H_G}$ are fused together in order to align the representation in the same vector space and capture both local and global aspects of the graph.
To fuse the information together we use the cross-attention, using $\mathcal{H_G}$ as query and $\mathcal{S_G}$ as key and value. The idea is enrich the semantic information with its structural influence in the graph.
We obtain a new representation $\mathcal{S'_G}$ 
$$
\mathcal{S'_G} = \{s'_1, \dots s'_n\} = CrossAtt(\mathcal{H_G}; \mathcal{S_G}) = Softmax((W_qH_G)(W_kS_G))(W_vS_G)
$$

After that, we now need to define a predictor $\mathcal{P_e}$ to train the node representation based on the graph-level importance-fused structure representation called Contrastive Representation-Structure Pre-Training (CReSP) model. This contrastive learning objective is designed to capture both local node features and global structural roles by aligning these two complementary modalities in the same embedding space.
In the context of the Graph-Level Node Representation proposed by the CenPre framework, contrastive learning is not used to compare two augmented (distorted) versions of the same graph as in traditional methods, but rather to align two complementary modalities of information for the same node: its semantics and its global topological role. The loss $\mathcal{L_e}$ (not reported for simplicity) reflects this idea.
### Alignment
#### Representation Alignment module
it aligns the pre-trained node representation using the original one. Once the pre-training is finished, the two representation are "fused" in order to hold both the semantic informations (original) and structure information (both the one learned during reconstruction and CenPre). Differently from the contrastive learning strategy used before, here we are not "fusing" the two representation, but we are "aligning" the two representation in order that the model does not focus too much on learning the complex graph structure (and for numerical stability). Here, we align $\mathcal{H_G}$ and $\mathcal{S'_G}$.
The loss is called $\mathcal{L_r}$ and it exploits both $f^t$ (the pre-trained GNN learned by CenPre) and $f^o$ its the original representation.

### Overall learning
The final loss is the sum of our three losses
$$
\mathcal{L} = \lambda_1\mathcal{L_d} + \lambda_2\mathcal{L_e} + \lambda_3\mathcal{L_r}
$$
$\lambda_i$ are hyperparameters balance the three losses