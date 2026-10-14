---
Type: Paper
---
![[Papers/rethinking_graph_transformers.pdf]]
# Architecture Overview
![[Pasted image 20261009175338.png]]

# Objective
Extending the capability of transformers to non-sequential data through the **Spectral Attention Network (SAN)**.

It exploit the **Learned Positional Encoding (LPE)** in order to globally acquire the position of every node inside a graph.

The model use LPE to extract and acquire the positional informations from the **[[Laplacian Spectrum]]**.

This LPE is then added to the features of the graph and passed to a [[fully-connected transformer]].
> [!WARNING] Discrepancy in the paper
> Actually in the code and in Section 3 the LPE is concatenated to the embedding vector representing a node/edge. It is not a sum, that makes more sense, in fact the embedding vector for a node/edge is of dimention $n$, while the LPE embedding vector has a different dimention named $k$.

# Improvements
* Surpassing the typical [[over-squashing]] and [[over-smoothing]] symptom of GNNs.
* Performs better or equal than every state of the art attention based architecture and GNN on graphs.
* First fully-connected model that performs good on graphs
* Exploits the full laplacian spectrum to learn the position of every node (uses a mask for smaller graphs)

The idea behind this work born from the observation that GNN has a strict architecture, also based on the input graph, which leads to a strong inductive bias. Transformers instead use positional encoding to acquire the structure of the graph, introducing softer biases and stay independent from the input graph structure; differently from GNNs, the self attention mechanism of a transformer can be seen as passing messages between all nodes while staying independent from the input graph connectivity.
the GT model was proposed as a generalization of Transformers to graphs, where they
experimented with sparse and full graph attention while providing low-frequency eigenvectors of the Laplacian as positional encodings.

Former studies already tried to use Transformers in combination with positional encoding using the eigenfunctions of their Laplacian Matrix. They pushed the idea further by using the Laplacian Spectrum instead of Laplacian Matrix eigenvalues as positional encoding. In this way the new SAN architecture obtained these advantages in respect to previous architectures:
* Considers the spectrum of eigenvalues 
* Considers variable # of eigenvectors 
* Aware of eigenvalue multiplicities

By exploiting the fourier transform, we can view the eigenvectors as vectors positioned on the axis of eigenvalues rather than components of a matrix
![[Pasted image 20260424183846.png]]

## Intro
* What are message passing GNNs
* Weisfeiler-Lehman iso-morphism hierarchy
* Message-passing GNN limitations due to oversmoothing and over-squashing sympthoms.
* How does GNNs encode graphs on their structure
* Understanding the relation between sine function and eigenvalues for PEs and how they are embedded inside the transformer through Fourirer transform.
* Understand how sine and cosine are used for positional encoding in classic trasnformers and how they can let the model understand the exact position of a word in respect to another based on the PE vector.
* https://gemini.google.com/app/e6bd6cc3a56113c9 for positional encoding through eigenvectors
##  Learned Positional Encoding
It is "learned", so it needs some machine learning model trained to create the correct encoding. In this case a normal transformer is the choice, because we have all eigenvalues in the frequency axis, this unlocks the possibility of treating them as a sequence.
### LPE Over Nodes
First an embedding matrix $2 \times m$ is created, with $m$ the fixed number of eigenvalues chosen. the $i,j$-th position contains the value of the $j$-th dimension of the eigenvector associated to the $i$-th eigenvalue. A FFN is applied to the embedding in order to obtain a $k \times m$ matrix, with $k$ be the number of hidden features. after that, a trasformer encoder (the classical from [[Attention Is All You Need.pdf]] but without standard PE embedding) transforms the output but maintaining the matrix dimension; this process can be parallelized thanks to the multi head attention, also, we can easily parallelize the computation for every node, by creating a tensor of $m \times n \times k$ and so trating the $n$ nodes as a batch. Finally a sum pooling reduces the sequence into a fixed $k$-dimensional node embedding. The output of this LPE is **concatenated** to the input embedding, in fact in the code, the input embedding is of a smaller dimension to let the space for $k$:
```python
embedding_h = nn.Embedding(num_atom_type, d − k)
```

> [!WARNING] Discrepancy in the paper
> Here, a sign invertion during training is needed as augmentation technique. Although the function for doing so is present in the code, it is never used. So every test reported for nodes LPE is actually missing this important augmentation technique that it is cited to be used in the paper.
### LPE Over Edges
This method also makes the LPE sign-invariant. Moreover, the LPE over nodes is an *absolute* positional encoding, because hte positional vector of a node $j$ is just built upon its component $\phi_{i,j}$ and its spectrum. Here the LPE is *relative*. The idea is to consider relative positions encoded as edge features: given two nodes $j_1$ and $j_2$, we'll use two operators:
$$
|\phi_{i,j_1} - \phi_{i,j_2}|
$$
$$
\phi_{i,j_1} \phi_{i,j_2}
$$
What matters here is the sign between the two nodes, not the sign of the single node making the choice of the eigenvalue sign obsolete.
The two operators over all the edges and eigenvalues will form a matrix (better saying, a tensor) where a transformer encoder will learn the positional encoding of dimension $k$.
Here the input embedding will stay of dimention $d$ since we are not concatenating the LPE.
```python
embedding_e = nn.Embedding(num_bond_type, d)
```
The LPE over edges solves the sign-invariance, though it's very expensive on the computation.
# Proposals
* Give a precise overview of nomenclature and formulas between the 3 fundamental papers (Attention Is All You Need, A Generalization of Transformer Networks to Graphs, Rethinking Graph Transformers with Spectral Attention) in order to have a homogenous and easier recap.
*  compare the formulas to briefly explain the various correlation between NLP and Graph processing.