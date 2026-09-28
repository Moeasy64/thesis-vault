---
Type: Paper
---
![[Papers/rethinking_graph_transformers.pdf]]
# Objective
Extending the capability of transformers to non-sequential data through the **Spectral Attention Network (SAN)**.

It exploit the **Learned Positional Encoding (LPE)** in order to globally acquire the position of every node inside a graph.

The model use LPE to extract and acquire the positional informations from the **[[Laplacian Spectrum]]**.

This LPE is then added to the features of the graph and passed to a [[fully-connected transformer]].
# Improvements
* Surpassing the typical [[over-squashing]] and [[oversmoothing]] symptom of GNNs.
* Performs better or equal than every state of the art attention based architecture and GNN on graphs.
* First fully-connected model that performs good on graphs
* Exploits the full laplacian spectrum to learn the position of every node

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


# Proposals
* Give a precise overview of nomenclature and formulas between the 3 fundamental papers (Attention Is All You Need, A Generalization of Transformer Networks to Graphs, Rethinking Graph Transformers with Spectral Attention) in order to have a homogenous and easier recap.
*  compare the formulas to briefly explain the various correlation between NLP and Graph processing.