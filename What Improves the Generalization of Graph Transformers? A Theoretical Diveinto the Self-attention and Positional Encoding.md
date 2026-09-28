![[what_improvise_generalization_of_graph.pdf]]
This paper theoretically demonstrates that positional embedding improves the generalization. Moreover, the positional embedding can be learned by the network, leading to fewer training
iterations and a smaller sample complexity.
#### (P2). Self-attention helps GTs perform better than Graph convolutional networks. 
We theoretically illustrate that the attention weights, i.e., softmax values of each node in the self-attention module, become increasingly sparse during the training and are concentrated at discriminative nodes. GTs can then learn more distinguishable representations for different classes, outperforming GCNs.
#### (P3) Positional embedding promotes the core neighborhood.
We prove that starting from zero initialization, the positional embedding eventually finds the core neighborhoodand assigns nodes in the core neighborhood with higher weights, which improves the generalization.
#### What Does Self-Attention Improve? A Comparisonwith GCN
We show that the attention weights become concentrated on class-relevant nodes in Lemma 4.5. It increases the distance between output vectors from different classes, which in turn improves the test accuracy. In contrast, Theorem 4.6 shows that without the self-attention layer, GCN requires more iterations and training samples.
#### Lemma 4.5. (Sparse attention map) 
The attention weights for each node become increasingly concentrated on those correlated with class-relevant nodes during the training,
indicates that the outputs of the self-attention layer for all nodes, which are weighted summations of value vectors, evolve in the direction of the class-relevant value features along the training. Then it promotes learning classrelevant features while ignoring other features. Lemma 4.5 is a generalization of Proposition 2 in (Li et al., 2023a), which considers a shallow ViT with one self-attention layer without positional embedding or graph structure. Here, we extend the analysis to node classification on graphs with PE.
#### Theorem 4.6
indicates that without the self-attention layer, the resulting GCN requires more training iterations and samples to achieve the desired generalization, even if the core distance zm is known, and the learning is performed on the core neighborhood only.


This paper could be useful for understanding how to properly check training stats on GTs. Also it offers some hints on how the positional encoding helps in those models, although there is not much about spectral analysis. 