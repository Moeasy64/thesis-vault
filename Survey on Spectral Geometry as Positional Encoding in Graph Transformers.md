
## Foundational Paper (Starting Point)

**Kreuzer et al. (2021) — "Rethinking Graph Transformers with Spectral Attention" (SAN)** → https://arxiv.org/abs/2106.03893

Introduces the **Spectral Attention Network (SAN)**, which uses a Learned Positional Encoding (LPE) that can leverage the full Laplacian spectrum to learn the position of each node in a given graph, becoming the first fully-connected architecture to perform well on graph benchmarks. The key contribution is the observation that the eigenvectors of the graph Laplacian are the natural analogue of the sinusoidal functions used in the original Transformer: lower eigenvalues carry more weight in determining distances between nodes (analogous to low frequencies), and the model can in principle exploit all $m \leq N$ eigenvectors.

---

## Line 1 — Sign and Basis Invariance of Eigenvectors

The central problem with Laplacian eigenvectors is that if $v$ is an eigenvector, so is $-v$, and in degenerate eigenspaces there exist infinitely many equivalent basis choices.

### Lim et al. (2022) — "Sign and Basis Invariant Networks for Spectral Graph Representation Learning" (SignNet/BasisNet)

→ https://arxiv.org/abs/2202.13013

Introduces SignNet and BasisNet, neural architectures invariant to the two key symmetries of eigenvectors: sign flips, and more general basis symmetries arising in higher-dimensional eigenspaces. The networks are provably universal — they can approximate any continuous function of eigenvectors with the desired invariances. When used with Laplacian eigenvectors, they are provably more expressive than existing spectral methods on graphs: they subsume all spectral graph convolutions and previously proposed graph positional encodings as special cases.

---

### Wang et al. (2022) — "Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks" (PEG)

→ https://arxiv.org/abs/2203.00199

Proposes PEG, a class of GNN layers that uses separate channels to update original node features and positional features. PEG simultaneously imposes permutation equivariance on the original node features and $O(p)$-equivariance (orthogonal group) on the positional features, where $p$ is the dimension of the positional features used. The contribution with respect to Kreuzer is a formally guaranteed solution for stability and generalizability to unseen graphs at inference time.

---

## Line 2 — Stability of Spectral Encodings

### Huang et al. (2024) — "On the Stability of Expressive Positional Encodings for Graphs" (SPE)

→ https://arxiv.org/abs/2310.02579

Addresses two fundamental challenges of using Laplacian eigenvectors as positional encodings: (1) **Non-uniqueness** — there are many different eigendecompositions of the same Laplacian, and (2) **Instability** — small perturbations to the Laplacian can result in completely different eigenspaces, leading to unpredictable changes in positional encoding. The root cause of instability is identified as a "hard partition" of eigenspaces. SPE instead introduces a **soft and learnable partition** of eigenspaces in an eigenvalue-dependent manner, simultaneously achieving stability (from the soft partition) and expressivity. This paper is particularly relevant for understanding why the Fiedler value ($\lambda_2$) is unstable in the presence of degenerate or near-degenerate eigenvalues.

---

## Line 3 — Full Spectrum vs. Partial Spectrum

### Bo et al. (2023) — "Specformer: Spectral Graph Neural Networks Meet Transformers"

→ https://arxiv.org/abs/2303.01028

Introduces Specformer, which effectively encodes the full set of eigenvalues and performs self-attention in the spectral domain, yielding a learnable set-to-set spectral filter. A decoder with learnable bases is also designed to enable non-local graph convolution. Specformer is equivariant to node permutation. The key difference with respect to Kreuzer is that classical spectral filters are scalar-to-scalar functions — mapping a single eigenvalue to a single filtered value — thereby ignoring the global pattern of the spectrum. Specformer instead applies self-attention over the entire set of $N$ eigenvalues after positional encoding to construct learnable filter bases.

---

## Line 4 — Unified GPS Framework (Laplacian PE + Local MPNN)

### Rampášek et al. (2022) — "Recipe for a General, Powerful, Scalable Graph Transformer" (GraphGPS)

→ https://arxiv.org/abs/2205.12454

Proposes a recipe for building a General, Powerful, Scalable (GPS) Graph Transformer with linear complexity and state-of-the-art results across a diverse set of benchmarks. It categorizes positional and structural encodings as local, global, or relative, and proposes the first architecture with $O(N+E)$ complexity by decoupling local real-edge aggregation from the fully-connected global Transformer. LapPE (Laplacian eigenvectors) and RWSE (random walk structural encodings) are included as modular, interchangeable PE components. This is the work that systematized and unified the field following Kreuzer.

---

## Line 5 — Directed Graphs and the Magnetic Laplacian

### Geisler et al. (2023) — "Transformers Meet Directed Graphs"

→ https://proceedings.mlr.press/v202/geisler23a/geisler23a.pdf

Observes that most of the literature on spectral positional encodings symmetrizes the graph in order to use the standard combinatorial Laplacian, potentially discarding essential directional information. Proposes two direction-aware positional encodings: (1) eigenvectors of the **Magnetic Laplacian**, which naturally generalizes the combinatorial Laplacian to directed graphs with complex-valued eigenvectors, and (2) a variant inspired by SignNet and Kreuzer to process these complex eigenvectors in a sign- and basis-invariant manner.

---

## Line 6 — Fiedler Value and Spectral Structure

Regarding the **Fiedler value ($\lambda_2$):** no paper in the Graph Transformer literature studies it in isolation as a positional encoding, but its relevance emerges from two angles. The Fiedler value — the first non-zero eigenvalue of the graph Laplacian, also known as algebraic connectivity — can predict GCN performance and inform the choice of the number of layers: a low Fiedler value signals a graph that is difficult to aggregate over with deep message passing. In the context of spectral encodings, its instability in graphs with near-degenerate eigenvalues is precisely the problem addressed by SPE (Huang et al., 2024). Kreuzer himself observes that lower eigenvalues carry more weight in determining inter-node distances, which implies that the Fiedler vector is structurally the most informative spectral positional encoding, but also the most sensitive to perturbations.

---

## Chronological Summary

|Year|Paper|Key Contribution|
|---|---|---|
|2021|**SAN** (Kreuzer et al.)|First use of the full Laplacian spectrum as a learnable LPE|
|2022|**SignNet/BasisNet** (Lim et al.)|Sign and basis invariance, provable universality|
|2022|**PEG** (Wang et al.)|Stability + $O(p)$-equivariance guarantees|
|2022|**GraphGPS** (Rampášek et al.)|Unified modular framework (LapPE + MPNN + global attention)|
|2023|**Specformer** (Bo et al.)|Self-attention in the spectral domain over eigenvalues|
|2023|**MagLap** (Geisler et al.)|Extension to directed graphs via the Magnetic Laplacian|
|2024|**SPE** (Huang et al.)|Simultaneous stability and expressivity via soft eigenspace partition|