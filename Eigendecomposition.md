$L = V\Lambda V^\top$ or sometimes written as $L=V diag(\lambda)V^\top$ for a positive semi-definite matrix $L$ (Laplacian for us) is the eigendecomposition of the matrix that let us extract eigenvalues $\lambda$ and eigenvectors $V$ (its columns) of the matrix.
The eigendecomposition plays two different roles: one for the Message Passing in the MPNN and one for the positional encodings for graphs.
## Message Passing
When it is used in message passing we can use the eigendecomposition as follows:
Let's consider a graph $\mathcal{G}(V,E)$ with $n$ nodes having feature dimension $d$.
* create a feature matrix $X$ of dimension $\mathbb{R^{n \times d}}$
* take the (normalized) Laplacian matrix
* calculate its eigendecomposition $L = V\Lambda V^\top$

If we compute $LX= (V\Lambda V^\top)X$ we are measuring the local variance of each node, meaning how much the node features are different from the one of its neighbors. This grants to the network the ability of update the node feature embedding with the one of its neighbours. To have a clear view of how it work let's just watch for a single row of the resulting matrix:
$$
LX_i = d_iX_i -\sum_{j \in \mathcal{N}(i)} X_j 
$$
where d_i is the degree of the node $i$.

Actually here, we don't have the need of calculate the eigendecomposition, but gives us some insight about the functioning of the laplacian applied to the node feature. Let's see the "spectral meaning" of this operation:
Let's assume we just use a single node embedding $x$ instead of the matrix $X$,
* $V^\top x$ is the projection (change of basis) of  the original signal $x$ into the space of the eigenvector $V$. This operation is called **Graph Fourier Transform**, we are now working in the spectral field.
* $\Lambda (V^\top x)$ is the filtering (which is a fancy way of calling the linear "scaling") of each component of $x$ based on its corresponding eigenvalue. So we are moduling (amplifying or attenuate) the spectral frequencies of $x$ in the new domain
* $V(\dots)$ is the reconstruction of $x$ in its canonical basis, now weighted based on the eigenvalues.
## Positional Encoding
Here we do need to explicitly use the eigendecomposition. We are not interested in the $X$ feature matrix for this scope.
* take the (normalized) Laplacian matrix
* calculate its eigendecomposition $L = V \Lambda V^\top$
* keep just the matrix $V \in \mathbb{R}^{n \times n}$
* Usually we are not interested in every single eigenvector, we can just take the $k$-th smallest eigevalues and keep the corresponding eigenvectors, excluding the first one (which, for a fully-connected graph is 0). Essentially we are just keeping the small frequencies. We get a $V_k \in \mathbb{R}^{n \times k}$ matrix.
The $i$-th row is the positional encoding for the node $i$
$$
PE(v_i) = [V_{i,1}, V_{i,2} \dots V_{i,k}]
$$
Note that $V_{i,1}$ is the value associated the second smallest eigenvector, we are excluding the constant first eigenvector, because its eigenvalue is 0.
### correlation between Euclidean distance and random walks
One important thing to note is that if we compute the euclidean distance between $PE(v_i)$ and $PE(v_j)$ we get a small value if the nodes are "close" in the graph, or a big one in the opposite case, that is because the positional encoding reflects the random walk between the node $i$ and the node $j$.
The reason why the Euclidean distance between $PE(v_i)$ and $PE(v_j)$ reflects the graph structure is deeply rooted in the concept of **Commute Time Distance**. 
The commute time $c(i, j)$ is the expected number of steps a random walk takes to travel from node $i$ to node $j$ and back. According to Spectral Graph Theory, this distance can be computed using the eigenvalues $\lambda$ and eigenvectors $V$ of the Laplacian:

$$ c(i, j) = \text{Vol}(G) \sum_{k=2}^{n} \frac{1}{\lambda_k} (V_{i, k} - V_{j, k})^2 $$

*(where $\text{Vol}(G)$ is the volume of the graph, i.e., the sum of all node degrees).*

Notice the term $(V_{i, k} - V_{j, k})^2$. This is exactly the squared Euclidean distance between the $k$-th components of our positional encodings. This formula tells us two critical things:
1. **The Euclidean Link:** If we scale our eigenvectors by $1/\sqrt{\lambda_k}$, the Euclidean distance between two nodes in this new PE latent space *is exactly* proportional to their random walk commute time. Nodes that are structurally "close" (many paths between them, highly connected) have a small commute time, meaning their PEs will be close in the Euclidean space.
2. **The Low-Frequency Filter:** The term $1/\lambda_k$ gives a massive weight to the smallest eigenvalues (low frequencies) and almost ignores the large ones. This mathematically justifies why we only keep the $k$-th smallest eigenvectors: they contain almost all the necessary information to approximate the global distance between nodes.

At the end, before feeding the input to the network we need to add the positional encoding to that input, this usually translates into concatenate the node input feature and the PE:
$$
\hat{X_i} = X_i \oplus PE(v_i)
$$
Doing so, we can expand the latent space of the feature with important positional encoding information as new coordinates.