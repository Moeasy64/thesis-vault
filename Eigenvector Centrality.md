The eigenvector centrality is another type of centrality measurement, such as the degree centrality. The objective of any centrality measurement is to give a score of importance to every node based on certain characteristics.

The Eigenvector centrality is based on a circular dependence, this means that every node $v_i$ has a score that depends on the sum of the score of every neighbour $v_j$, and the score of any $v_j$ also depends on the score of $v_i$.
For a node $v_i$ the node centrality is given by
$$
C_E(v_i) = \frac{1}{\lambda}\sum_jA_{ij}C_E(v_j)
$$
Where $\lambda$ is the largest eigenvalue of the adjacency matrix $A$.
The system can be solved with the matrix form that is
$$
\lambda X = A_{ij}X
$$
We take the largest eigenvalue because, for a symmetric matrix of real values, the last eigenvalue has an associated eigenvector of all positive components, this let us have only positive associated scores for the centrality.