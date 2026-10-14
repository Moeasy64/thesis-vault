[[Spectral Geometry Processing with Manifold Harmonics]]
The paper presents a way of processing 3D objects using the Fourier transform. The main idea raise from the concrete relation between sign analysis (its basis function for Fourier transform) and the eigenvalues of the Laplace Matrix.

To be a [[Manifolds]], a mesh must respect two condition:
1. Every edge must be shared exactly by 2 triangles.
2. for every vertex, the set of all neighbor triangles forms just one "fan".

Thanks to this link, we are able to use important operators, such as gradient or laplacian, on meshes.

The **Manifold Harmonics** $\phi_i$ are just the eigenfunction of the laplace-beltrami operator, as said, all the functions so that
$$
\nabla \phi_i = \lambda \phi_i
$$
The first step of the paper is expanding the fourier analysis to manifolds, so that the signal of the mesh can be extracted.
through Fourier analysis, we know that given a square-integrable periodic function with domain defined on $[0,1]$, the function can be expanded into an infinite series of sines and cosines of **increasing frequencies**, that is, the inverse Fourier transform
$$
f(x) = \sum_{k=1}^{\infty} \tilde{f}_k \, H^k(x), \qquad
\begin{cases}
H^{2k}(x) = \sin(kx) \\
H^{2k+1}(x) = \cos(kx)
\end{cases}
$$
where the coefficients $\tilde{f}_k$ is derived by the inner product (dot procuct) between $f$ and $H^k$
$$
\tilde{f}_k = \langle f, H^k \rangle = \int_0^1 f(x)\, H^k(x)\, dx
$$
Meanwhile, $\tilde{f}_k$ represents the Fourier transform
Notice How this reflects the same structure for the Positional Encoding described in [[Attention Is All You Need.pdf]]. Actually here the formula is a bit simplified, but the idea is that every $H^k$ forms an orthonormal basis, meaning we can invert the function (as shown above). The real function would have been
$$
f(x) = \sum_{k=1}^{\infty} \tilde{f}_k \, H^k(x), \qquad
\begin{cases}
H^{2k}(x) = \sin(2\pi kx) \\
H^{2k+1}(x) = \cos(2\pi kx)
\end{cases}
$$
plus a normalization factor.

The set of all $\tilde{f}_k$ is the spectrum: it tells us how much of the function $f(x)$ is present at the corresponding frequency, it is the **energy** present at that frequency.

The key observation is that the Fourier basis functions $H^k$ ($\sin kx$, $\cos kx$) are the eigenfunctions of the operator $-\partial^2/\partial x^2$, with eigenvalues $k^2$.

This suggests generalizing Fourier analysis to a manifold by using as basis the eigenfunctions of its Laplace–Beltrami operator, the natural generalization of the second derivative. To do so we need a Laplacian and an inner product defined on the mesh, which the paper obtains through Discrete Exterior Calculus (via the Hodge star $\star_0$).

On a generic manifold no closed-form expression for these eigenfunctions exists, so the Manifold Harmonic Basis (MHB) must be computed numerically: we assemble the discrete symmetric Laplacian matrix $\bar\Delta$ and compute its eigenvectors $\bar H^k$, then map them back with $H^k = \star_0^{-1/2}\bar H^k$.

The eigendecomposition is the expensive step, and making it scalable is the main computational contribution of the paper (Sec. 4).

to hold an orthonormalized matrix for manifolds, we begin by creating a **discrete laplacian matrix**, which it is not described in the standard basis $\phi_i$, but in an orthonormal basis $\bar\phi_i$; we then compute its eigenvector $\bar H^k$. As last step, we map back the eigenvectors in the standard basis through a special matrix $\star_0$: $H^k=\star_0^{-\frac{1}{2}}\bar{H}^k$