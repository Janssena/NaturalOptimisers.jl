# NaturalOptimisers.jl

[![Julia Version](https://img.shields.io/badge/Julia-1.10%2B-9558B2?style=flat-square&logo=julia)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-Passed-success?style=flat-square)](https://github.com/alexander/Postdoc/Projects/NaturalOptimisers.jl)

`NaturalOptimisers.jl` is a Julia package for Natural Gradient Descent (NGD) optimization on Gaussian variational distributions. Built to integrate with the `Optimisers.jl` ecosystem, it provides a unified interface to perform variational inference and natural gradient updates across various geometric manifolds.

An architectural advantage of this library is its ability to **turn any parameter object supported by `Optimisers.jl` (including model weights, structs, or neural networks) into a set of variational parameters by storing the variational distribution state $q$ directly inside the optimizer's state object.** This design enables plug-and-play integration: existing Julia models can be transitioned into a variational setup transparently without any modification to the model's structural parameters.

## Features

* **Unified Interface**: Integrates directly with the `Optimisers.jl` ecosystem.
* **Manifold Representations**: Supports three distinct geometric manifolds:
  * `LieGroupManifold`: Scale parameters represented via Cholesky factorizations; updates performed using matrix exponentials to preserve positive-definiteness (Lin et al. 2020, Shabo et al. 2024).
  * `RiemannianManifold`: Natural gradients along Riemannian geodesics using mean and precision parameters (Lin et al. 2021).
  * `EuclidianManifold`: Parameterized Euclidean coordinate updates using change-of-variable mappings (Khan et al. 2018).
* **Variational Distributions**:
  * `DiagNormal`: Diagonal covariance (mean-field) Gaussian approximations.
  * `FullNormal`: Full-covariance Gaussian representations.
* **Automatic Differentiation**: Fully compatible with `ForwardDiff.jl`.

## Mathematical Foundation

In variational inference, we minimize a variational objective (such as the Negative ELBO) with respect to a distribution $q_{\theta}(z)$:
$$\mathcal{L}(\theta) = \mathbb{E}_{q_{\theta}}[\ell(z)] - \tau \mathbb{H}(q_{\theta})$$

where $\ell(z) = -\log p(y, z)$ represents the negative log-joint (unnormalized negative log-posterior / potential) of the target distribution, and $\tau \in [0, 1]$ is a **temperature parameter** (or entropy scaling factor):
* **$\tau = 1.0$**: Corresponds to standard Bayesian variational inference (minimizing the exact Negative ELBO).
* **$\tau = 0.0$**: Ignores the entropy term entirely, turning the variational update into standard Maximum A Posteriori (MAP) optimization where the distribution collapses to the mode.
* **$0.0 < \tau < 1.0$**: Corresponds to tempered variational inference, allowing a controlled trade-off between the data likelihood and the entropy of the variational posterior.

Unlike standard gradient descent, which takes steps in the Euclidean space of coordinates $\theta$, Natural Gradient Descent takes steps along the steepest descent direction with respect to the Fisher Information Matrix (FIM) $F(\theta)$:
$$\theta_{t+1} = \theta_t - \eta F(\theta)^{-1} \nabla_{z} \mathcal{L}(\theta)$$

`NaturalOptimisers.jl` implements these updates implicitly using reparameterized sample-level gradients $\nabla_z \ell(z_i)$ under different manifolds, bypassing the need to construct or invert the full Fisher Information Matrix explicitly.

## Natural Momentum

The optimizer supports **Natural Momentum** (an Adam-like formulation in the natural gradient space). Instead of accumulating moving averages on the standard Euclidean gradients, the exponential moving averages are computed directly in the natural gradient space (the tangent space of the manifold):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \tilde{\nabla} \theta$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \tilde{\nabla}^2 \theta$$

Using the bias-corrected estimates $\hat{m}_t = m_t / (1 - \beta_1^t)$ and $\hat{v}_t = v_t / (1 - \beta_2^t)$, the updates are then projected onto their respective manifolds. This ensures that the momentum directions respect the underlying geometry of the probability space rather than being distorted by coordinate representation curvature.

<!-- ## Installation

This package can be installed directly from its repository or developed locally:

```julia
using Pkg
Pkg.develop(path="/path/to/NaturalOptimisers.jl")
``` -->

## Quick Start

The following example demonstrates how to initialize, sample, and update a diagonal Gaussian variational distribution under the `LieGroupManifold`:

```julia
using NaturalOptimisers
using Optimisers
using Random

rng = Random.MersenneTwister(1234)

# Define optimization rule (learning rate η = 0.05, temperature τ = 1.0)
rule = NaturalDescent(0.05; tau=1.0, meanfield=true, manifold=LieGroupManifold())

# Initialize variational parameters (mean and standard deviation)
m = [1.0, -0.5]
σ = [0.5, 0.8]

# Initialize state and bind parameters
state = Optimisers.init(rule, m)
state = (q=(m, σ), momentum=state.momentum, epsilon=[randn(rng, 2) for _ in 1:100])

# Sample from the variational distribution
z = [NaturalOptimisers.sample(rule, state, i) for i in 1:100]

# Compute sample-level Euclidean gradients of the loss function
dx = gradient(loss, z)

# Compute natural gradients
∇̃m, ∇̃σ = natgrad(rule, state, dx)

# Perform parameter update
updated_m, updated_σ = update(rule, m, σ, ∇̃m, ∇̃σ)
```

## Supported Manifolds

### `LieGroupManifold`
Represents the scale parameter via the Cholesky factor $L$ (or diagonal standard deviation $\sigma$). The updates are performed using multiplicative matrix exponentials (or element-wise exponentials for `DiagNormal`) to ensure positive-definiteness:
$$L_{t+1} = L_t \exp(-\eta U)$$

where

$$U = L^T \cdot \mathbb{E}_{q_\theta}[\nabla_z \ell(z) \cdot \epsilon^T] - \tau I$$

### `RiemannianManifold`
Performs natural gradient descent directly along the Riemannian manifold of Gaussian distributions. Covariance parameters are updated using precision matrices $S = \Sigma^{-1}$ along Riemannian geodesics with second-order corrections:
$$S_{t+1} = S_t - \eta \hat{G} + \frac{\eta^2}{2} \hat{G} \Sigma \hat{G}$$
where $$\hat{G} = -2 \nabla \Sigma \mathcal{L}$$

### `EuclidianManifold`
Performs standard Euclidean updates in a projected coordinate space. Uses inverse-link functions (such as $\phi = \text{softplus}^{-1}(\sigma)$) to enforce scale constraints under standard projected Euclidean Fisher metrics.


## Testing

An extensive test suite is provided to verify mathematical correctness and AD compatibility.

To run the tests:
```bash
julia --project=test test/runtests.jl
```

The test suite validates:
1. **Closed-Form Convergence**: Checks natural gradients against analytical mathematical expectations under Quadratic, Linear, and Exponential losses in 1D.
2. **2D FullNormal Verification**: Validates covariance updates on full covariance matrices for all three manifolds.
3. **ForwardDiff AD Verification**: Compares sample-average natural gradients with exact parameter-level natural gradients calculated using `ForwardDiff` AD.
4. **Monte Carlo FIM Verification**: Confirms that empirical natural gradients derived from the empirical Monte Carlo score-covariance Fisher Information Matrix match the library's updates within a tight $2\%$ tolerance.
