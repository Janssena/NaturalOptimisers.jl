# ─────────────────────────────────────────────────────────────────────────────
# 2D Bayesian logistic regression — posterior comparison (EVON paper, Fig. 1)
#
# A binary logistic regression with a 2-dimensional weight vector, chosen so the
# exact posterior is a *correlated* (tilted) banana. We fit the posterior with each
# optimiser in NaturalOptimisers.jl and overlay the resulting Gaussian approximations
# on the exact posterior:
#
#   • Exact          — optimal full Gaussian variational posterior (deterministic).
#   • EVON           — structured (rotated) full Gaussian; discovers its own eigenbasis.
#   • IVON           — mean-field (axis-aligned) Gaussian.
#   • NaturalDescent — full Gaussian via the Lie-group manifold.
#
# EVON's rotated ellipse aligns with the posterior's principal axes and keeps its mean
# near the mode, whereas IVON's axis-aligned ellipse is forced off the mode. Produces
# scripts/figures/logistic_regression.png.
# ─────────────────────────────────────────────────────────────────────────────

using NaturalOptimisers
using Optimisers
using Random, LinearAlgebra, Statistics
using LogExpFunctions
using StableRNGs
using Plots

# ── data ─────────────────────────────────────────────────────────────────────
# Two correlated 2D features and a handful of points → a genuinely non-diagonal
# posterior (as in Khan et al., 2018, Fig. 2a / Murphy, 2012).
const α = 0.25                      # isotropic prior precision:  p(w) = N(0, α⁻¹ I)

# All datapoints lie on a single line at angle θ (mixed labels straddling the origin). The
# weight component along `u=(cosθ,sinθ)` is then tightly constrained by the data, while the
# perpendicular component is informed only by the prior — giving a strongly *tilted*, elongated
# posterior whose principal axis is `v ⟂ u`. Because the data (hence every gradient) lies in a
# single direction, EVON's discovered eigenbasis is exactly {u, v} and it recovers the exact full
# Gaussian (Cor. 1). This is also the hardest case for IVON's axis-aligned family, so it reproduces
# the paper's Fig. 1 message directly.
function make_data()
    θ = deg2rad(35.0)
    u = [cos(θ), sin(θ)]
    ts = [-2.0, -1.0, 1.0, 2.0]
    X = reduce(vcat, [(t .* u)' for t in ts])
    y = [0.0, 0.0, 1.0, 1.0]
    return X, y
end

# ── exact optimal Gaussian variational posterior (Opper–Archambeau + quadrature) ──
_sigmoid(z) = logistic(z)

function _gauss_expect(f, μ, s2; npts=801)
    s = sqrt(s2)
    zs = range(μ - 9s, μ + 9s; length=npts)
    return sum(f(z) * exp(-(z - μ)^2 / (2s2)) for z in zs) * step(zs) / sqrt(2π * s2)
end

function optimal_gaussian(X, y, α; meanfield=false, iters=600)
    n, d = size(X)
    m = zeros(d)
    Σ = Matrix(1.0 / α * I, d, d)
    for _ in 1:iters
        μ = X * m
        s2 = [dot(view(X, i, :), Σ, view(X, i, :)) for i in 1:n]
        σ̄ = [_gauss_expect(_sigmoid, μ[i], s2[i]) for i in 1:n]
        h̄ = [_gauss_expect(z -> _sigmoid(z) * (1 - _sigmoid(z)), μ[i], s2[i]) for i in 1:n]
        Prec = Symmetric(X' * Diagonal(h̄) * X + α * I)
        Σ = meanfield ? Matrix(Diagonal(1 ./ diag(Prec))) : inv(Prec)
        m = m - Σ * (X' * (σ̄ .- y) + α * m)
    end
    return m, Σ
end

# ── generic training loop over the per-leaf primitives ───────────────────────
function run_method!(rng, rule, state; gradient, nsteps, nsamp)
    dims = size(state.q[1])
    for _ in 1:nsteps
        empty!(state.epsilon)
        for _ in 1:nsamp
            push!(state.epsilon, randn(rng, Float64, dims))
        end
        zs = [NaturalOptimisers.sample(rule, state, s) for s in 1:nsamp]
        dx = [gradient(z) for z in zs]
        state, _ = Optimisers.apply!(rule, state, state.q[1], dx)
    end
    return state
end

# 2σ ellipse (x, y) coordinates for a Gaussian N(m, Σ).
function ellipse(m, Σ; nσ=2.0, npts=200)
    A = cholesky(Symmetric(Matrix(Σ))).L
    t = range(0, 2π; length=npts)
    pts = [m .+ nσ .* (A * [cos(θ), sin(θ)]) for θ in t]
    return getindex.(pts, 1), getindex.(pts, 2)
end

function main()
    X, y = make_data()

    # negative log-likelihood gradient (prior handled via δ for IVON/EVON)
    ∇nll(θ) = X' * (logistic.(X * θ) .- y)
    ∇joint(θ) = ∇nll(θ) .+ α .* θ            # prior folded in (NaturalDescent)

    # references
    m_exact, Σ_exact = optimal_gaussian(X, y, α; meanfield=false)

    # IVON — mean-field
    ivon = IVON(0.05, (0.9, 0.999); delta=α, lambda=1.0, init_scale=1.0)
    st_ivon = run_method!(StableRNG(1), ivon, Optimisers.init(ivon, zeros(2));
        gradient=∇nll, nsteps=8000, nsamp=64)
    m_ivon = st_ivon.q[1]
    Σ_ivon = Diagonal(1 ./ (ivon.lambda .* (st_ivon.q[2] .+ α)))

    # EVON — discovers its own eigenbasis (weight as a 2×1 matrix)
    evon = EVON(0.2, (0.9, 0.99, 0.95); delta=α, zeta=1.0, init_scale=1.0, precond_freq=5)
    st_evon = run_method!(StableRNG(1), evon, Optimisers.init(evon, zeros(2, 1));
        gradient=Θ -> reshape(∇nll(vec(Θ)), 2, 1), nsteps=6000, nsamp=128)
    m_evon = vec(st_evon.q[1])
    V_evon = 1 ./ (evon.zeta .* (st_evon.q[2] .+ α))
    P_evon = kron(st_evon.precond[4], st_evon.precond[3])   # kron(Q_R, Q_L)
    Σ_evon = P_evon * Diagonal(vec(V_evon)) * P_evon'

    # NaturalDescent — full Gaussian on the Lie-group manifold
    nd = NaturalDescent(0.01; tau=1.0, scale=1.0, meanfield=false, manifold=LieGroupManifold())
    st_nd = run_method!(StableRNG(1), nd, Optimisers.init(nd, zeros(2));
        gradient=∇joint, nsteps=8000, nsamp=64)
    m_nd = st_nd.q[1]
    Σ_nd = Matrix(st_nd.q[2]) * Matrix(st_nd.q[2])'

    # ── plot ──────────────────────────────────────────────────────────────────
    # exact (non-Gaussian) log-posterior as a filled-contour backdrop.
    # log-likelihood term for label yᵢ∈{0,1}: yᵢ logσ(fᵢ) + (1-yᵢ) log(1-σ(fᵢ)),
    # with logσ(f) = -log1pexp(-f) and log(1-σ(f)) = -log1pexp(f).
    function logpost(w)
        f = X * w
        ll = sum(@. -y * log1pexp(-f) - (1 - y) * log1pexp(f))
        return ll - α / 2 * sum(abs2, w)
    end
    w1 = range(m_exact[1] - 4.5, m_exact[1] + 4.5; length=220)
    w2 = range(m_exact[2] - 4.5, m_exact[2] + 4.5; length=220)
    Z = [logpost([a, b]) for b in w2, a in w1]

    plt = contourf(w1, w2, Z; levels=25, color=:bone, colorbar=false,
        xlabel="w₁", ylabel="w₂", title="2D Bayesian logistic regression — posterior",
        size=(680, 620), legend=:bottomright)

    for (m, Σ, lab, col, ls, lw) in (
        (m_exact, Σ_exact, "Exact (full Gaussian)", :black, :dot, 5.0),
        (m_evon, Σ_evon, "EVON", :red, :solid, 3.),
        (m_nd, Σ_nd, "NaturalDescent (LieGroups)", :limegreen, :dash, 3.),
        (m_ivon, Σ_ivon, "IVON", :deepskyblue, :solid, 3.),
    )
        ex, ey = ellipse(m, Σ)
        plot!(plt, ex, ey; color=col, lw=lw, linestyle=ls, label=lab)
        scatter!(plt, [m[1]], [m[2]]; color=col, ms=5, markerstrokewidth=0, label="")
    end

    out = joinpath("scripts", "figures", "logistic_regression.png")
    savefig(plt, out)
    println("saved ", out)
    # quick numeric sanity
    println("‖m_evon−m_exact‖ = ", round(norm(m_evon - m_exact), sigdigits=3),
        "   ‖Σ_evon−Σ_exact‖ = ", round(norm(Σ_evon - Σ_exact), sigdigits=3))
    return nothing
end

main()
