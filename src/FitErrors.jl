# =============================================================================
# FitErrors.jl - parameter uncertainties from the curvature of an objective
# =============================================================================
# Depends on nothing outside the standard library.
#
#     julia> ] dev /Users/noobyte7/Documents/code/julia/FitErrors
#     julia> using FitErrors
#
# See README.md for the manual.
#
# It turns any objective you already minimise into standard deviations, so an
# optimiser that reports nothing but a best point - MPA, PSO, GA, simulated
# annealing, a point picked off a Pareto front - gets error bars on the same
# footing as a Levenberg-Marquardt fit.  Nothing here knows about the
# optimiser.  It needs the objective and the point that came back.
#
# The idea: near a minimum the objective has no linear term, so
#
#     chi2(p) ~ chi2(p_hat) + 1/2 dp' H dp,   H = second derivative matrix
#
# With Gaussian errors L ~ exp(-chi2/2), which matched against a Gaussian
# likelihood gives the covariance
#
#     C = 2 H^-1                 (chi2 / sum of weighted squares)
#     C = H^-1                   (negative log likelihood)
#
# Equivalently: one sigma is where chi2 rises by 1 while the other parameters
# re-minimise.  `profile_interval` measures that directly when you do not trust
# the quadratic picture.
# =============================================================================

module FitErrors

using LinearAlgebra
using Printf

export fit_errors, error_table, profile_interval

# -----------------------------------------------------------------------------
# internals
# -----------------------------------------------------------------------------

_finite(x) = isa(x, Real) && isfinite(x)

"""
    _steps(p, lower, upper, rel)

Per-parameter finite-difference step.  Scaled by the parameter itself, or by
the bound span when the parameter sits near zero, then clipped so a step never
leaves the feasible box.
"""
function _steps(p, lower, upper, rel)
    k = length(p)
    h = zeros(k)
    for i in 1:k
        span = (lower === nothing || upper === nothing) ? 1.0 :
               (isfinite(lower[i]) && isfinite(upper[i]) ? upper[i] - lower[i] : 1.0)
        base = max(abs(p[i]), span / 100, 1e-12)
        room = min(lower === nothing ? Inf : p[i] - lower[i],
                   upper === nothing ? Inf : upper[i] - p[i])
        h[i] = min(rel * base, isfinite(room) ? 0.5 * room : Inf)
        h[i] > 0 || (h[i] = rel * base)
    end
    return h
end

"""
    _hessian(f, p, h, free)

Central-difference Hessian over the `free` indices.  Returns `nothing` if any
evaluation was not finite, since a single bad point poisons the whole matrix.
"""
function _hessian(f, p::Vector{Float64}, h::Vector{Float64}, free::Vector{Int})
    n = length(free)
    H = zeros(n, n)
    ok = true
    ev(q) = (v = f(q); _finite(v) || (ok = false); Float64(v))

    f0 = ev(p)
    for (a, i) in enumerate(free)
        q = copy(p)
        q[i] = p[i] + h[i]; fp = ev(q)
        q[i] = p[i] - h[i]; fm = ev(q)
        H[a, a] = (fp - 2 * f0 + fm) / h[i]^2
        for (b, j) in enumerate(free)
            b < a || continue
            q = copy(p)
            q[i] = p[i] + h[i]; q[j] = p[j] + h[j]; fpp = ev(q)
            q[j] = p[j] - h[j];                     fpm = ev(q)
            q[i] = p[i] - h[i]; q[j] = p[j] + h[j]; fmp = ev(q)
            q[j] = p[j] - h[j];                     fmm = ev(q)
            H[a, b] = H[b, a] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
        end
    end
    return ok ? H : nothing
end

"""
    _gradient(f, p, h, free)

Central-difference gradient.  Used only as a diagnostic: the whole method
assumes the gradient vanishes, which is worth verifying for a point that came
from a global optimiser rather than a local one.
"""
function _gradient(f, p::Vector{Float64}, h::Vector{Float64}, free::Vector{Int})
    g = zeros(length(p))
    for i in free
        q = copy(p)
        q[i] = p[i] + h[i]; fp = Float64(f(q))
        q[i] = p[i] - h[i]; fm = Float64(f(q))
        g[i] = (fp - fm) / (2 * h[i])
    end
    return g
end

_at_bound(p, lower, upper, i) =
    (lower !== nothing && isfinite(lower[i]) &&
     isapprox(p[i], lower[i]; rtol = 1e-6, atol = 1e-12)) ||
    (upper !== nothing && isfinite(upper[i]) &&
     isapprox(p[i], upper[i]; rtol = 1e-6, atol = 1e-12))

# -----------------------------------------------------------------------------
# the main entry point
# -----------------------------------------------------------------------------

"""
    fit_errors(f, p; lower, upper, ndata, kind, step, fixed, names)

Standard deviations of the parameters `p` from the curvature of `f` at that
point.  `f(p) -> Real` is the objective that was minimised.

Keywords:

- `lower`, `upper`  bounds, if the fit had any.  A parameter sitting on a bound
                    is held fixed and reports `NaN`: its uncertainty is not
                    defined there, and the right response is a wider bound.
- `ndata`           number of data points, needed for `std_scaled` and the
                    reduced chi-squared.
- `kind`            `:chi2` (default) when `f` is a sum of weighted squared
                    residuals, `:neglogl` when it is a negative log likelihood.
- `step`            relative first-pass step, default `1e-3`.  A second pass
                    re-differences with steps of one sigma, where the quadratic
                    approximation is most accurate and the difference is best
                    conditioned.
- `fixed`           extra indices (or a Bool vector) to hold fixed.
- `names`           parameter names, carried through for `error_table`.

Returns a NamedTuple with `std`, `std_scaled`, `covariance`, `correlation`,
`hessian`, `gradient`, `stationarity`, `chi2`, `ndata`, `dof`,
`reduced_chi2`, `fixed`, `names`, `value`.

`std` assumes the data sigmas are absolute.  `std_scaled` multiplies it by
`sqrt(chi2/dof)`, which is the usual choice when the sigmas are relative or the
reduced chi-squared is far from 1, and is unchanged if the objective is
rescaled by a constant.

`stationarity` is `max |dchi2/dp_i| * sigma_i`, the drop in the objective from
moving one sigma downhill.  It must be small compared with 1 for any of this to
mean anything.  A local optimiser gives ~1e-6; a large value means the point is
not a minimum of *this* objective, which is the thing to check for a point
taken off a multi-objective front.

```julia
chi2(p) = sum(((model(x, p) .- y) ./ sigma) .^ 2)
best = my_global_optimiser(chi2, lb, ub)
err  = fit_errors(chi2, best; lower = lb, upper = ub, ndata = length(y))
error_table(err)
```
"""
function fit_errors(f, p::AbstractVector{<:Real};
                    lower = nothing, upper = nothing,
                    ndata = nothing, kind::Symbol = :chi2,
                    step::Real = 1e-3, fixed = Int[], names = nothing)
    pv = Float64.(collect(p))
    k = length(pv)
    factor = kind === :chi2 ? 2.0 : kind === :neglogl ? 1.0 :
             error("kind must be :chi2 or :neglogl")

    held = falses(k)
    if isa(fixed, AbstractVector{Bool})
        length(fixed) == k || error("fixed mask must match the parameter length")
        held .= fixed
    else
        for i in fixed; held[i] = true; end
    end
    for i in 1:k
        _at_bound(pv, lower, upper, i) && (held[i] = true)
    end
    free = [i for i in 1:k if !held[i]]
    isempty(free) && error("every parameter is fixed or sitting on a bound")

    chi2 = Float64(f(pv))
    _finite(chi2) || error("the objective is not finite at this point")

    h = _steps(pv, lower, upper, step)
    std = fill(NaN, k)
    cov = fill(NaN, k, k)

    function solve(H)
        H === nothing && return nothing
        C = try
            factor .* inv(Symmetric(H))
        catch
            return nothing
        end
        (all(isfinite, C) && all(>(0), diag(C))) || return nothing
        return C
    end

    hess = fill(NaN, k, k)
    function keep!(H)
        H === nothing && return
        for (a, i) in enumerate(free), (b, j) in enumerate(free)
            hess[i, j] = H[a, b]
        end
    end

    H1 = _hessian(f, pv, h, free)
    keep!(H1)
    C = solve(H1)
    if C !== nothing
        # Re-differencing with one-sigma steps keeps the objective difference of
        # order 1, which is well conditioned even when chi2 itself is large.
        for (a, i) in enumerate(free)
            room = min(lower === nothing ? Inf : pv[i] - lower[i],
                       upper === nothing ? Inf : upper[i] - pv[i])
            h[i] = min(sqrt(C[a, a]), isfinite(room) ? 0.5 * room : Inf)
        end
        H2 = _hessian(f, pv, h, free)
        C2 = solve(H2)
        if C2 !== nothing
            keep!(H2)
            C = C2
        end
    end

    if C === nothing
        @warn """
        the objective is not locally quadratic here, so there are no errors.
        Either this is not a minimum, or a finite-difference step left the
        valid region. Check `stationarity`, and widen any active bound."""
    else
        for (a, i) in enumerate(free), (b, j) in enumerate(free)
            cov[i, j] = C[a, b]
        end
        for i in free
            std[i] = sqrt(cov[i, i])
        end
    end

    grad = _gradient(f, pv, h, free)
    stat = maximum([abs(grad[i]) * (isnan(std[i]) ? 0.0 : std[i]) for i in free];
                   init = 0.0)

    dof = ndata === nothing ? nothing : Int(ndata) - length(free)
    scale = (dof === nothing || dof <= 0) ? NaN : sqrt(chi2 / dof)
    corr = [isfinite(cov[i, j]) ? cov[i, j] / (std[i] * std[j]) : NaN
            for i in 1:k, j in 1:k]

    return (value = pv, std = std, std_scaled = std .* scale,
            covariance = cov, correlation = corr, hessian = hess, gradient = grad,
            stationarity = stat, chi2 = chi2, ndata = ndata, dof = dof,
            reduced_chi2 = dof === nothing ? NaN : chi2 / dof,
            fixed = held, names = names)
end

# -----------------------------------------------------------------------------
# reporting
# -----------------------------------------------------------------------------

"""
    error_table(res; io=stdout, correlations=true, threshold=0.7)

Print the result of [`fit_errors`](@ref): value, both error scales, and the
parameter pairs whose correlation exceeds `threshold` in magnitude, since those
are the ones whose individual errors mean the least.
"""
function error_table(res; io::IO = stdout, correlations::Bool = true,
                     threshold::Real = 0.7)
    k = length(res.value)
    names = res.names === nothing ? ["p$i" for i in 1:k] : string.(res.names)
    fmt(x) = rpad(@sprintf("%.6g", x), 14)

    println(io, "  ", rpad("parameter", 20), rpad("value", 14), rpad("std", 14),
            rpad("std (scaled)", 14))
    for i in 1:k
        println(io, "  ", rpad(names[i], 20), fmt(res.value[i]), fmt(res.std[i]),
                fmt(res.std_scaled[i]), res.fixed[i] ? " fixed / at bound" : "")
    end

    if res.dof !== nothing
        @printf(io, "  chi2 = %.6g, %d points, %d dof, reduced chi2 = %.4g\n",
                res.chi2, res.ndata, res.dof, res.reduced_chi2)
    end
    @printf(io, "  stationarity = %.2e  (must be << 1)\n", res.stationarity)
    res.stationarity > 0.1 &&
        println(io, "  WARNING: this does not look like a minimum of this objective")

    if correlations
        pairs = [(res.correlation[i, j], names[i], names[j])
                 for i in 1:k for j in 1:(i - 1)
                 if isfinite(res.correlation[i, j]) &&
                    abs(res.correlation[i, j]) >= threshold]
        if !isempty(pairs)
            println(io, "  strong correlations:")
            for (c, a, b) in sort(pairs; by = t -> -abs(t[1]))
                @printf(io, "    %-18s %-18s %+.3f\n", a, b, c)
            end
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# profile likelihood, for when the quadratic picture is doubtful
# -----------------------------------------------------------------------------

"""
    profile_interval(f, p, i; sigma, level=1.0, minimizer=nothing,
                     lower=nothing, upper=nothing, fixed=Int[])

The interval in parameter `i` over which `f` rises by `level` above its value
at `p`, found by bracketing and bisection.  This is the definition the
curvature method approximates, so it is the check to run when a valley is
banana-shaped or an error looks too good.  `level = 1` is one sigma,
`3.84` is 95 per cent for one parameter.

`minimizer(g, q0, lo, hi) -> Real` re-minimises the *other* free parameters at
each trial value and returns the minimum.  Without it the others stay frozen,
which gives the conditional (too narrow) interval instead of the marginal one.

```julia
using Optim
mini(g, q0, lo, hi) = Optim.minimum(optimize(g, lo, hi, q0, Fminbox(LBFGS()),
                                             Optim.Options(iterations = 500);
                                             autodiff = :finite))
lo, hi = profile_interval(chi2, best, 2; sigma = err.std[2], minimizer = mini,
                          lower = lb, upper = ub)
```
"""
function profile_interval(f, p::AbstractVector{<:Real}, i::Integer;
                          sigma::Real, level::Real = 1.0, minimizer = nothing,
                          lower = nothing, upper = nothing, fixed = Int[])
    pv = Float64.(collect(p))
    k = length(pv)
    held = falses(k)
    isa(fixed, AbstractVector{Bool}) ? (held .= fixed) : (for j in fixed; held[j] = true; end)
    for j in 1:k
        _at_bound(pv, lower, upper, j) && (held[j] = true)
    end
    others = [j for j in 1:k if j != i && !held[j]]

    function value(x)
        q = copy(pv); q[i] = x
        (minimizer === nothing || isempty(others)) && return Float64(f(q))
        g = z -> (r = copy(q); r[others] .= z; Float64(f(r)))
        lo = lower === nothing ? fill(-Inf, length(others)) : Float64.(lower[others])
        hi = upper === nothing ? fill(Inf, length(others)) : Float64.(upper[others])
        return Float64(minimizer(g, pv[others], lo, hi))
    end

    base = value(pv[i])

    function edge(dir)
        limit = dir < 0 ? (lower === nothing ? -Inf : lower[i]) :
                          (upper === nothing ? Inf : upper[i])
        inside = pv[i]          # last point known to be below `level`
        x = pv[i]
        stepsize = Float64(sigma)
        for _ in 1:12
            x = pv[i] + dir * stepsize
            if (dir < 0 && x < limit) || (dir > 0 && x > limit)
                x = limit
                if value(x) - base < level
                    @warn "profile hits the bound on parameter $i before Delta = $level"
                    return NaN
                end
                break
            end
            value(x) - base >= level && break
            inside = x
            stepsize *= 2
        end
        if value(x) - base < level
            @warn "could not bracket Delta = $level for parameter $i"
            return NaN
        end
        a, b = inside, x
        for _ in 1:40
            m = 0.5 * (a + b)
            value(m) - base < level ? (a = m) : (b = m)
            abs(b - a) < 1e-4 * sigma && break
        end
        return 0.5 * (a + b)
    end

    return (lo = edge(-1), hi = edge(+1))
end

end # module
