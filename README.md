# FitErrors.jl

Parameter uncertainties from the curvature of an objective function, for fits
whose optimiser does not provide any.

Global optimisers — MPA, particle swarm, genetic algorithms, simulated
annealing — return a best point and nothing else. That is often read as "these
methods cannot give error bars". They can: the uncertainty was never a property
of the optimiser, it is a property of the shape of chi-squared around the
minimum. This package measures that shape directly, so any optimiser gets error
bars on the same footing as a Levenberg–Marquardt fit.

Depends only on `LinearAlgebra` and `Printf`.

## Install

```julia
] dev /Users/noobyte7/Documents/code/julia/FitErrors
```

Then `using FitErrors` from any project. To remove it again, `] rm FitErrors`.

## Quick start

The only contract is that your objective takes a vector and returns a number,
and that the point you pass is a minimum of *that* function.

```julia
using FitErrors

chi2(p) = sum(((model(x, p) .- y) ./ sigma) .^ 2)

best = my_optimiser(chi2, lb, ub)        # MPA, Optim, PSO, by hand, anything

err = fit_errors(chi2, best; lower = lb, upper = ub,
                 ndata = length(y), names = [:amplitude, :centre, :width])
error_table(err)
```

```
  parameter           value         std           std (scaled)
  amplitude           3.51204       0.0412        0.0658
  centre              11.1687       0.00391       0.00625
  width               1.24          0.0233        0.0372
  chi2 = 412.6, 180 points, 177 dof, reduced chi2 = 2.331
  stationarity = 3.10e-07  (must be << 1)
```

## Which sigma do I quote?

`fit_errors` reports two, and the difference is not cosmetic.

| | meaning | use when |
|---|---|---|
| `std` | assumes your `sigma` are absolute and correct, and every weight is 1 | you trust your error bars in an absolute sense |
| `std_scaled` | `std * sqrt(chi2/dof)` | reduced chi-squared is far from 1, your sigmas are relative, or your objective carries arbitrary weights |

`std_scaled` is what `scipy.optimize.curve_fit` and gnuplot report by default.
It amounts to assuming the model is right and the sigmas were off by a constant
factor. It is also unchanged if you rescale the whole objective, which `std` is
not — so if your chi-squared has hand-tuned target weights, `std_scaled` is the
only one of the two that means anything.

If the reduced chi-squared is very large (say above ~5), neither is a rigorous
68 % confidence interval: the model is not describing the data within the
stated uncertainties. Quote `std_scaled`, and say how you got it.

## The API

### `fit_errors(f, p; ...)`

| keyword | default | purpose |
|---|---|---|
| `lower`, `upper` | `nothing` | bounds used in the fit. A parameter on a bound is held fixed and reports `NaN` |
| `ndata` | `nothing` | number of data points; needed for `std_scaled` and `dof` |
| `kind` | `:chi2` | `:chi2` for a sum of weighted squared residuals, `:neglogl` for a negative log likelihood |
| `step` | `1e-3` | relative first-pass finite-difference step |
| `fixed` | `Int[]` | extra indices, or a `Bool` vector, to hold fixed |
| `names` | `nothing` | parameter names, carried into `error_table` |

`kind` matters: the covariance is `2H⁻¹` for a chi-squared but `H⁻¹` for a
negative log likelihood. Getting it wrong scales every error by `sqrt(2)`.

Returns a NamedTuple:

| field | |
|---|---|
| `std`, `std_scaled` | the two error scales above |
| `covariance`, `correlation` | full matrices; fixed parameters are `NaN` |
| `hessian` | the finite-difference second-derivative matrix |
| `gradient` | central-difference gradient, a diagnostic |
| `stationarity` | `max abs(grad_i) * sigma_i` — see below |
| `chi2`, `ndata`, `dof`, `reduced_chi2` | |
| `fixed` | `Bool` vector of what was held |
| `value`, `names` | echoed back |

### `error_table(res; io, correlations, threshold)`

Prints the table shown above, plus any parameter pair whose correlation exceeds
`threshold` (default 0.7) in magnitude. Those are the pairs whose individual
error bars mean the least — the data constrains a combination of them, not each
one separately.

### `profile_interval(f, p, i; sigma, level, minimizer, lower, upper, fixed)`

Finds where the objective rises by `level` above its value at `p`, by
bracketing and bisection. This is the *definition* that the curvature method
approximates, so it is the check to run when you do not trust the quadratic
picture. `level = 1` is one sigma; `level = 3.84` is 95 % for one parameter.

Pass a `minimizer` to re-optimise the other parameters at each trial value:

```julia
using Optim
mini(g, q0, lo, hi) = Optim.minimum(optimize(g, lo, hi, q0, Fminbox(LBFGS()),
                                             Optim.Options(iterations = 500);
                                             autodiff = :finite))

lo, hi = profile_interval(chi2, best, 2; sigma = err.std[2], minimizer = mini,
                          lower = lb, upper = ub)
```

Without a `minimizer` the other parameters stay frozen, which gives the
*conditional* interval — always narrower than the truth, sometimes much
narrower. With one you get the *marginal* interval, which is what `fit_errors`
reports and what you should quote.

## Worked examples

### A local fit with Optim

```julia
using Optim, FitErrors

chi2(p) = sum(((gauss.(x, p[1], p[2], p[3]) .- y) ./ sigma) .^ 2)
lb, ub = [0.0, 10.0, 0.1], [10.0, 12.0, 5.0]

res  = optimize(chi2, lb, ub, [1.0, 11.0, 1.0], Fminbox(LBFGS()))
best = Optim.minimizer(res)

err = fit_errors(chi2, best; lower = lb, upper = ub, ndata = length(y))
error_table(err)
```

### A global optimiser (MPA / SOMPA)

Identical, because nothing about `fit_errors` depends on how `best` was found:

```julia
using MPAOP, FitErrors

chi2val, best, curve = SOMPA(fobj = chi2, lb = lb, ub = ub,
                             SearchAgents_no = 60, Max_iter = 2000)

err = fit_errors(chi2, best; lower = lb, upper = ub, ndata = length(y))
```

Check `err.stationarity` here. A global optimiser stops when its population
converges, not when the gradient vanishes, so it can land slightly off the true
minimum. If `stationarity` is not small, polish with a few steps of a local
optimiser before taking errors.

### A point off a multi-objective front (MOMPA)

A Pareto point minimises *some* weighted combination of the objectives, not
necessarily the one you would write down. Build the scalar objective that the
point actually optimises, and check that it is stationary there:

```julia
front = MOMPA(fobj = p -> chi2_vector(p), lb = lb, ub = ub)
best  = front.positions[i, :]              # the compromise you chose

scalar(p) = sum(w .* chi2_vector(p))       # the weights that point represents
err = fit_errors(scalar, best; lower = lb, upper = ub, ndata = n)
```

If `stationarity` comes out large, the point is not a minimum of `scalar` and
the errors are meaningless — the quadratic expansion assumed a vanishing
gradient. Either find the weights it does minimise, or polish locally first.

### Asymmetric errors

```julia
err = fit_errors(chi2, best; ndata = n)
lo, hi = profile_interval(chi2, best, 2; sigma = err.std[2], minimizer = mini)
println("p2 = $(best[2]) +$(hi - best[2]) -$(best[2] - lo)")
```

## Diagnostics

**`stationarity`** is the drop in the objective from moving one sigma downhill:
`max abs(grad_i) * sigma_i`. The whole method assumes the gradient vanishes, so
this must be small compared with 1. A local optimiser gives ~1e-7 or less. If it
is large, `error_table` warns, and the number is not to be trusted.

**`NaN` on a parameter** means it sat on a bound. Its chi-squared is one-sided
there, so no Gaussian error exists. Widen the bound and refit — the `NaN` is
telling you the bound is doing the constraining, not the data.

**A warning and all-`NaN` errors** means the Hessian was not positive definite,
so this is not a minimum, or a finite-difference step left the valid region.
Check `stationarity`, check for active bounds, and inspect `res.hessian`.

**Strong correlations** near ±1 mean the data constrains a combination of
parameters rather than each individually. The individual error bars are then
large and misleading; quote the covariance, or reparameterise.

## When the method does not apply

1. **Multiple minima.** The curvature describes one basin. If your global
   optimiser finds different basins from different seeds, that spread is a
   separate and usually larger uncertainty. Report both.
2. **Strongly non-quadratic valleys.** Use `profile_interval`.
3. **Parameters at bounds.** See above.
4. **Correlated or underestimated data errors.** Nothing here can detect that;
   `std_scaled` is the crude patch.

For a fully model-independent answer, bootstrap (resample residuals, refit,
take the spread) or MCMC (samples the posterior, gives asymmetric intervals
honestly). Both cost many full fits. Curvature is the cheap first answer and
`profile_interval` is how you check whether it was good enough.

## Why `C = 2H⁻¹`

Near a minimum the objective has no linear term:

    chi2(p) ≈ chi2(p̂) + ½ Δpᵀ H Δp

For Gaussian errors the likelihood is `L ∝ exp(-chi2/2)`, so

    L ∝ exp(-¼ Δpᵀ H Δp)

Matching against a multivariate Gaussian `L ∝ exp(-½ Δpᵀ C⁻¹ Δp)` gives
`C⁻¹ = ½H`, hence `C = 2H⁻¹`. For a negative log likelihood the factor is 1
instead. Expanding `H` for a least-squares problem gives `H ≈ 2 JᵀWJ`, so
`C = (JᵀWJ)⁻¹` — the textbook linear-least-squares covariance. The finite
differences here keep the second-derivative term that Gauss–Newton drops.

Equivalently, and more memorably: one sigma is where chi-squared rises by 1
while the other parameters re-minimise.

## Numerical notes

Second derivatives by finite differences are delicate: roundoff scales as
`eps*|chi2|/h^2` while truncation scales as `h^2`, so the optimal step is
around `eps^(1/4)` — far larger than for a gradient, and too small a step is
worse than too large.

`fit_errors` therefore differences twice. The first pass uses a relative step to
get a rough sigma; the second re-differences with steps of one sigma, where the
objective changes by about 1 and the difference is well conditioned even when
chi-squared itself is ~1e4. This matters for parameters of very different
magnitude — differencing a parameter of order 1e-7 with a relative step of 1e-3
otherwise compares chi-squared values agreeing to 12 digits.

Cost is about `2k^2` objective evaluations per pass for `k` free parameters.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite checks the method against answers known independently: an analytic
quadratic with a built-in correlation, the textbook `(AᵀWA)⁻¹` of a weighted
linear fit, the Monte Carlo scatter of that fit over 4000 noise realisations
(the check that an error bar means what it should), the exact marginal/
conditional ratio `1/sqrt(1-rho^2)` for a bivariate Gaussian, the chi-squared
versus log-likelihood factor, bound and fixed-parameter handling, the
diagnostics, and asymmetric profile intervals.

## License

MIT — see [LICENSE](LICENSE).
