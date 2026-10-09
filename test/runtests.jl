using FitErrors
using Test
using LinearAlgebra
using Random

@testset "FitErrors.jl" begin

    @testset "analytic quadratic" begin
        # chi2 built from a known covariance: the method must give it back.
        a = [1.0, -3.0e-4]
        s = [0.2, 2.0e-5]
        rho = 0.6
        C = [s[1]^2  rho*s[1]*s[2]; rho*s[1]*s[2]  s[2]^2]
        chi2 = p -> dot(p .- a, inv(C) * (p .- a))

        r = fit_errors(chi2, a; ndata = 100, names = [:u, :v])
        @test r.std[1] ≈ s[1] rtol = 1e-4
        @test r.std[2] ≈ s[2] rtol = 1e-4
        @test r.correlation[1, 2] ≈ rho rtol = 1e-4
        @test r.covariance[1, 2] ≈ C[1, 2] rtol = 1e-3
        @test r.dof == 98
        # at the minimum chi2 is 0, so the scaled errors collapse
        @test r.std_scaled[1] ≈ 0.0 atol = 1e-12
        @test r.stationarity < 1e-6
        # the Hessian of a chi2 with covariance C is 2*inv(C)
        @test r.hessian ≈ 2 .* inv(C) rtol = 1e-3
    end

    @testset "weighted linear fit matches (A'WA)^-1" begin
        # A linear model has a closed-form solution and covariance, so this
        # compares against the textbook answer with no optimiser involved.
        x = collect(0.0:0.25:10.0)
        sig = 0.05 .+ 0.02 .* x
        A = hcat(ones(length(x)), x)
        W = Diagonal(1 ./ sig .^ 2)
        C = inv(A' * W * A)

        rng = MersenneTwister(20260930)
        y = A * [2.0, -0.35] .+ sig .* randn(rng, length(x))
        phat = C * (A' * W * y)
        chi2 = p -> sum(((A * p .- y) ./ sig) .^ 2)

        r = fit_errors(chi2, phat; ndata = length(y), names = [:intercept, :slope])
        @test r.std[1] ≈ sqrt(C[1, 1]) rtol = 1e-5
        @test r.std[2] ≈ sqrt(C[2, 2]) rtol = 1e-5
        @test r.correlation[1, 2] ≈ C[1, 2] / sqrt(C[1, 1] * C[2, 2]) rtol = 1e-4
        @test r.reduced_chi2 ≈ r.chi2 / (length(y) - 2)
    end

    @testset "sigma predicts the Monte Carlo scatter" begin
        # The point of an error bar: refit many noise realisations and the
        # spread of the estimates must match the single-fit prediction.
        x = collect(0.0:0.25:10.0)
        sig = 0.05 .+ 0.02 .* x
        A = hcat(ones(length(x)), x)
        W = Diagonal(1 ./ sig .^ 2)
        C = inv(A' * W * A)
        truth = [2.0, -0.35]

        rng = MersenneTwister(11)
        n = 4000
        est = zeros(n, 2)
        for t in 1:n
            yt = A * truth .+ sig .* randn(rng, length(x))
            est[t, :] = C * (A' * W * yt)
        end
        scatter(v) = (m = sum(v) / length(v); sqrt(sum((v .- m) .^ 2) / (length(v) - 1)))

        y = A * truth .+ sig .* randn(rng, length(x))
        phat = C * (A' * W * y)
        chi2 = p -> sum(((A * p .- y) ./ sig) .^ 2)
        r = fit_errors(chi2, phat; ndata = length(y))

        @test scatter(est[:, 1]) ≈ r.std[1] rtol = 0.06
        @test scatter(est[:, 2]) ≈ r.std[2] rtol = 0.06
    end

    @testset "marginal vs conditional" begin
        # With correlated parameters the marginal error (full inverse) must
        # exceed the conditional one (curvature along the axis alone).
        a = [0.0, 0.0]
        s = [1.0, 1.0]
        rho = 0.8
        C = [s[1]^2  rho*s[1]*s[2]; rho*s[1]*s[2]  s[2]^2]
        chi2 = p -> dot(p .- a, inv(C) * (p .- a))
        r = fit_errors(chi2, a)

        conditional = sqrt(2 / r.hessian[1, 1])
        @test r.std[1] > conditional
        # for a bivariate Gaussian the ratio is exactly 1/sqrt(1-rho^2)
        @test r.std[1] / conditional ≈ 1 / sqrt(1 - rho^2) rtol = 1e-4
    end

    @testset "chi2 versus negative log likelihood" begin
        f = p -> ((p[1] - 2.0) / 0.5)^2
        rc = fit_errors(f, [2.0]; kind = :chi2)
        rl = fit_errors(f, [2.0]; kind = :neglogl)
        @test rc.std[1] ≈ 0.5 rtol = 1e-5
        @test rl.std[1] ≈ 0.5 / sqrt(2) rtol = 1e-5
        @test_throws ErrorException fit_errors(f, [2.0]; kind = :nonsense)
    end

    @testset "bounds and fixed parameters" begin
        C = Diagonal([0.25, 4.0])
        chi2 = p -> dot(p, inv(C) * p)

        # second parameter pinned on its upper bound
        r = fit_errors(chi2, [0.0, 1.0]; lower = [-5.0, -5.0], upper = [5.0, 1.0])
        @test r.fixed == [false, true]
        @test isnan(r.std[2])
        @test !isnan(r.std[1])
        @test r.std[1] ≈ 0.5 rtol = 1e-4

        # holding one by hand gives the conditional error of the other
        rf = fit_errors(chi2, [0.0, 0.0]; fixed = [2])
        @test rf.fixed == [false, true]
        @test rf.std[1] ≈ 0.5 rtol = 1e-4

        @test_throws ErrorException fit_errors(chi2, [0.0, 0.0]; fixed = [1, 2])
    end

    @testset "diagnostics" begin
        C = Diagonal([1.0, 1.0])
        chi2 = p -> dot(p, inv(C) * p)

        # away from the minimum the gradient does not vanish and it says so
        off = fit_errors(chi2, [3.0, 0.0])
        @test off.stationarity > 1.0

        # a maximum is not a minimum: no errors, and a warning
        peak = p -> -dot(p, p)
        r = @test_logs (:warn,) match_mode = :any fit_errors(peak, [0.0, 0.0])
        @test all(isnan, r.std)

        @test_throws ErrorException fit_errors(p -> NaN, [0.0])
    end

    @testset "profile interval" begin
        # one parameter, so there is nothing to re-minimise and the profile
        # interval is exactly the quadratic one
        f = p -> ((p[1] - 3.0) / 0.25)^2
        r = fit_errors(f, [3.0])
        lo, hi = profile_interval(f, [3.0], 1; sigma = r.std[1])
        @test hi - 3.0 ≈ 0.25 rtol = 1e-3
        @test 3.0 - lo ≈ 0.25 rtol = 1e-3

        # 95 per cent for one parameter is Delta chi2 = 3.84
        lo95, hi95 = profile_interval(f, [3.0], 1; sigma = r.std[1], level = 3.84)
        @test hi95 - 3.0 ≈ 0.25 * sqrt(3.84) rtol = 1e-3

        # an asymmetric objective gives an asymmetric interval, which is the
        # whole reason to profile rather than trust the curvature
        g = p -> (exp(p[1]) - 1)^2
        rg = fit_errors(g, [0.0])
        alo, ahi = profile_interval(g, [0.0], 1; sigma = rg.std[1])
        @test ahi - 0.0 < 0.0 - alo
    end

    @testset "error_table prints" begin
        C = Diagonal([0.25, 1.0])
        chi2 = p -> dot(p, inv(C) * p)
        r = fit_errors(chi2, [0.0, 0.0]; ndata = 20, names = [:a, :b])
        error_table(r; io = devnull)
        error_table(r; io = devnull, correlations = false)
        # unnamed parameters fall back to p1, p2, ...
        error_table(fit_errors(chi2, [0.0, 0.0]); io = devnull)
    end
end
