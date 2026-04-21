using LinearAlgebra, Random

olddir = pwd()
cd(@__DIR__)
include("pls_functions_refactored.jl")
cd(olddir)

# Build a 20x4 X matrix of sinusoids (each column is a different frequency)
n = 20
t = range(0, 2π, length=n)
X = hcat(2 * sin.(t), cos.(t), sin.(2t), 0.5 * cos.(2t))  # 20x4

# True mixing matrix P: 4x2
P = [1.0  0.5;
     0.3  1.0;
     0.8  0.2;
     0.1  0.9]

# True Y = X * P, so Y is 20x2
Y_true = X * P

# Fit PLS model
pls = make_pls_regressor(X, Y_true, 4; standardize=false, already_centered=false)

# Predict
Y_pred = predict(pls, X)

# Evaluate
residuals = Y_true .- Y_pred
rel_err = norm(residuals) / norm(Y_true)

println("=== PLS Refactored Test ===")
println("X size: ", size(X))
println("Y_true size: ", size(Y_true))
println("Y_pred size: ", size(Y_pred))
println("Relative reconstruction error: ", round(rel_err; sigdigits=4))
println("Max absolute residual: ", round(maximum(abs, residuals); sigdigits=4))

if rel_err < 1e-6
    println("PASS: PLS learned the linear relationship to near machine precision.")
else
    println("FAIL: Residuals are larger than expected.")
end

# Also check the coefficient matrix directly
coeffs = make_matrix_to_multiply_by_X_to_get_Y(pls)
println("\nRecovered coefficient matrix (should be close to P):")
println("P true:\n", P)
println("Coeffs recovered:\n", round.(coeffs; sigdigits=4))
println("Coeff error: ", round(norm(coeffs .- P) / norm(P); sigdigits=4))

# ── Noisy test ──────────────────────────────────────────────────────────────
println("\n=== PLS Refactored Test (with noise) ===")

noise_level = 0.1   # tune this: std of additive Gaussian noise on Y
rng = MersenneTwister(42)

Y_noisy = Y_true .+ noise_level .* randn(rng, size(Y_true))

pls_noisy = make_pls_regressor(X, Y_noisy, 4; standardize=false, already_centered=false)
Y_pred_noisy = predict(pls_noisy, X)

signal_power   = norm(Y_true)
noise_power    = norm(Y_noisy .- Y_true)
residuals_noisy = Y_true .- Y_pred_noisy
rel_err_noisy  = norm(residuals_noisy) / signal_power

println("Noise level (σ): ", noise_level)
println("SNR (signal/noise norms): ", round(signal_power / noise_power; sigdigits=4))
println("Relative prediction error vs Y_true: ", round(rel_err_noisy; sigdigits=4))
println("Max absolute residual vs Y_true: ", round(maximum(abs, residuals_noisy); sigdigits=4))
