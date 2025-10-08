"""
Investigate the order dependency of detrending and deseasonalizing operations
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/why_ceres_why"

using Statistics, Plots

# Create synthetic data with both trend and seasonal cycle
t = 1:100
months = repeat(1:10, 10)
months = circshift(months, 5)
y = t/33 + sin.(2π .* months ./ 10)  # Linear trend + seasonal cycle
y .-= mean(y)

println("=== Analysis of Order-Dependent Detrending and Deseasonalizing ===\n")

# Store original data for comparison
y_original = copy(y)

println("Step 1: Plot original data")
p1 = plot(t, y_original, label="Original Data", lw=2, title="Original Data with Trend and Seasonal Cycle")
display(p1)

# ===== METHOD A: Remove seasonal cycle first, then trend =====
println("\n=== METHOD A: Remove Seasonal Cycle First, Then Trend ===")

# Step A1: Calculate and remove seasonal cycle from original data
y_A = copy(y_original)
seasonal_cycle_A = calc_seasonal_cycle(y_A, months)
seasonal_cycle_full_time_A = [seasonal_cycle_A[m] for m in months]

println("Step A1: Calculate original seasonal cycle")
p_A1 = plot(t, y_A, label="Original Data", lw=2)
plot!(t, seasonal_cycle_full_time_A, label="Seasonal Cycle", lw=2)
plot!(title="Method A1: Original Data + Seasonal Cycle")
display(p_A1)

# Remove seasonal cycle
deseasonalize!(y_A, months)

println("Step A2: After removing seasonal cycle, calculate trend")
trend_A = least_squares_fit(t, y_A)
println("Trend after deseasonalizing: slope=$(round(trend_A.slope, digits=6)), intercept=$(round(trend_A.intercept, digits=6))")

p_A2 = plot(t, y_A, label="Deseasonalized Data", lw=2)
plot!(t, trend_A.slope .* t .+ trend_A.intercept, label="Trend", lw=2)
plot!(title="Method A2: Deseasonalized Data + Fitted Trend")
display(p_A2)

# Step A3: Remove the trend
detrend!(y_A, t, trend_A.slope, trend_A.intercept)

println("Step A3: After removing trend, check remaining seasonal cycle")
remaining_seasonal_A = calc_seasonal_cycle(y_A, months)
remaining_seasonal_full_A = [remaining_seasonal_A[m] for m in months]

p_A3 = plot(t, y_A, label="Final Processed Data (A)", lw=2)
plot!(t, remaining_seasonal_full_A, label="Remaining Seasonal Cycle", lw=2)
plot!(title="Method A3: Final Result + Remaining Seasonality")
display(p_A3)

println("Remaining seasonal cycle amplitude (A): $(round(maximum(remaining_seasonal_A) - minimum(remaining_seasonal_A), digits=6))")
println("Final data std (A): $(round(std(y_A), digits=6))")

# ===== METHOD B: Remove trend first, then seasonal cycle =====
println("\n=== METHOD B: Remove Trend First, Then Seasonal Cycle ===")

# Step B1: Calculate and remove trend from original data
y_B = copy(y_original)
trend_B = least_squares_fit(t, y_B)
println("Original trend: slope=$(round(trend_B.slope, digits=6)), intercept=$(round(trend_B.intercept, digits=6))")

println("Step B1: Calculate original trend")
p_B1 = plot(t, y_B, label="Original Data", lw=2)
plot!(t, trend_B.slope .* t .+ trend_B.intercept, label="Trend", lw=2)
plot!(title="Method B1: Original Data + Trend")
display(p_B1)

# Remove trend
detrend!(y_B, t, trend_B.slope, trend_B.intercept)

println("Step B2: After removing trend, calculate seasonal cycle")
seasonal_cycle_B = calc_seasonal_cycle(y_B, months)
seasonal_cycle_full_time_B = [seasonal_cycle_B[m] for m in months]

p_B2 = plot(t, y_B, label="Detrended Data", lw=2)
plot!(t, seasonal_cycle_full_time_B, label="Seasonal Cycle", lw=2)
plot!(title="Method B2: Detrended Data + Seasonal Cycle")
display(p_B2)

# Check if seasonal cycle has a trend
seasonal_trend_B = least_squares_fit(t, seasonal_cycle_full_time_B)
println("Trend in seasonal cycle (B): slope=$(round(seasonal_trend_B.slope, digits=6)), intercept=$(round(seasonal_trend_B.intercept, digits=6))")

# Step B3: Remove seasonal cycle
deseasonalize!(y_B, months)

println("Step B3: After removing seasonal cycle, check remaining trend")
remaining_trend_B = least_squares_fit(t, y_B)
remaining_trend_full_B = remaining_trend_B.slope .* t .+ remaining_trend_B.intercept

p_B3 = plot(t, y_B, label="Final Processed Data (B)", lw=2)
plot!(t, remaining_trend_full_B, label="Remaining Trend", lw=2)
plot!(title="Method B3: Final Result + Remaining Trend")
display(p_B3)

println("Remaining trend (B): slope=$(round(remaining_trend_B.slope, digits=6)), intercept=$(round(remaining_trend_B.intercept, digits=6))")
println("Final data std (B): $(round(std(y_B), digits=6))")

# ===== COMPARISON =====
println("\n=== COMPARISON OF METHODS ===")

p_compare = plot(t, y_A, label="Method A (Deseason→Detrend)", lw=2)
plot!(t, y_B, label="Method B (Detrend→Deseason)", lw=2)
plot!(t, y_original, label="Original Data", lw=1, alpha=0.5)
plot!(title="Comparison of Final Results")
display(p_compare)
savefig(p_compare, joinpath(visdir, "invertibility_comparison.png"))

# Compare differences
diff_AB = y_A .- y_B
p_diff = plot(t, diff_AB, label="Method A - Method B", lw=2)
plot!(title="Difference Between Methods")
println("Max absolute difference: $(round(maximum(abs.(diff_AB)), digits=6))")
println("RMS difference: $(round(sqrt(mean(diff_AB.^2)), digits=6))")
display(p_diff)

# Summary statistics
println("\n=== SUMMARY ===")
println("Original data statistics:")
println("  - Trend slope: $(round(trend_B.slope, digits=6))")
println("  - Seasonal amplitude: $(round(maximum(seasonal_cycle_full_time_A) - minimum(seasonal_cycle_full_time_A), digits=6))")
println("  - Total std: $(round(std(y_original), digits=6))")
println()
println("Method A (Deseason→Detrend) results:")
println("  - Fitted trend slope: $(round(trend_A.slope, digits=6))")
println("  - Remaining seasonal amplitude: $(round(maximum(remaining_seasonal_A) - minimum(remaining_seasonal_A), digits=6))")
println("  - Final std: $(round(std(y_A), digits=6))")
println()
println("Method B (Detrend→Deseason) results:")
println("  - Seasonal cycle trend slope: $(round(seasonal_trend_B.slope, digits=6))")
println("  - Remaining trend slope: $(round(remaining_trend_B.slope, digits=6))")
println("  - Final std: $(round(std(y_B), digits=6))")
println()
println("Difference between methods:")
println("  - Max |difference|: $(round(maximum(abs.(diff_AB)), digits=6))")
println("  - RMS difference: $(round(sqrt(mean(diff_AB.^2)), digits=6))")
println("  - Are methods equivalent? $(maximum(abs.(diff_AB)) < 1e-10 ? "Yes" : "No")")

# ===== NEW METHOD B2: Apply Method A sequence twice =====
println("\n=== NEW METHOD B2: Apply Deseason→Detrend Twice ===")

y_B2 = copy(y_original)

# First iteration of deseason→detrend
println("First iteration:")
deseasonalize!(y_B2, months)
trend_B2_1 = least_squares_fit(t, y_B2)
detrend!(y_B2, t, trend_B2_1.slope, trend_B2_1.intercept)
println("  - Trend slope 1: $(round(trend_B2_1.slope, digits=6))")

# Second iteration of deseason→detrend
println("Second iteration:")
seasonal_cycle_B2_2 = calc_seasonal_cycle(y_B2, months)
deseasonalize!(y_B2, months)
trend_B2_2 = least_squares_fit(t, y_B2)
detrend!(y_B2, t, trend_B2_2.slope, trend_B2_2.intercept)
println("  - Seasonal amplitude 2: $(round(maximum(seasonal_cycle_B2_2) - minimum(seasonal_cycle_B2_2), digits=6))")
println("  - Trend slope 2: $(round(trend_B2_2.slope, digits=6))")

# ===== NEW METHOD C: Apply Method B sequence twice =====
println("\n=== NEW METHOD C: Apply Detrend→Deseason Twice ===")

y_C = copy(y_original)

# First iteration of detrend→deseason
println("First iteration:")
trend_C_1 = least_squares_fit(t, y_C)
detrend!(y_C, t, trend_C_1.slope, trend_C_1.intercept)
seasonal_cycle_C_1 = calc_seasonal_cycle(y_C, months)
deseasonalize!(y_C, months)
println("  - Trend slope 1: $(round(trend_C_1.slope, digits=6))")
println("  - Seasonal amplitude 1: $(round(maximum(seasonal_cycle_C_1) - minimum(seasonal_cycle_C_1), digits=6))")

# Second iteration of detrend→deseason
println("Second iteration:")
trend_C_2 = least_squares_fit(t, y_C)
detrend!(y_C, t, trend_C_2.slope, trend_C_2.intercept)
seasonal_cycle_C_2 = calc_seasonal_cycle(y_C, months)
deseasonalize!(y_C, months)
println("  - Trend slope 2: $(round(trend_C_2.slope, digits=6))")
println("  - Seasonal amplitude 2: $(round(maximum(seasonal_cycle_C_2) - minimum(seasonal_cycle_C_2), digits=6))")

# ===== COMPREHENSIVE COMPARISON =====
println("\n=== COMPREHENSIVE COMPARISON ===")

p_all = plot(t, y_original, label="Original Data", lw=3, alpha=0.7, color=:black)
plot!(t, y_A, label="Method A (Deseason→Detrend)", lw=2, color=:red)
plot!(t, y_B, label="Method B (Detrend→Deseason)", lw=2, color=:blue)
plot!(t, y_B2, label="Method B2 (Deseason→Detrend × 2)", lw=2, color=:green)
plot!(t, y_C, label="Method C (Detrend→Deseason × 2)", lw=2, color=:orange)
plot!(title="Comprehensive Comparison of All Methods")
plot!(xlabel="Time", ylabel="Value")
plot!(legend=:topleft)
display(p_all)

# Calculate final statistics for all methods
println("\n=== FINAL STATISTICS ===")
println("Method A (single Deseason→Detrend):")
println("  - Final std: $(round(std(y_A), digits=6))")
println("  - Mean: $(round(mean(y_A), digits=6))")

println("Method B (single Detrend→Deseason):")
println("  - Final std: $(round(std(y_B), digits=6))")
println("  - Mean: $(round(mean(y_B), digits=6))")

println("Method B2 (double Deseason→Detrend):")
println("  - Final std: $(round(std(y_B2), digits=6))")
println("  - Mean: $(round(mean(y_B2), digits=6))")

println("Method C (double Detrend→Deseason):")
println("  - Final std: $(round(std(y_C), digits=6))")
println("  - Mean: $(round(mean(y_C), digits=6))")

# Check convergence by comparing double methods
diff_B2_C = y_B2 .- y_C
println("\nConvergence analysis:")
println("  - Max |B2 - C|: $(round(maximum(abs.(diff_B2_C)), digits=6))")
println("  - RMS |B2 - C|: $(round(sqrt(mean(diff_B2_C.^2)), digits=6))")
println("  - Do double methods converge? $(maximum(abs.(diff_B2_C)) < 1e-6 ? "Yes" : "No")")

# Plot differences between methods
p_diffs = plot()
plot!(t, y_A .- y_B, label="A - B", lw=2)
plot!(t, y_A .- y_B2, label="A - B2", lw=2)
plot!(t, y_A .- y_C, label="A - C", lw=2)
plot!(t, y_B .- y_B2, label="B - B2", lw=2)
plot!(t, y_B .- y_C, label="B - C", lw=2)
plot!(t, y_B2 .- y_C, label="B2 - C", lw=2)
plot!(title="Pairwise Differences Between Methods")
plot!(xlabel="Time", ylabel="Difference")
plot!(legend=:topright)
display(p_diffs)