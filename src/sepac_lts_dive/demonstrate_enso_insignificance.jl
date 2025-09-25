"""
Perform gridded decomposition analysis for different stratocumulus regions.
This script generalizes the decomposition analysis from gridded_effects.jl to work with multiple regions.
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")

using Statistics, DataFrames, Dictionaries, JLD2, CSV, Dates

date_range = (Date(2000,3), Date(2024, 3, 31))

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/enso_insignificance"
mkpath(visdir)

# Define the lags of interest
lags = -24:24

#Load in the LTS, Theta_1000, and Theta_700 data

# Load ENSO data for all lags
enso_data, enso_coords = load_enso_data(date_range; lags=collect(lags))

# Load theta/LTS data from local SEPac region time series (similar to gridded_effects.jl)
region = "SEPac_feedback_definition"
local_ts_dir = "../../data/sepac_lts_data/local_region_time_series"
era5_local_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_lagged_$(region).csv"), DataFrame)

# Filter to date range
era5_local_df[!, :date] = Date.(era5_local_df[!, :date])
era5_local_df = filter(row -> date_range[1] <= row.date <= date_range[2], era5_local_df)

# Extract the variables of interest at lag 0
lts_data = era5_local_df[!, "LTS_1000_lag_0"]
theta_1000_data = era5_local_df[!, "θ_1000_lag_0"] 
theta_700_data = era5_local_df[!, "θ_700_lag_0"]

# Create dictionary of lagged ENSO data for calculate_lag_correlations
enso_lagged_dict = Dictionary()
for lag in lags
    lag_col_name = "oni_lag_$lag"
    if haskey(enso_data, lag_col_name)
        set!(enso_lagged_dict, lag, enso_data[lag_col_name])
    end
end

#Plot the correlation of ENSO with each of the three variables as a function of lag

# Calculate lagged correlations for each variable
println("Calculating correlations...")
println("ENSO data available for lags: ", sort(collect(keys(enso_lagged_dict))))
println("Length of LTS data: ", length(lts_data))

lts_correlations = calculate_lag_correlations(lts_data, enso_lagged_dict; lags=lags)
theta_1000_correlations = calculate_lag_correlations(theta_1000_data, enso_lagged_dict; lags=lags)
theta_700_correlations = calculate_lag_correlations(theta_700_data, enso_lagged_dict; lags=lags)

println("Sample correlations calculated:")
println("LTS correlations for first few lags: ", collect(lts_correlations)[1:5])
println("Theta 1000 correlations for first few lags: ", collect(theta_1000_correlations)[1:5])

# Find maximum correlations and their lags for each variable
function find_max_correlation(corr_dict)
    max_corr = 0.0
    max_lag = 0
    for (lag, corr) in pairs(corr_dict)
        if !isnan(corr) && !isinf(corr) && abs(corr) > abs(max_corr)
            max_corr = corr
            max_lag = lag
        end
    end
    return max_corr, max_lag
end

lts_max_corr, lts_max_lag = find_max_correlation(lts_correlations)
theta_1000_max_corr, theta_1000_max_lag = find_max_correlation(theta_1000_correlations)
theta_700_max_corr, theta_700_max_lag = find_max_correlation(theta_700_correlations)

# Create plots
using PythonCall
@py import matplotlib.pyplot as plt

# Calculate standard deviations for scaling
lts_std = std(lts_data)
theta_1000_std = std(theta_1000_data)
theta_700_std = std(theta_700_data)

# Find overall maximum correlation magnitude for consistent y-axis scaling
all_corr_values = vcat(collect(values(lts_correlations)), 
                      collect(values(theta_1000_correlations)), 
                      collect(values(theta_700_correlations)))
max_abs_corr = maximum(abs.(filter(!isnan, all_corr_values)))
ylim_range = [-max_abs_corr * 1.1, max_abs_corr * 1.1]

# First plot: Individual correlations with same y-axis scale
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot LTS correlations
lag_values = collect(keys(lts_correlations))
lts_corr_values = collect(values(lts_correlations))
axs[0].plot(lag_values, lts_corr_values, "o-", linewidth=2, markersize=4)
axs[0].set_title("ENSO vs LTS₁₀₀₀ (Max: $(round(lts_max_corr, digits=3)) at lag $(lts_max_lag))")
axs[0].set_ylabel("Correlation")
axs[0].set_ylim(ylim_range)
axs[0].grid(true, alpha=0.3)
axs[0].axhline(y=0, color="k", linestyle="--", alpha=0.5)

# Plot Theta 1000 correlations  
theta_1000_corr_values = collect(values(theta_1000_correlations))
axs[1].plot(lag_values, theta_1000_corr_values, "o-", linewidth=2, markersize=4, color="orange")
axs[1].set_title("ENSO vs θ₁₀₀₀ (Max: $(round(theta_1000_max_corr, digits=3)) at lag $(theta_1000_max_lag))")
axs[1].set_ylabel("Correlation")
axs[1].set_ylim(ylim_range)
axs[1].grid(true, alpha=0.3)
axs[1].axhline(y=0, color="k", linestyle="--", alpha=0.5)

# Plot Theta 700 correlations
theta_700_corr_values = collect(values(theta_700_correlations))
axs[2].plot(lag_values, theta_700_corr_values, "o-", linewidth=2, markersize=4, color="green")
axs[2].set_title("ENSO vs θ₇₀₀ (Max: $(round(theta_700_max_corr, digits=3)) at lag $(theta_700_max_lag))")
axs[2].set_xlabel("Lag (months)")
axs[2].set_ylabel("Correlation")
axs[2].set_ylim(ylim_range)
axs[2].grid(true, alpha=0.3)
axs[2].axhline(y=0, color="k", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(joinpath(visdir, "enso_theta_lts_lag_correlations.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Second plot: All variables on one pane with scaling factors
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Calculate scaling factors
theta_1000_scale = -(theta_1000_std / lts_std)
theta_700_scale = theta_700_std / lts_std

# Apply scaling and plot
scaled_theta_1000_corrs = theta_1000_corr_values .* theta_1000_scale
scaled_theta_700_corrs = theta_700_corr_values .* theta_700_scale

ax.plot(lag_values, lts_corr_values, "o-", linewidth=2, markersize=4, 
        label="LTS₁₀₀₀", color="blue")
ax.plot(lag_values, scaled_theta_1000_corrs, "o-", linewidth=2, markersize=4, 
        label="θ₁₀₀₀ × (-σ_θ₁₀₀₀/σ_LTS)", color="orange")
ax.plot(lag_values, scaled_theta_700_corrs, "o-", linewidth=2, markersize=4, 
        label="θ₇₀₀ × (σ_θ₇₀₀/σ_LTS)", color="green")

ax.set_title("ENSO Correlations with SEPac Variables (Scaled)")
ax.set_xlabel("Lag (months)")
ax.set_ylabel("Scaled Correlation")
ax.grid(true, alpha=0.3)
ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(joinpath(visdir, "enso_theta_lts_combined_scaled.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Print summary statistics
println("\nSummary of maximum correlations:")
println("LTS₁₀₀₀: $(round(lts_max_corr, digits=3)) at lag $(lts_max_lag)")
println("θ₁₀₀₀: $(round(theta_1000_max_corr, digits=3)) at lag $(theta_1000_max_lag)")  
println("θ₇₀₀: $(round(theta_700_max_corr, digits=3)) at lag $(theta_700_max_lag)")

println("\nStandard deviations:")
println("LTS₁₀₀₀ std: $(round(lts_std, digits=3))")
println("θ₁₀₀₀ std: $(round(theta_1000_std, digits=3))")
println("θ₇₀₀ std: $(round(theta_700_std, digits=3))")

println("\nScaling factors for combined plot:")
println("θ₁₀₀₀ scaling: $(round(theta_1000_scale, digits=3))")
println("θ₇₀₀ scaling: $(round(theta_700_scale, digits=3))")

println("\nPlots saved to: $visdir")

# PLS Analysis: Use all ENSO lags to predict LTS
println("\n" * "="^60)
println("PLS REGRESSION ANALYSIS")
println("="^60)

# Include PLS functions
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")

# Create a DataFrame with all ENSO lags and LTS data
println("Creating joint ENSO-LTS dataframe...")

# Create a complete dataframe with all ENSO lags and LTS
joint_df = DataFrame()
joint_df.time = enso_coords["time"]

# Add all ENSO lags
for lag in lags
    lag_col_name = "oni_lag_$lag"
    if haskey(enso_data, lag_col_name)
        joint_df[!, Symbol(lag_col_name)] = enso_data[lag_col_name]
    end
end

# Add LTS data (need to align with ENSO time)
# First convert ENSO times to same format as ERA5 times
enso_times_dates = Date.(round.(enso_coords["time"], Month(1), RoundDown))
era5_times_dates = era5_local_df.date

# Find matching times
time_matches = []
lts_matched = []
for (i, enso_time) in enumerate(enso_times_dates)
    era5_idx = findfirst(==(enso_time), era5_times_dates)
    if era5_idx !== nothing
        push!(time_matches, i)
        push!(lts_matched, lts_data[era5_idx])
    end
end

# Filter joint_df to matching times and add LTS
joint_df = joint_df[time_matches, :]
joint_df.LTS_1000 = lts_matched

# Remove rows with any missing values
println("Original joint dataframe size: $(nrow(joint_df))")
complete_cases_mask = [!any(ismissing, row) for row in eachrow(joint_df)]
joint_df_complete = joint_df[complete_cases_mask, :]
println("After removing missing values: $(nrow(joint_df_complete))")

# Prepare data for PLS
enso_cols = [Symbol("oni_lag_$lag") for lag in lags if haskey(enso_data, "oni_lag_$lag")]
X_enso = Matrix{Float64}(select(joint_df_complete, enso_cols))
y_lts = Vector{Float64}(joint_df_complete.LTS_1000)

println("ENSO predictor matrix size: $(size(X_enso))")
println("LTS target vector length: $(length(y_lts))")

# Perform 1-component PLS regression
println("Performing 1-component PLS regression...")
pls_model = make_pls_regressor(X_enso, reshape(y_lts, :, 1), 1; print_updates=false)

# Extract PLS scores (predictor time series)
enso_pls_predictor = pls_model.X_scores[:, 1]

# Perform linear least squares regression: LTS ~ PLS_predictor
println("Performing least squares regression...")
A = hcat(Float64.(enso_pls_predictor), ones(Float64, length(enso_pls_predictor)))
lsq_coeffs = A \ Float64.(y_lts)
pls_slope = lsq_coeffs[1]
pls_intercept = lsq_coeffs[2]

# Calculate predicted LTS values
lts_predicted = pls_slope .* enso_pls_predictor .+ pls_intercept

# Calculate residuals
lts_residuals = y_lts .- lts_predicted

# Calculate R² and correlation
r_squared = 1 - sum(lts_residuals.^2) / sum((y_lts .- mean(y_lts)).^2)
correlation = cor(y_lts, lts_predicted)

println("PLS Regression Results:")
println("  Correlation (observed vs predicted): $(round(correlation, digits=4))")
println("  R²: $(round(r_squared, digits=4))")
println("  Slope: $(round(pls_slope, digits=4))")
println("  Intercept: $(round(pls_intercept, digits=4))")

# Create time array for plotting
plot_times = joint_df_complete.time

# Find common y-axis limits for all three panels
all_values = vcat(y_lts, lts_predicted, lts_residuals)
y_min, y_max = extrema(all_values)
y_padding = (y_max - y_min) * 0.05
ylim_common = [y_min - y_padding, y_max + y_padding]

# Create 3-panel plot
fig, axs = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: Original LTS
axs[0].plot(plot_times, y_lts, "b-", linewidth=2, label="Observed LTS")
axs[0].set_title("Original LTS Index")
axs[0].set_ylabel("LTS (K)")
axs[0].set_ylim(ylim_common)
axs[0].grid(true, alpha=0.3)
axs[0].legend()

# Panel 2: Predicted LTS
axs[1].plot(plot_times, lts_predicted, "r-", linewidth=2, label="ENSO PLS Predicted LTS")
axs[1].plot(plot_times, y_lts, "b-", alpha=0.5, linewidth=1, label="Observed LTS")
axs[1].set_title("ENSO PLS Predicted LTS (r=$(round(correlation, digits=3)), R²=$(round(r_squared, digits=3)))")
axs[1].set_ylabel("LTS (K)")
axs[1].set_ylim(ylim_common)
axs[1].grid(true, alpha=0.3)
axs[1].legend()

# Panel 3: Residuals
axs[2].plot(plot_times, lts_residuals, "g-", linewidth=2, label="Residuals (Observed - Predicted)")
axs[2].axhline(y=0, color="k", linestyle="--", alpha=0.7)
axs[2].set_title("Residuals (Observed - Predicted LTS)")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Residual LTS (K)")
axs[2].set_ylim(ylim_common)
axs[2].grid(true, alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.savefig(joinpath(visdir, "enso_pls_lts_prediction.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

println("\nPLS analysis complete. Plot saved to: $(joinpath(visdir, "enso_pls_lts_prediction.png"))")

