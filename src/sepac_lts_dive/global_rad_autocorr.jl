"""
This script loads the lagged global radiation data and creates autocorrelation plots
to examine the persistence and temporal structure of global radiation variables.
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")

using CSV, DataFrames, Dates
using PythonCall
@py import matplotlib.pyplot as plt
@py import numpy as np

# Define analysis bounds
analysis_bounds = (Date(2000, 3), Date(2023, 2, 28))

# Load in the lagged global radiation time series from CERES
ceres_global_df = CSV.read("../../data/CERES/lagged/global_ceres_lagged_detrended_deseasonalized.csv", DataFrame)

# Convert date column and filter to analysis bounds
ceres_global_df[!, :date] = Date.(ceres_global_df[!, :date])
ceres_global_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], ceres_global_df)

# Create output directory
output_dir = "../../vis/lts_global_rad/global_rad_autocorr"
mkpath(output_dir)

# Extract the base variable names (without lag suffix)
all_cols = names(ceres_global_df)
base_vars = unique([split(col, "_lag_")[1] for col in all_cols if occursin("_lag_", col)])

println("Found $(length(base_vars)) base variables: ", base_vars)

# Extract all lag values from the column names
all_lags = Int[]
for col in all_cols
    if occursin("_lag_", col)
        lag_str = split(col, "_lag_")[2]
        try
            lag_val = parse(Int, lag_str)
            push!(all_lags, lag_val)
        catch
            # Skip if not a valid integer
        end
    end
end
all_lags = sort(unique(all_lags))

println("Found lags: ", all_lags)

# Function to calculate autocorrelation for a given variable at different lags
function calculate_autocorrelations(df, base_var, lags)
    autocorrs = Float64[]
    valid_lags = Int[]
    
    # The lag_0 column is the reference time series
    ref_col = "$(base_var)_lag_0"
    if !(ref_col in names(df))
        println("Warning: Reference column $ref_col not found")
        return valid_lags, autocorrs
    end
    
    ref_series = df[!, ref_col]
    
    for lag in lags
        lag_col = "$(base_var)_lag_$(lag)"
        if lag_col in names(df)
            lagged_series = df[!, lag_col]
            
            # Calculate correlation, skipping missing values
            valid_idx = .!ismissing.(ref_series) .& .!ismissing.(lagged_series)
            if sum(valid_idx) > 0
                corr_val = cor(ref_series[valid_idx], lagged_series[valid_idx])
                push!(autocorrs, corr_val)
                push!(valid_lags, lag)
            end
        end
    end
    
    return valid_lags, autocorrs
end

# Create autocorrelation plots for each variable
for base_var in base_vars
    println("Processing autocorrelation for $base_var...")
    
    lags, autocorrs = calculate_autocorrelations(ceres_global_df, base_var, all_lags)
    
    if isempty(autocorrs)
        println("No valid autocorrelations found for $base_var, skipping...")
        continue
    end
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot autocorrelation
    ax.plot(lags, autocorrs, marker="o", linewidth=2, markersize=8, color="steelblue")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=1)
    ax.grid(true, alpha=0.3)
    
    # Add confidence interval lines (approximate 95% CI for white noise)
    n = nrow(ceres_global_df)
    ci = 1.96 / sqrt(n)
    ax.axhline(y=ci, color="red", linestyle="--", alpha=0.5, linewidth=1, label="95% CI")
    ax.axhline(y=-ci, color="red", linestyle="--", alpha=0.5, linewidth=1)
    
    ax.set_xlabel("Lag (months)", fontsize=12)
    ax.set_ylabel("Autocorrelation", fontsize=12)
    ax.set_title("Autocorrelation Function: $base_var", fontsize=14, fontweight="bold")
    ax.legend()
    
    # Save the figure
    output_path = joinpath(output_dir, "$(base_var)_autocorr.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    println("Saved: $output_path")
    plt.close(fig)
end

# Create a combined plot showing all variables together
println("Creating combined autocorrelation plot...")

fig, ax = plt.subplots(figsize=(12, 8))

colors = ["steelblue", "coral", "seagreen", "purple", "orange", "brown", "pink", "gray"]
markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

for (idx, base_var) in enumerate(base_vars)
    lags, autocorrs = calculate_autocorrelations(ceres_global_df, base_var, all_lags)
    
    if !isempty(autocorrs)
        color = colors[mod1(idx, length(colors))]
        marker = markers[mod1(idx, length(markers))]
        ax.plot(lags, autocorrs, marker=marker, linewidth=2, markersize=6, 
                label=base_var, color=color, alpha=0.8)
    end
end

ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, linewidth=1)
ax.grid(true, alpha=0.3)

# Add confidence interval
n = nrow(ceres_global_df)
ci = 1.96 / sqrt(n)
ax.axhline(y=ci, color="red", linestyle="--", alpha=0.5, linewidth=1, label="95% CI")
ax.axhline(y=-ci, color="red", linestyle="--", alpha=0.5, linewidth=1)

ax.set_xlabel("Lag (months)", fontsize=12)
ax.set_ylabel("Autocorrelation", fontsize=12)
ax.set_title("Autocorrelation Functions: Global Radiation Variables", fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=9)

# Save combined figure
output_path = joinpath(output_dir, "all_variables_autocorr.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
println("Saved: $output_path")
plt.close(fig)

println("Autocorrelation analysis complete!")
