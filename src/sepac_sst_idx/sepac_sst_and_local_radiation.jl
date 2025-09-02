"""
This script plots the correlation between local SEPac net rad, sw rad, and lw rad, and the 3 time series 
(sepac_sst_index, oni_at_optimal_lag, sepac_sst_residual) from the ENSO-SEPac correlation 
results as a function of lags from -24 to 24 months.
Uses local SEPac radiation data instead of global mean CERES data.
"""

using Plots, Statistics, StatsBase, Dates, DataFrames, CSV
using LinearAlgebra

# Include necessary modules
olddir = pwd()
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
cd(olddir)

"""
    preprocess_data(data, time_points)

Detrend and deseasonalize time series data using existing utility functions.
"""
function preprocess_data(data, time_points)
    # Make a copy to avoid modifying the original
    processed_data = copy(data)
    
    # Convert time points to float times for detrending
    float_times = @. year(time_points) + (month(time_points) - 1) / 12
    months = month.(time_points)
    
    # Use the existing utility function to detrend and deseasonalize
    detrend_and_deseasonalize!(processed_data, float_times, months)
    
    return processed_data
end

"""
    load_saved_enso_sepac_data()

Load the saved ENSO-SEPac correlation results and filter for standard time period.
"""
function load_saved_enso_sepac_data()
    # Load the saved results
    csv_path = "../../data/v1draft_saved_data/enso_sepac_correlation_results.csv"
    df = CSV.read(csv_path, DataFrame)
    
    # Convert Date column to DateTime for consistency
    df.Date = DateTime.(df.Date)
    
    # Filter for standard time period
    standard_start = time_period[1]
    standard_end = time_period[2]
    
    # Convert to Date for comparison
    df_dates = Date.(year.(df.Date), month.(df.Date), 1)
    mask = (df_dates .>= standard_start) .& (df_dates .< standard_end)
    
    filtered_df = df[mask, :]
    
    println("Loaded $(nrow(df)) total points, filtered to $(nrow(filtered_df)) for standard period")
    println("Standard period: $(minimum(filtered_df.Date)) to $(maximum(filtered_df.Date))")
    
    return filtered_df
end

"""
    calculate_lagged_correlations(radiation_data, index_data, lag_range)

Calculate correlations between radiation data and index data for different lags.
Uses time_lag function from utils.
"""
function calculate_lagged_correlations(radiation_data, index_data, lag_range)
    correlations = Float64[]
    
    for lag in lag_range
        # Apply lag to the index data
        lagged_index = time_lag(index_data, lag)
        
        # Remove missing values for correlation calculation
        valid_mask = .!ismissing.(lagged_index) .& .!ismissing.(radiation_data)
        
        if sum(valid_mask) < 10  # Need at least 10 valid points
            push!(correlations, NaN)
            continue
        end
        
        # Calculate correlation
        rad_valid = radiation_data[valid_mask]
        idx_valid = collect(skipmissing(lagged_index[valid_mask]))
        
        if length(rad_valid) != length(idx_valid)
            push!(correlations, NaN)
            continue
        end
        
        corr_val = cor(rad_valid, idx_valid)
        push!(correlations, corr_val)
    end
    
    return correlations
end

"""
    plot_correlation_matrix()

Main function to create correlation plots.
"""
function plot_correlation_matrix()
    println("Starting correlation analysis of radiation vs ENSO-SEPac indices...")
    
    # Load the saved ENSO-SEPac data for standard period
    enso_sepac_df = load_saved_enso_sepac_data()
    
    # Extract the three time series
    sepac_sst = enso_sepac_df.sepac_sst_index
    oni_optimal = enso_sepac_df.oni_at_optimal_lag
    sepac_residual = enso_sepac_df.sepac_sst_residual
    time_points = enso_sepac_df.Date
    
    # Load CERES local SEPac radiation data for the standard time period
    println("Loading local SEPac CERES radiation data...")
    ceres_data = CSV.read("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/SEPac_SST/sepac_ceres_flux_time_series.csv", DataFrame)
    
    # Extract radiation data and preprocess (detrend and deseasonalize)
    net_rad = ceres_data[!,"SEPac_Net_Radiation"]
    sw_rad = ceres_data[!,"SEPac_Net_SW"] 
    lw_rad = ceres_data[!,"SEPac_Minus_LW"]
    ceres_time = ceres_data[!,"Date"]
    
    println("Loaded local SEPac radiation data: $(length(net_rad)) points from $(minimum(ceres_time)) to $(maximum(ceres_time))")
    
    # Preprocess radiation data (detrend and deseasonalize)
    println("Preprocessing local SEPac radiation data (detrend and deseasonalize)...")
    net_rad_processed = preprocess_data(net_rad, ceres_time)
    sw_rad_processed = preprocess_data(sw_rad, ceres_time)
    lw_rad_processed = preprocess_data(lw_rad, ceres_time)
    
    # Ensure time alignment
    # Convert both to year-month pairs for comparison (ignoring day)
    enso_ym = [(year(tp), month(tp)) for tp in time_points]
    ceres_ym = [(year(ct), month(ct)) for ct in ceres_time]
    
    # Find common year-month pairs
    common_ym = intersect(enso_ym, ceres_ym)
    sort!(common_ym)
    
    if length(common_ym) == 0
        error("No overlapping year-month pairs between ENSO-SEPac data and CERES data")
    end
    
    println("Found $(length(common_ym)) overlapping year-month pairs")
    
    # Find indices for alignment based on year-month pairs
    enso_indices = [findfirst(==(ym), enso_ym) for ym in common_ym]
    ceres_indices = [findfirst(==(ym), ceres_ym) for ym in common_ym]
    
    # Extract aligned data
    sepac_sst_aligned = sepac_sst[enso_indices]
    oni_optimal_aligned = oni_optimal[enso_indices]
    sepac_residual_aligned = sepac_residual[enso_indices]
    net_rad_aligned = net_rad_processed[ceres_indices]
    sw_rad_aligned = sw_rad_processed[ceres_indices]
    lw_rad_aligned = lw_rad_processed[ceres_indices]
    
    println("Aligned data: $(length(net_rad_aligned)) time points")
    
    # Define lag range
    lag_range = -24:24
    
    println("Calculating correlations for lags -24 to 24...")
    
    # Calculate correlations for each combination
    # Net radiation vs indices
    corr_net_sepac = calculate_lagged_correlations(net_rad_aligned, sepac_sst_aligned, lag_range)
    corr_net_oni = calculate_lagged_correlations(net_rad_aligned, oni_optimal_aligned, lag_range)
    corr_net_residual = calculate_lagged_correlations(net_rad_aligned, sepac_residual_aligned, lag_range)
    
    # SW radiation vs indices  
    corr_sw_sepac = calculate_lagged_correlations(sw_rad_aligned, sepac_sst_aligned, lag_range)
    corr_sw_oni = calculate_lagged_correlations(sw_rad_aligned, oni_optimal_aligned, lag_range)
    corr_sw_residual = calculate_lagged_correlations(sw_rad_aligned, sepac_residual_aligned, lag_range)
    
    # LW radiation vs indices
    corr_lw_sepac = calculate_lagged_correlations(lw_rad_aligned, sepac_sst_aligned, lag_range)
    corr_lw_oni = calculate_lagged_correlations(lw_rad_aligned, oni_optimal_aligned, lag_range)
    corr_lw_residual = calculate_lagged_correlations(lw_rad_aligned, sepac_residual_aligned, lag_range)
    
    # Create subplots in the style of plot_sepac_radiation_correlation_vs_lag
    println("Creating correlation plots...")
    
    # Define plot styling
    colors = [:blue, :red, :green]
    line_styles = [:solid, :dash, :dot]
    radiation_labels = ["Local Net Radiation", "Local SW Radiation", "Local LW Radiation"]
    
    # Plot 1: SEPac SST Index vs all radiation types
    p1 = plot(size=(800, 600), dpi=300)
    plot!(p1, collect(lag_range), corr_net_sepac,
          label=radiation_labels[1],
          color=colors[1],
          linestyle=line_styles[1],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p1, collect(lag_range), corr_sw_sepac,
          label=radiation_labels[2],
          color=colors[2],
          linestyle=line_styles[2],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p1, collect(lag_range), corr_lw_sepac,
          label=radiation_labels[3],
          color=colors[3],
          linestyle=line_styles[3],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p1, xlabel="SEPac SST Index Lag (months)",
          ylabel="Correlation with Local Radiation",
          title="SEPac SST Index-Local Radiation Correlations vs Lag",
          grid=true,
          legend=:topright,
          xlims=(-25, 25))
    hline!(p1, [0], color=:black, linestyle=:dashdot, alpha=0.5, label="")
    
    # Plot 2: ONI at optimal lag vs all radiation types  
    p2 = plot(size=(800, 600), dpi=300)
    plot!(p2, collect(lag_range), corr_net_oni,
          label=radiation_labels[1],
          color=colors[1],
          linestyle=line_styles[1],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p2, collect(lag_range), corr_sw_oni,
          label=radiation_labels[2],
          color=colors[2],
          linestyle=line_styles[2],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p2, collect(lag_range), corr_lw_oni,
          label=radiation_labels[3],
          color=colors[3],
          linestyle=line_styles[3],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p2, xlabel="ONI (optimal lag) Lag (months)",
          ylabel="Correlation with Local Radiation",
          title="ONI (optimal lag)-Local Radiation Correlations vs Lag",
          grid=true,
          legend=:topright,
          xlims=(-25, 25))
    hline!(p2, [0], color=:black, linestyle=:dashdot, alpha=0.5, label="")
    
    # Plot 3: SEPac SST Residual vs all radiation types
    p3 = plot(size=(800, 600), dpi=300)
    plot!(p3, collect(lag_range), corr_net_residual,
          label=radiation_labels[1],
          color=colors[1],
          linestyle=line_styles[1],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p3, collect(lag_range), corr_sw_residual,
          label=radiation_labels[2],
          color=colors[2],
          linestyle=line_styles[2],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p3, collect(lag_range), corr_lw_residual,
          label=radiation_labels[3],
          color=colors[3],
          linestyle=line_styles[3],
          linewidth=2,
          marker=:circle,
          markersize=3)
    plot!(p3, xlabel="SEPac SST Residual Lag (months)",
          ylabel="Correlation with Local Radiation",
          title="SEPac SST Residual-Local Radiation Correlations vs Lag",
          grid=true,
          legend=:topright,
          xlims=(-25, 25))
    hline!(p3, [0], color=:black, linestyle=:dashdot, alpha=0.5, label="")
    
    # Combine plots vertically
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1400))
    
    # Save plot
    output_dir = "../../vis/sepac_sst_local_rad_effects"
    mkpath(output_dir)
    plot_filename = joinpath(output_dir, "local_radiation_enso_sepac_lagged_correlations.png")
    savefig(combined_plot, plot_filename)
    println("Plot saved to: $plot_filename")
    
    # Print summary statistics
    println("\n=== CORRELATION SUMMARY ===")
    println("All data detrended and deseasonalized before correlation calculation")
    println("Data period: $(minimum([Date(ym[1], ym[2], 1) for ym in common_ym])) to $(maximum([Date(ym[1], ym[2], 1) for ym in common_ym]))")
    println("Number of time points: $(length(net_rad_aligned))")
    println("Lag range: $(minimum(lag_range)) to $(maximum(lag_range)) months")
    
    # Find maximum correlations for each radiation type
    println("\nMaximum absolute correlations (Local SEPac Radiation):")
    println("Local Net Radiation:")
    max_net_sepac_idx = argmax(abs.(skipmissing(corr_net_sepac)))
    max_net_oni_idx = argmax(abs.(skipmissing(corr_net_oni)))
    max_net_residual_idx = argmax(abs.(skipmissing(corr_net_residual)))
    println("  vs SEPac SST: $(round(corr_net_sepac[max_net_sepac_idx], digits=3)) at lag $(lag_range[max_net_sepac_idx])")
    println("  vs ONI: $(round(corr_net_oni[max_net_oni_idx], digits=3)) at lag $(lag_range[max_net_oni_idx])")
    println("  vs SEPac Residual: $(round(corr_net_residual[max_net_residual_idx], digits=3)) at lag $(lag_range[max_net_residual_idx])")
    
    println("Local SW Radiation:")
    max_sw_sepac_idx = argmax(abs.(skipmissing(corr_sw_sepac)))
    max_sw_oni_idx = argmax(abs.(skipmissing(corr_sw_oni)))
    max_sw_residual_idx = argmax(abs.(skipmissing(corr_sw_residual)))
    println("  vs SEPac SST: $(round(corr_sw_sepac[max_sw_sepac_idx], digits=3)) at lag $(lag_range[max_sw_sepac_idx])")
    println("  vs ONI: $(round(corr_sw_oni[max_sw_oni_idx], digits=3)) at lag $(lag_range[max_sw_oni_idx])")
    println("  vs SEPac Residual: $(round(corr_sw_residual[max_sw_residual_idx], digits=3)) at lag $(lag_range[max_sw_residual_idx])")
    
    println("Local LW Radiation:")
    max_lw_sepac_idx = argmax(abs.(skipmissing(corr_lw_sepac)))
    max_lw_oni_idx = argmax(abs.(skipmissing(corr_lw_oni)))
    max_lw_residual_idx = argmax(abs.(skipmissing(corr_lw_residual)))
    println("  vs SEPac SST: $(round(corr_lw_sepac[max_lw_sepac_idx], digits=3)) at lag $(lag_range[max_lw_sepac_idx])")
    println("  vs ONI: $(round(corr_lw_oni[max_lw_oni_idx], digits=3)) at lag $(lag_range[max_lw_oni_idx])")
    println("  vs SEPac Residual: $(round(corr_lw_residual[max_lw_residual_idx], digits=3)) at lag $(lag_range[max_lw_residual_idx])")
    
    return combined_plot
end

# Run the analysis
if true
    cd(@__DIR__)
    plot_correlation_matrix()
end
