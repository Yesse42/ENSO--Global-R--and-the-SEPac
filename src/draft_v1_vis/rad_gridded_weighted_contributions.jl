using Plots, Statistics, StatsBase, Dates, SplitApplyCombine
using CSV, DataFrames, Dictionaries

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

"""
This script will:
Calculate correlations and weighted components of radiation with three time series 
from the ENSO-SEPac analysis.

The three time series are:
1. SEPac SST Index (detrended & deseasonalized)
2. ONI at optimal lag (detrended & deseasonalized) 
3. SEPac SST Residual (after removing ONI influence)

For each time series, the script calculates:
- Net radiation correlation
- SW weighted component: correlation(SW) × std(SW)/std(SW + LW)
- LW weighted component: correlation(LW) × std(LW)/std(SW + LW)

Each row of the plot will be Net/SW/LW components.
Each column will show components for a different time series.
"""

# Set up output directory
visdir = "../../vis/draft_v1_vis/radiation_weighted_contributions/"
if !isdir(visdir)
    mkpath(visdir)
end

# Define the radiation variables
net_var = "toa_net_all_mon"
sw_var = "gridded_net_sw"
lw_var = "toa_lw_all_mon"
radiation_vars = [net_var, sw_var, lw_var]

# Define the three time series names
time_series_names = ["sepac_sst_index", "oni_at_optimal_lag", "sepac_sst_residual"]
time_series_labels = ["SEPac SST Index", "ONI at Optimal Lag", "SEPac SST Residual"]

function calculate_weighted_contributions()
    println("Starting weighted radiation contribution analysis...")
    
    # Use standard time period from constants.jl
    println("Using standard time period: $(time_period[1]) to $(time_period[2])")
    
    # Load the three time series from CSV
    println("Loading time series data from CSV...")
    csv_file = "../../data/v1draft_saved_data/enso_sepac_correlation_results.csv"
    df = CSV.read(csv_file, DataFrame)
    
    # Extract time information and filter for standard time period
    times = DateTime.(df.Date)
    csv_dates = Date.(times)
    
    # Filter CSV data for standard time period
    time_mask = (csv_dates .>= time_period[1]) .& (csv_dates .< time_period[2])
    filtered_df = df[time_mask, :]
    filtered_times = times[time_mask]
    
    println("Filtered to $(length(filtered_times)) time points within standard period")
    
    # Extract time series data
    time_series_data = Dictionary()
    for col_name in time_series_names
        series_data = filtered_df[:, col_name]
        set!(time_series_data, col_name, series_data)
    end
    
    # Load all radiation data (net, SW, and LW)
    println("Loading CERES radiation data...")
    ceres_data, ceres_coords = load_ceres_data(radiation_vars, time_period)
    
    # Get CERES time coordinates
    ceres_times = ceres_coords["time"]
    println("CERES data period: $(minimum(ceres_times)) to $(maximum(ceres_times))")
    println("Number of CERES time points: $(length(ceres_times))")
    
    # Prepare for processing
    println("Preprocessing CERES radiation data...")
    float_times = @. year(ceres_times) + (month(ceres_times) - 1) / 12
    month_groups = groupfind(month.(ceres_times))
    
    # Get and process net radiation data
    println("Processing Net radiation data...")
    net_data = ceres_data[net_var]
    println("  Net data dimensions: $(size(net_data))")
    
    # Check if it's gridded data
    if ndims(net_data) != 3
        error("Net radiation data must be gridded (3D: lon, lat, time)")
    end
    
    # Detrend and deseasonalize net data
    println("  Detrending and deseasonalizing Net data...")
    net_data_processed = copy(net_data)
    for slice in eachslice(net_data_processed, dims=(1,2))
        detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
    end
    
    # Get and process SW radiation data
    println("Processing SW radiation data...")
    sw_data = ceres_data[sw_var]
    println("  SW data dimensions: $(size(sw_data))")
    
    # Check if it's gridded data
    if ndims(sw_data) != 3
        error("SW radiation data must be gridded (3D: lon, lat, time)")
    end
    
    # Detrend and deseasonalize SW data
    println("  Detrending and deseasonalizing SW data...")
    sw_data_processed = copy(sw_data)
    for slice in eachslice(sw_data_processed, dims=(1,2))
        detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
    end
    
    # Get and process LW radiation data
    println("Processing LW radiation data...")
    lw_data = ceres_data[lw_var]
    println("  LW data dimensions: $(size(lw_data))")
    
    # Check if it's gridded data
    if ndims(lw_data) != 3
        error("LW radiation data must be gridded (3D: lon, lat, time)")
    end
    
    # Apply sign convention: multiply longwave by -1 to make positive downward
    println("  Applying sign convention: multiplying LW radiation by -1 (positive downward)")
    lw_data = -1 .* lw_data
    
    # Detrend and deseasonalize LW data
    println("  Detrending and deseasonalizing LW data...")
    lw_data_processed = copy(lw_data)
    for slice in eachslice(lw_data_processed, dims=(1,2))
        detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
    end
    
    # Calculate correlations and weighted contributions for each time series
    all_contrib_slices = []
    all_contrib_subtitles = []
    all_verification_slices = []
    all_verification_subtitles = []
    
    # Calculate SW-LW correlation at each grid point (only need to do this once)
    println("Calculating SW-LW correlations at each grid point...")
    sw_lw_correlations = zeros(size(sw_data_processed, 1), size(sw_data_processed, 2))
    for i in 1:size(sw_data_processed, 1)
        for j in 1:size(sw_data_processed, 2)
            sw_series = sw_data_processed[i, j, :]
            lw_series = lw_data_processed[i, j, :]
            sw_lw_correlations[i, j] = cor(sw_series, lw_series)
        end
    end
    sw_lw_correlations = Float64.(sw_lw_correlations)
    sw_lw_correlations = replace!(sw_lw_correlations, NaN => 0)
    
    for (ts_idx, ts_name) in enumerate(time_series_names)
        println("Calculating correlations and weighted contributions for $(time_series_labels[ts_idx])...")
        ts_data = time_series_data[ts_name]
        
        # Calculate net radiation correlation (unchanged)
        println("  Calculating net radiation correlation...")
        net_corrs = cor.(eachslice(net_data_processed, dims=(1,2)), Ref(ts_data))
        net_corrs = Float64.(net_corrs)
        
        # Add net correlation to results
        push!(all_contrib_slices, net_corrs)
        push!(all_contrib_subtitles, "Net - $(time_series_labels[ts_idx])")
        
        # Calculate correlations with SW and LW for weighting
        println("  Calculating SW and LW correlations...")
        sw_corrs = cor.(eachslice(sw_data_processed, dims=(1,2)), Ref(ts_data))
        lw_corrs = cor.(eachslice(lw_data_processed, dims=(1,2)), Ref(ts_data))
        
        # Calculate standard deviations for weighting
        println("  Calculating statistics for weighting...")
        sw_stds = std.(eachslice(sw_data_processed, dims=(1,2)))
        lw_stds = std.(eachslice(lw_data_processed, dims=(1,2)))
        
        # Calculate standard deviation of (SW + LW) directly
        println("  Calculating std(SW + LW) directly...")
        sw_plus_lw_stds = zeros(size(sw_stds))
        for i in 1:size(sw_data_processed, 1)
            for j in 1:size(sw_data_processed, 2)
                sw_series = sw_data_processed[i, j, :]
                lw_series = lw_data_processed[i, j, :]
                combined_series = sw_series .+ lw_series
                sw_plus_lw_stds[i, j] = std(combined_series)
            end
        end
        
        # Use directly calculated std(SW + LW) as denominator
        println("  Using std(SW + LW) as weighting denominator...")
        denominators = sw_plus_lw_stds
        
        # Handle potential division by zero
        denominators = replace!(denominators, 0 => NaN)
        
        # Calculate weighted components
        println("  Calculating weighted components...")
        sw_weighted = sw_corrs .* sw_stds ./ denominators
        lw_weighted = lw_corrs .* lw_stds ./ denominators
        
        # Convert to Float64 and handle any remaining issues
        sw_weighted = Float64.(sw_weighted)
        lw_weighted = Float64.(lw_weighted)
        
        # Replace any remaining NaN/Inf with 0
        sw_weighted = replace!(sw_weighted, NaN => 0, Inf => 0, -Inf => 0)
        lw_weighted = replace!(lw_weighted, NaN => 0, Inf => 0, -Inf => 0)
        
        # Add to collection for plotting
        push!(all_contrib_slices, sw_weighted)
        push!(all_contrib_subtitles, "SW - $(time_series_labels[ts_idx])")
        
        push!(all_contrib_slices, lw_weighted)
        push!(all_contrib_subtitles, "LW - $(time_series_labels[ts_idx])")
        
        # Calculate sum of weighted components for verification
        sw_lw_sum = sw_weighted .+ lw_weighted
        
        # Add verification data
        push!(all_verification_slices, net_corrs)
        push!(all_verification_subtitles, "Net Correlation - $(time_series_labels[ts_idx])")
        
        push!(all_verification_slices, sw_lw_sum)
        push!(all_verification_subtitles, "SW + LW Sum - $(time_series_labels[ts_idx])")
        
        # Calculate difference for verification
        difference = net_corrs .- sw_lw_sum
        push!(all_verification_slices, difference)
        push!(all_verification_subtitles, "Difference (Net - Sum) - $(time_series_labels[ts_idx])")
    end
    
    # Create the plot layout: rows = Net/SW/LW, columns = time series
    # We have 3 rows (Net, SW weighted, LW weighted) × 3 columns (time series)
    layout = (3, length(time_series_names))  # 3 rows × 3 columns
    
    println("Creating radiation component plot...")
    println("Layout: $(layout[1]) rows (Net/SW/LW) × $(layout[2]) columns (time series)")
    println("Number of subplots: $(length(all_contrib_slices))")
    
    # Get coordinate information for plotting
    lat = Float64.(ceres_coords["latitude"])
    lon = Float64.(ceres_coords["longitude"])
    
    # Create the radiation component plot
    contrib_fig = plot_multiple_levels(lat, lon, all_contrib_slices, layout;
                                     subtitles=all_contrib_subtitles,
                                     colorbar_label="Radiation Component")
    
    # Save the plot
    plot_filename = joinpath(visdir, "radiation_components.png")
    contrib_fig.suptitle("Radiation Components: Net Correlation & SW/LW Weighted Components", fontsize=16)
    contrib_fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close(contrib_fig)
    
    println("Plot saved to: $plot_filename")
    
    # Create verification plot: Net vs Sum of weighted components
    verification_layout = (3, length(time_series_names))  # 3 rows × 3 columns
    
    println("Creating verification plot...")
    println("Verification layout: $(verification_layout[1]) rows (Net/Sum/Difference) × $(verification_layout[2]) columns (time series)")
    println("Number of verification subplots: $(length(all_verification_slices))")
    
    # Create the verification plot
    verification_fig = plot_multiple_levels(lat, lon, all_verification_slices, verification_layout;
                                          subtitles=all_verification_subtitles,
                                          colorbar_label="Correlation Value")
    
    # Save the verification plot
    verification_filename = joinpath(visdir, "radiation_verification.png")
    verification_fig.suptitle("Verification: Net Correlation vs Sum of Weighted Components", fontsize=16)
    verification_fig.savefig(verification_filename, dpi=300, bbox_inches="tight")
    plt.close(verification_fig)
    
    println("Verification plot saved to: $verification_filename")
    
    # Create SW-LW correlation plot
    println("Creating SW-LW correlation plot...")
    
    # Create a simple single plot for SW-LW correlations
    sw_lw_fig = plot_multiple_levels(lat, lon, [sw_lw_correlations], (1, 1);
                                   subtitles=["SW-LW Correlation"],
                                   colorbar_label="Correlation Coefficient")
    
    # Save the SW-LW correlation plot
    sw_lw_filename = joinpath(visdir, "sw_lw_correlations.png")
    sw_lw_fig.suptitle("Correlation between SW and LW Radiation", fontsize=16)
    sw_lw_fig.savefig(sw_lw_filename, dpi=300, bbox_inches="tight")
    plt.close(sw_lw_fig)
    
    println("SW-LW correlation plot saved to: $sw_lw_filename")
    
    # Calculate and print summary statistics
    println("\n=== SUMMARY ===")
    println("Analysis period: $(minimum(ceres_times)) to $(maximum(ceres_times))")
    println("Number of time points used: $(length(ceres_times))")
    println("Time series analyzed:")
    for (i, (name, label)) in enumerate(zip(time_series_names, time_series_labels))
        data = time_series_data[name]
        println("  $i. $label: mean = $(round(mean(data), digits=3)), std = $(round(std(data), digits=3))")
    end
    
    # Print contribution statistics for each time series
    contrib_idx = 1
    verification_idx = 1
    for (ts_idx, ts_label) in enumerate(time_series_labels)
        net_corr = all_contrib_slices[contrib_idx]
        sw_weighted = all_contrib_slices[contrib_idx + 1]
        lw_weighted = all_contrib_slices[contrib_idx + 2]
        
        # Get verification data
        net_verification = all_verification_slices[verification_idx]
        sum_verification = all_verification_slices[verification_idx + 1]
        difference_verification = all_verification_slices[verification_idx + 2]
        
        # Calculate global statistics (excluding NaN/zeros)
        net_valid = net_corr[isfinite.(net_corr) .& (net_corr .!= 0)]
        sw_valid = sw_weighted[isfinite.(sw_weighted) .& (sw_weighted .!= 0)]
        lw_valid = lw_weighted[isfinite.(lw_weighted) .& (lw_weighted .!= 0)]
        diff_valid = difference_verification[isfinite.(difference_verification)]
        
        println("\n$ts_label results:")
        if !isempty(net_valid)
            println("  Net: mean = $(round(mean(net_valid), digits=4)), std = $(round(std(net_valid), digits=4)), range = [$(round(minimum(net_valid), digits=4)), $(round(maximum(net_valid), digits=4))]")
        else
            println("  Net: no valid values")
        end
        
        if !isempty(sw_valid)
            println("  SW: mean = $(round(mean(sw_valid), digits=4)), std = $(round(std(sw_valid), digits=4)), range = [$(round(minimum(sw_valid), digits=4)), $(round(maximum(sw_valid), digits=4))]")
        else
            println("  SW: no valid values")
        end
        
        if !isempty(lw_valid)
            println("  LW: mean = $(round(mean(lw_valid), digits=4)), std = $(round(std(lw_valid), digits=4)), range = [$(round(minimum(lw_valid), digits=4)), $(round(maximum(lw_valid), digits=4))]")
        else
            println("  LW: no valid values")
        end
        
        # Print verification statistics
        if !isempty(diff_valid)
            println("  Verification - Net vs Sum difference: mean = $(round(mean(diff_valid), digits=6)), std = $(round(std(diff_valid), digits=6)), max_abs = $(round(maximum(abs.(diff_valid)), digits=6))")
        else
            println("  Verification: no valid differences")
        end
        
        contrib_idx += 3
        verification_idx += 3
    end
    
    # Print SW-LW correlation statistics
    sw_lw_valid = sw_lw_correlations[isfinite.(sw_lw_correlations) .& (sw_lw_correlations .!= 0)]
    println("\nSW-LW Correlation Statistics:")
    if !isempty(sw_lw_valid)
        println("  Mean = $(round(mean(sw_lw_valid), digits=4))")
        println("  Std = $(round(std(sw_lw_valid), digits=4))")
        println("  Range = [$(round(minimum(sw_lw_valid), digits=4)), $(round(maximum(sw_lw_valid), digits=4))]")
        println("  Median = $(round(median(sw_lw_valid), digits=4))")
    else
        println("  No valid SW-LW correlations")
    end
    
    println("\nFormulas used:")
    println("Net = correlation(Net)")
    println("SW = correlation(SW) × std(SW) / std(SW + LW)")
    println("LW = correlation(LW) × std(LW) / std(SW + LW)")
    
    return contrib_fig, verification_fig, sw_lw_fig, time_series_data, ceres_times, all_contrib_slices, all_verification_slices, sw_lw_correlations
end

# Run the analysis
println("Starting Radiation Component Analysis")
println("="^50)
calculate_weighted_contributions()
