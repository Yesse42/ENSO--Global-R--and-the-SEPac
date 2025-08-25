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
Visualize the regression slopes between three time series from the ENSO-SEPac analysis
and gridded radiation data (net, SW, and LW).

The three time series are:
1. SEPac SST Index (detrended & deseasonalized)
2. ONI at optimal lag (detrended & deseasonalized) 
3. SEPac SST Residual (after removing ONI influence)

Each row of the plot will be a different radiation type.
Each column will show regression slopes with a different time series.
"""

# Set up output directory
visdir = "../../vis/draft_v1_vis/radiation_correlations/"
if !isdir(visdir)
    mkpath(visdir)
end

# Define radiation variables and their descriptive names
radiation_vars = ["toa_net_all_mon", "gridded_net_sw", "toa_lw_all_mon"]
radiation_labels = ["Net Radiation", "SW Radiation", "LW Radiation"]

# Define the three time series names
time_series_names = ["sepac_sst_index", "oni_at_optimal_lag", "sepac_sst_residual"]
time_series_labels = ["SEPac SST Index", "ONI at Optimal Lag", "SEPac SST Residual"]

function analyze_radiation_regression_slopes()
    println("Starting radiation regression slope analysis...")
    
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
    
    # Load all CERES radiation data at once
    println("Loading CERES radiation data...")
    ceres_data, ceres_coords = load_ceres_data(radiation_vars, time_period)
    
    # Get CERES time coordinates
    ceres_times = ceres_coords["time"]
    println("CERES data period: $(minimum(ceres_times)) to $(maximum(ceres_times))")
    println("Number of CERES time points: $(length(ceres_times))")
    
    # Prepare for regression slope calculations
    println("Preprocessing CERES radiation data...")
    float_times = @. year(ceres_times) + (month(ceres_times) - 1) / 12
    month_groups = groupfind(month.(ceres_times))
    
    all_slope_slices = []
    all_slope_subtitles = []
    
    # First, collect all valid radiation variables for processing
    valid_rad_vars = []
    valid_rad_labels = []
    valid_rad_data = []
    
    for (rad_idx, rad_var) in enumerate(radiation_vars)
        println("Processing $rad_var...")
        
        # Get radiation data
        rad_data = ceres_data[rad_var]
        println("  Data dimensions: $(size(rad_data))")
        
        # Check if it's gridded data or global time series
        if ndims(rad_data) == 1
            println("  Warning: $rad_var is a global time series, skipping spatial analysis")
            continue
        end
        
        # Apply sign convention: multiply longwave by -1 to make all radiations positive downward
        if rad_var == "toa_lw_all_mon"
            println("  Applying sign convention: multiplying LW radiation by -1 (positive downward)")
            rad_data = -1 .* rad_data
        end
        
        # Detrend and deseasonalize the radiation data
        println("  Detrending and deseasonalizing...")
        rad_data_processed = copy(rad_data)
        for slice in eachslice(rad_data_processed, dims=(1,2))
            detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
        end
        
        push!(valid_rad_vars, rad_var)
        push!(valid_rad_labels, radiation_labels[rad_idx])
        push!(valid_rad_data, rad_data_processed)
    end
    
    # Check if we have any valid regression slopes to plot
    if isempty(valid_rad_vars)
        error("No gridded radiation data found for spatial regression analysis")
    end
    
    # Now calculate regression slopes in column-major order (time series first, then radiation types)
    for (ts_idx, ts_name) in enumerate(time_series_names)
        println("Calculating regression slopes with $(time_series_labels[ts_idx])...")
        ts_data = time_series_data[ts_name]
        
        for (rad_idx, (rad_var, rad_label, rad_data_processed)) in enumerate(zip(valid_rad_vars, valid_rad_labels, valid_rad_data))
            println("  Processing $(rad_label)...")
            
            # Calculate spatial regression slopes
            slopes = zeros(Float64, size(rad_data_processed)[1:2])
            for i in 1:size(rad_data_processed, 1)
                for j in 1:size(rad_data_processed, 2)
                    rad_series = rad_data_processed[i, j, :]
                    fit_result = least_squares_fit(ts_data, rad_series)
                    slopes[i, j] = fit_result.slope
                end
            end
            
            push!(all_slope_slices, slopes)
            push!(all_slope_subtitles, "$(rad_label) - $(time_series_labels[ts_idx])")
        end
    end
    
    # Create the plot layout: rows = radiation types, columns = time series
    n_valid_radiation_vars = length(valid_rad_vars)
    n_time_series = length(time_series_names)
    layout = (n_valid_radiation_vars, n_time_series)  # rows × columns
    
    println("Creating regression slope plot...")
    println("Layout: $(layout[1]) rows (radiation types) × $(layout[2]) columns (time series)")
    println("Number of subplots: $(length(all_slope_slices))")
    
    # Get coordinate information for plotting
    lat = Float64.(ceres_coords["latitude"])
    lon = Float64.(ceres_coords["longitude"])
    
    # Create the comprehensive regression slope plot
    slope_fig = plot_multiple_levels(lat, lon, all_slope_slices, layout;
                                    subtitles=all_slope_subtitles,
                                    colorbar_label="Regression Slope (W/m²/unit)")
    
    # Save the plot
    plot_filename = joinpath(visdir, "radiation_regression_slopes_with_three_indices.png")
    slope_fig.suptitle("Radiation Regression Slopes: Radiation Types (rows) × Time Series (columns)", fontsize=16)
    slope_fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close(slope_fig)
    
    println("Plot saved to: $plot_filename")
    
    # Print summary statistics
    println("\n=== SUMMARY ===")
    println("Analysis period: $(minimum(ceres_times)) to $(maximum(ceres_times))")
    println("Number of time points used: $(length(ceres_times))")
    println("Time series analyzed:")
    for (i, (name, label)) in enumerate(zip(time_series_names, time_series_labels))
        data = time_series_data[name]
        println("  $i. $label: mean = $(round(mean(data), digits=3)), std = $(round(std(data), digits=3))")
    end
    println("Radiation variables analyzed:")
    for (i, (var, label)) in enumerate(zip(valid_rad_vars, valid_rad_labels))
        println("  $i. $label ($var): gridded data")
    end
    println("Plot layout: $(layout[1]) radiation types × $(layout[2]) time series")
    println("Regression slopes show radiation change (W/m²) per unit change in time series")
    
    return slope_fig, time_series_data, ceres_times
end

# Run the analysis
println("Starting Radiation Regression Slope Analysis")
println("="^60)
analyze_radiation_regression_slopes()
