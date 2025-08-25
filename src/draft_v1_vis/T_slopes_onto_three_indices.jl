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
and gridded temperature data at the surface and on pressure levels.

The three time series are:
1. SEPac SST Index (detrended & deseasonalized)
2. ONI at optimal lag (detrended & deseasonalized) 
3. SEPac SST Residual (after removing ONI influence)

Each row of the plot will be a different pressure level.
Each column will show regression slopes with a different time series.
"""

# Set up output directory
visdir = "../../vis/draft_v1_vis/temperature_correlations/"
if !isdir(visdir)
    mkpath(visdir)
end

# Define level names
level_names = ["sfc", "850hPa", "700hPa", "500hPa", "250hPa"]
stacked_level_vars = ["t2m", "t"]  # surface temperature and pressure level temperature

# Define the three time series names
time_series_names = ["sepac_sst_index", "oni_at_optimal_lag", "sepac_sst_residual"]
time_series_labels = ["SEPac SST Index", "ONI at Optimal Lag", "SEPac SST Residual"]

function analyze_temperature_regression_slopes()
    println("Starting temperature regression slope analysis...")
    
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
    
    # Load ERA5 temperature data for standard time period
    println("Loading ERA5 temperature data...")
    era5_data, era5_coords = load_era5_data(stacked_level_vars, time_period)
    
    # Get ERA5 time coordinates
    era5_times = era5_coords["time"]
    println("ERA5 data period: $(minimum(era5_times)) to $(maximum(era5_times))")
    println("Number of ERA5 time points: $(length(era5_times))")
    
    # Extract time series data (no need for complex alignment since both datasets cover the standard period)
    time_series_data = Dictionary()
    for col_name in time_series_names
        series_data = filtered_df[:, col_name]
        set!(time_series_data, col_name, series_data)
    end
    
    # Prepare ERA5 temperature data
    sfc_name = stacked_level_vars[1]  # "t2m"
    level_name = stacked_level_vars[2]  # "t"
    
    # Use ERA5 data directly (already filtered to standard time period)
    sfc_data_aligned = era5_data[sfc_name]
    level_data_aligned = era5_data[level_name]
    aligned_times = era5_times
    
    # Stack surface and level data
    sfc_dims = size(sfc_data_aligned)
    level_dims = size(level_data_aligned)
    
    # Add level dimension to surface data to match
    sfc_data_reshaped = reshape(sfc_data_aligned, sfc_dims[1], sfc_dims[2], 1, sfc_dims[3])
    
    # Concatenate along level dimension
    vertical_concat_data = cat(sfc_data_reshaped, level_data_aligned; dims=3)
    
    println("Combined temperature data shape: $(size(vertical_concat_data))")
    
    # Preprocess ERA5 data (detrend and deseasonalize)
    println("Preprocessing ERA5 temperature data...")
    float_times = @. year(aligned_times) + (month(aligned_times) - 1) / 12
    month_groups = groupfind(month.(aligned_times))
    
    for slice in eachslice(vertical_concat_data, dims=(1,2,3))
        detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
    end
    
    # Note: The CSV time series are already preprocessed, so we don't need to detrend them again
    
    # Calculate regression slopes for each time series and each level
    println("Calculating regression slopes...")
    lat = Float64.(era5_coords["latitude"])
    lon = Float64.(era5_coords["longitude"])
    
    all_slope_slices = []
    all_slope_subtitles = []
    
    # Loop through time series (columns) first, then levels (rows) for column-major ordering
    for (ts_idx, ts_name) in enumerate(time_series_names)
        for (level_idx, level_name) in enumerate(level_names)
            ts_data = time_series_data[ts_name]
            
            # Calculate regression slopes for this level
            level_temp_data = vertical_concat_data[:, :, level_idx, :]
            
            # Calculate slopes using least_squares_fit from utilfuncs
            slopes = zeros(Float64, size(level_temp_data)[1:2])
            for i in 1:size(level_temp_data, 1)
                for j in 1:size(level_temp_data, 2)
                    temp_series = level_temp_data[i, j, :]
                    fit_result = least_squares_fit(ts_data, temp_series)
                    slopes[i, j] = fit_result.slope
                end
            end
            
            push!(all_slope_slices, slopes)
            push!(all_slope_subtitles, "$(level_name) - $(time_series_labels[ts_idx])")
        end
    end
    
    # Create the plot layout: rows = levels, columns = time series
    layout = (length(level_names), length(time_series_names))  # 5 rows × 3 columns
    
    println("Creating regression slope plot...")
    println("Layout: $(layout[1]) rows (levels) × $(layout[2]) columns (time series)")
    println("Number of subplots: $(length(all_slope_slices))")
    
    # Create the comprehensive regression slope plot
    slope_fig = plot_multiple_levels(lat, lon, all_slope_slices, layout;
                                    subtitles=all_slope_subtitles,
                                    colorbar_label="Regression Slope (K/unit)")
    
    # Save the plot
    plot_filename = joinpath(visdir, "temperature_regression_slopes_with_three_indices.png")
    slope_fig.suptitle("Temperature Regression Slopes: Levels (rows) × Time Series (columns)", fontsize=16)
    slope_fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close(slope_fig)
    
    println("Plot saved to: $plot_filename")
    
    # Print summary statistics
    println("\n=== SUMMARY ===")
    println("Analysis period: $(minimum(aligned_times)) to $(maximum(aligned_times))")
    println("Number of time points used: $(length(aligned_times))")
    println("Time series analyzed:")
    for (i, (name, label)) in enumerate(zip(time_series_names, time_series_labels))
        data = time_series_data[name]
        println("  $i. $label: mean = $(round(mean(data), digits=3)), std = $(round(std(data), digits=3))")
    end
    println("Temperature levels analyzed: $(join(level_names, ", "))")
    println("Plot layout: $(layout[1]) levels × $(layout[2]) time series")
    println("Regression slopes show temperature change (K) per unit change in time series")
    
    return slope_fig, time_series_data, aligned_times
end

# Run the analysis
println("Starting Temperature Regression Slope Analysis")
println("="^60)
analyze_temperature_regression_slopes()
