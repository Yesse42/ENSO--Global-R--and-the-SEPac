using Plots, Statistics, StatsBase, Dates, DataFrames, CSV, LinearAlgebra

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
    find_optimal_oni_lag(oni_data_dict, sepac_sst_data, lag_range)

Find the ONI lag that correlates maximally with SEPac SST index.
Uses pre-loaded ONI data at different lags.
Returns the optimal lag, correlation value, and correlation array for all lags.
"""
function find_optimal_oni_lag(oni_data_dict, sepac_sst_data, lag_range)
    correlations = Float64[]
    
    # Calculate correlation for each lag
    for lag in lag_range
        lag_key = "oni_lag_$lag"
        
        if haskey(oni_data_dict, lag_key)
            oni_series = oni_data_dict[lag_key]
            
            # Calculate correlation
            corr_val = cor(oni_series, sepac_sst_data)
            push!(correlations, corr_val)
        else
            # If lag not available, set to NaN
            push!(correlations, NaN)
        end
    end
    
    # Find the lag with maximum absolute correlation
    abs_correlations = abs.(correlations)
    valid_indices = .!isnan.(abs_correlations)
    
    if !any(valid_indices)
        error("No valid correlations found for any lag")
    end
    
    max_idx = argmax(abs_correlations[valid_indices])
    # Get the actual index in the full array
    valid_lag_indices = findall(valid_indices)
    actual_max_idx = valid_lag_indices[max_idx]
    
    optimal_lag = lag_range[actual_max_idx]
    optimal_correlation = correlations[actual_max_idx]
    
    return optimal_lag, optimal_correlation, correlations, lag_range
end

"""
    get_lagged_oni_series(oni_data_dict, optimal_lag)

Get the ONI series at the optimal lag.
"""
function get_lagged_oni_series(oni_data_dict, optimal_lag)
    lag_key = "oni_lag_$optimal_lag"
    
    if haskey(oni_data_dict, lag_key)
        return oni_data_dict[lag_key]
    else
        error("ONI lag $optimal_lag not found in loaded data")
    end
end

"""
    regress_out_oni_influence(sepac_sst_data, oni_lagged_data)

Regress out the influence of ONI from SEPac SST using linear regression.
Returns the residual SEPac SST series and fit statistics.
"""
function regress_out_oni_influence(sepac_sst_data, oni_lagged_data)
    # Use the least squares fit function from utils
    fit = least_squares_fit(oni_lagged_data, sepac_sst_data)
    
    # Calculate predicted values: y_pred = slope * x + intercept
    y_predicted = fit.slope .* oni_lagged_data .+ fit.intercept
    
    # Calculate residuals: y_residual = y_actual - y_predicted
    residuals = sepac_sst_data .- y_predicted
    
    # Calculate R² for regression quality
    ss_res = sum((sepac_sst_data .- y_predicted).^2)  # Sum of squares of residuals
    ss_tot = sum((sepac_sst_data .- mean(sepac_sst_data)).^2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    return residuals, (fit=fit, r_squared=r_squared, predicted=y_predicted)
end

"""
    analyze_enso_sepac_correlation()

Main function to analyze correlation between ONI and SEPac SST indices.
"""
function analyze_enso_sepac_correlation()
    println("Starting ENSO-SEPac SST correlation analysis...")
    
    # Define extended time span for loading all available data
    extended_timespan = (Date(0), Date(10000000, 12, 31))
    
    # Define lag range to test
    max_lag = 24
    lag_range = -max_lag:max_lag
    
    # Load ONI data for all lags over extended time span
    println("Loading ONI data for lags $(-max_lag) to $max_lag over extended timespan...")
    oni_data_extended, oni_coords_extended = load_enso_data(extended_timespan; lags=collect(lag_range))
    
    # Load SEPac SST data over extended time span
    println("Loading SEPac SST data over extended timespan...")
    sepac_data_extended, sepac_coords_extended = load_sepac_sst_index(extended_timespan; lags=[0])
    sepac_series_extended = sepac_data_extended["SEPac_SST_Index_Lag0"]
    
    # Find the overlapping time period for all datasets
    println("Finding overlapping time periods...")
    oni_times = oni_coords_extended["time"]
    sepac_times = sepac_coords_extended["time"]
    
    # Convert to Date objects for easier comparison
    # ONI times are DateTime with 14-day offset, so extract year-month and use day 1
    oni_dates = Date.(year.(oni_times), month.(oni_times), 1)
    # SEPac times should already be Date objects, but let's be sure
    sepac_dates = Date.(sepac_times)
    
    # Find common dates
    common_dates = intersect(oni_dates, sepac_dates)
    
    if isempty(common_dates)
        error("No overlapping dates found between ONI and SEPac SST data")
    end
    
    println("Found $(length(common_dates)) overlapping time points")
    println("Extended data period: $(minimum(common_dates)) to $(maximum(common_dates))")
    
    # Create indices for the common dates
    oni_indices = [findfirst(==(d), oni_dates) for d in common_dates]
    sepac_indices = [findfirst(==(d), sepac_dates) for d in common_dates]
    
    # Extract aligned data over the extended period
    time_points_extended = oni_times[oni_indices]
    sepac_series_aligned = sepac_series_extended[sepac_indices]
    
    # Extract aligned ONI data for all lags
    oni_data_aligned = Dictionary()
    for lag in lag_range
        lag_key = "oni_lag_$lag"
        if haskey(oni_data_extended, lag_key)
            oni_series = oni_data_extended[lag_key][oni_indices]
            set!(oni_data_aligned, lag_key, oni_series)
        end
    end
    
    # Now subset to the standard time period for correlation analysis
    println("Subsetting to standard time period for correlation analysis...")
    standard_start, standard_end = time_period
    
    # Find indices within the standard time period
    standard_mask = (Date.(time_points_extended) .>= standard_start) .& (Date.(time_points_extended) .< standard_end)
    
    if !any(standard_mask)
        error("No data found in the standard time period $(standard_start) to $(standard_end)")
    end
    
    println("Standard period has $(sum(standard_mask)) time points")
    
    # Extract data for the standard period
    time_points_standard = time_points_extended[standard_mask]
    sepac_series_standard = sepac_series_aligned[standard_mask]
    
    oni_data_standard = Dictionary()
    for lag in lag_range
        lag_key = "oni_lag_$lag"
        if haskey(oni_data_aligned, lag_key)
            oni_series = oni_data_aligned[lag_key][standard_mask]
            set!(oni_data_standard, lag_key, oni_series)
        end
    end
    
    println("Preprocessing data for standard period (detrend and deseasonalize)...")
    # Preprocess SEPac SST series for standard period
    sepac_processed_standard = preprocess_data(sepac_series_standard, time_points_standard)
    
    # Preprocess ONI series for standard period
    oni_processed_standard = Dictionary()
    for lag in lag_range
        lag_key = "oni_lag_$lag"
        if haskey(oni_data_standard, lag_key)
            oni_series = oni_data_standard[lag_key]
            processed_series = preprocess_data(oni_series, time_points_standard)
            set!(oni_processed_standard, lag_key, processed_series)
        end
    end
    
    println("Finding optimal ONI lag using standard period...")
    # Find optimal ONI lag using standard period data
    optimal_lag, optimal_corr, all_correlations, lag_range_used = find_optimal_oni_lag(
        oni_processed_standard, sepac_processed_standard, lag_range
    )
    
    println("Optimal ONI lag: $optimal_lag months")
    println("Optimal correlation: $(round(optimal_corr, digits=3))")
    
    # Plot correlation as a function of lag
    println("Plotting correlation vs lag...")
    lag_plot = plot(collect(lag_range_used), all_correlations,
                   xlabel="Lag (months)", 
                   ylabel="Correlation Coefficient",
                   label = "",
                   title="ONI-SEPac SST Correlation vs Lag (Standard Period)",
                   linewidth=2,
                   marker=:circle,
                   markersize=3,
                   grid=true)
    
    # Mark the optimal lag
    vline!([optimal_lag], color=:red, linewidth=2, linestyle=:dash, 
           label="Optimal Lag: $optimal_lag months")
    
    # Display the plot
    display(lag_plot)
    
    # Save the lag correlation plot
    output_dir = "../../vis/draft_v1_vis"
    mkpath(output_dir)
    lag_plot_filename = joinpath(output_dir, "oni_sepac_correlation_vs_lag.png")
    savefig(lag_plot, lag_plot_filename)
    println("Lag correlation plot saved to: $lag_plot_filename")
    
    # Now preprocess the extended period data using the optimal lag
    println("Preprocessing extended period data...")
    sepac_processed_extended = preprocess_data(sepac_series_aligned, time_points_extended)
    
    # Get the optimal lag ONI series for the extended period
    optimal_lag_key = "oni_lag_$optimal_lag"
    if !haskey(oni_data_aligned, optimal_lag_key)
        error("Optimal lag $optimal_lag not found in extended data")
    end
    
    oni_optimal_extended = preprocess_data(oni_data_aligned[optimal_lag_key], time_points_extended)
    
    # Regress out ONI influence using extended data
    println("Regressing out ONI influence over extended period...")
    sepac_residual_extended, regression_stats = regress_out_oni_influence(sepac_processed_extended, oni_optimal_extended)
    
    println("Regression R² (extended period): $(round(regression_stats.r_squared, digits=3))")
    
    # Create 3x1 plot using standard period for visualization
    println("Creating visualization using standard period...")
    sepac_standard_for_plot = sepac_processed_standard
    oni_standard_for_plot = oni_processed_standard[optimal_lag_key]

    oni_predict_sepac_fit = least_squares_fit(oni_standard_for_plot, sepac_standard_for_plot)
    oni_scaled_to_sepac = oni_predict_sepac_fit.slope .* oni_standard_for_plot .+ oni_predict_sepac_fit.intercept
    display(oni_predict_sepac_fit)

    sepac_residual_standard = sepac_standard_for_plot .- oni_scaled_to_sepac
    
    # Calculate common y-axis limits for all three panels
    all_y_values = vcat(sepac_standard_for_plot, oni_scaled_to_sepac, sepac_residual_standard)
    y_min = minimum(all_y_values)
    y_max = maximum(all_y_values)
    y_margin = (y_max - y_min) * 0.05  # Add 5% margin
    common_ylims = (y_min - y_margin, y_max + y_margin)
    
    p1 = plot(time_points_standard, sepac_standard_for_plot,
              title="SEPac SST Index (detrended & deseasonalized)",
              ylabel="SEPac SST Index",
              linewidth=2,
              color=:blue,
              grid=true,
              legend=false,
              ylims=common_ylims)
    
    p2 = plot(time_points_standard, oni_scaled_to_sepac,
              title="SEPac SST Index Variability explained by $optimal_lag month lag ONI",
              ylabel="ONI",
              linewidth=2,
              color=:red,
              grid=true,
              legend=false,
              ylims=common_ylims)
    
    p3 = plot(time_points_standard, sepac_residual_standard,
              title="SEPac SST Index with ONI Influence Removed",
              ylabel="SEPac SST Residual",
              xlabel="Time",
              linewidth=2,
              color=:green,
              grid=true,
              legend=false,
              ylims=common_ylims)
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    
    # Save the plot
    output_dir = "../../vis/draft_v1_vis"
    mkpath(output_dir)
    plot_filename = joinpath(output_dir, "enso_sepac_correlation_analysis.png")
    savefig(combined_plot, plot_filename)
    println("Plot saved to: $plot_filename")
    
    # Create and save DataFrame using the extended period data
    println("Saving extended period data...")
    results_df = DataFrame(
        Date = time_points_extended,
        optimal_oni_lag = optimal_lag,
        sepac_sst_index = sepac_processed_extended,
        oni_at_optimal_lag = oni_optimal_extended,
        sepac_sst_residual = sepac_residual_extended
    )
    
    # Save to CSV
    data_output_dir = "../../data/v1draft_saved_data"
    mkpath(data_output_dir)
    csv_filename = joinpath(data_output_dir, "enso_sepac_correlation_results.csv")
    CSV.write(csv_filename, results_df)
    println("Data saved to: $csv_filename")
    
    # Print summary statistics
    println("\n=== SUMMARY ===")
    println("Optimal ONI lag: $optimal_lag months (calculated from standard period)")
    println("Correlation at optimal lag: $(round(optimal_corr, digits=3)) (from standard period)")
    println("Regression R² (extended period): $(round(regression_stats.r_squared, digits=3))")
    println("Standard analysis period: $(minimum(time_points_standard)) to $(maximum(time_points_standard))")
    println("Extended data period saved: $(minimum(time_points_extended)) to $(maximum(time_points_extended))")
    println("Number of time points in standard period: $(length(time_points_standard))")
    println("Number of time points in extended period: $(length(time_points_extended))")
    
    return results_df, optimal_lag, optimal_corr, combined_plot
end

# Run the analysis
analyze_enso_sepac_correlation()