using Plots, Statistics, StatsBase, Dates
using LinearAlgebra
using CSV, DataFrames, Dictionaries

# Include necessary modules
olddir = pwd()
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")
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

# Define CERES radiation variables
ceres_vars = ["gtoa_net_all_mon", "global_net_sw", "gtoa_lw_all_mon"]
radiation_labels = ["Net Radiation", "SW Radiation", "LW Radiation"]
radiation_short_labels = ["net", "sw", "lw"]

pls_lags = -24:24  # Extended range for PLS analysis

"""
    analyze_eli_radiation_effects()

Analyze the correlation between ELI at different lags (-6, -3, 0, 3, 6) and CERES radiation data.
Performs 1-component PLS regression and creates visualization plots.
"""
function analyze_eli_radiation_effects()
    # Use time period defined in constants.jl
    # time_period is already defined as (Date(2000, 3), Date(2022, 4))
    
    # Define ELI lags to analyze
    eli_lags = [-6, -3, 0, 3, 6]  # in months
    eli_columns = ["ELI_Lag$lag" for lag in eli_lags]
    
    
    println("Loading ELI data...")
    # Load ELI data
    eli_data, eli_coords = load_eli_data(time_period; lags=eli_lags)
    
    println("Loading CERES data...")
    # Load CERES global radiation data
    ceres_data, ceres_coords = load_ceres_data(ceres_vars, time_period)
    
    # Check data availability
    println("ELI data keys: ", keys(eli_data))
    println("CERES data keys: ", keys(ceres_data))
    println("ELI time length: ", length(eli_coords["time"]))
    println("CERES time length: ", length(ceres_coords["time"]))
    
    # Both datasets should already be filtered to the same time period
    # Use CERES times as reference since they align with the time period
    time_points = Date.(ceres_coords["time"])
    
    println("Time points for analysis: ", length(time_points))
    
    # Create plots for each radiation variable
    plots_list = []
    pls_weights_results = Dict{String, Vector{Float64}}()
    
    for (rad_idx, (rad_var, rad_label, rad_short)) in enumerate(zip(ceres_vars, radiation_labels, radiation_short_labels))
        println("Processing $rad_label...")
        
        # Extract and preprocess radiation data
        radiation_data = ceres_data[rad_var]
        radiation_processed = preprocess_data(radiation_data, time_points)
        
        # Create subplot layout (2 rows, 5 columns for 5 lags + PLS)
        subplot_layout = (2, cld(length(eli_lags) + 1, 2))
        p = plot(layout=subplot_layout, size=(1200, 800))
        plot!(p, plot_title="$rad_label: ELI Lag Correlations and PLS Analysis", 
              plot_titlefontsize=16)
        
        # Store correlations and ELI data for PLS
        correlations = Float64[]
        
        # Calculate correlations for each lag
        for (lag_idx, (lag, eli_col)) in enumerate(zip(eli_lags, eli_columns))
            if haskey(eli_data, eli_col)
                # Extract and preprocess ELI data
                eli_data_raw = eli_data[eli_col]
                eli_processed = preprocess_data(eli_data_raw, time_points)
                
                # Calculate correlation
                corr_val = cor(eli_processed, radiation_processed)
                push!(correlations, corr_val)
                
                # Create time series plot - both ELI and radiation on same axis
                plot!(p[lag_idx], time_points, eli_processed,
                      label="ELI Lag $lag", color=:blue, linewidth=2,
                      xlabel="Time", 
                      ylabel="Standardized Values",
                      title="Lag $lag (r=$(round(corr_val, digits=3)))")
                
                # Add radiation data on same axis
                plot!(p[lag_idx], time_points, radiation_processed,
                      label=rad_short, color=:red, linewidth=2)
            else
                push!(correlations, NaN)
                plot!(p[lag_idx], title="Lag $lag (no data)")
            end
        end
        
        # Perform 1-component PLS regression
        println("Performing PLS regression for $rad_label...")

        pls_eli_data, _ = load_eli_data(time_period; lags = collect(pls_lags))
        eli_matrix = hcat([pls_eli_data["ELI_Lag$lag"] for lag in pls_lags]...)
        
        # Use all data for PLS
        X_pls = eli_matrix
        Y_pls = radiation_processed
        
        # Fit PLS model with 1 component
        try
            # Use make_pls_regressor - it will handle normalization internally
            pls_result = make_pls_regressor(X_pls, Y_pls, 1; print_updates=false)
            
            # Extract X-scores (first component)
            x_scores = pls_result.X_scores[:, 1]
            
            # Store X weights for later saving
            x_weights = pls_result.X_weights[:, 1]  # First component weights
            pls_weights_results[rad_label] = x_weights
            
            # Calculate and display correlation for PLS
            pls_corr = cor(x_scores, Y_pls)

            pls_plot = p[length(eli_lags) + 1]  # PLS plot at the end

            # Plot X-scores and radiation as time series - both on same axis
            norm_factor = std(Y_pls)/std(x_scores)
            plot!(pls_plot, time_points, x_scores .* norm_factor,
                  label="PLS X-Score", color=:green, linewidth=2,
                  xlabel="Time",
                  ylabel="Standardized Values",
                  title="PLS (r=$(round(pls_corr, digits=3)))")
            
            # Add radiation data on same axis
            plot!(pls_plot, time_points, Y_pls,
                  label=rad_short, color=:red, linewidth=2)
            
        catch e
            println("PLS failed for $rad_label: ", e)
            plot!(pls_plot, title="PLS Analysis (failed)")
        end
        
        push!(plots_list, p)
    end
    
    return plots_list, pls_weights_results
end

# Run the analysis
println("Starting ELI-Radiation analysis...")
plots_list, pls_weights = analyze_eli_radiation_effects()

# Create vis directory path and subdirectory for these plots
vis_dir = joinpath(@__DIR__, "../../vis/eli_radiation_effects/")
eli_plots_dir = joinpath(vis_dir, "global_rad")

# Create the subdirectory if it doesn't exist
if !isdir(eli_plots_dir)
    mkpath(eli_plots_dir)
end

# Display plots
for (i, p) in enumerate(plots_list)
    display(p)
    
    # Save plots in dedicated subdirectory
    radiation_names = ["net", "sw", "lw"]
    output_path = joinpath(eli_plots_dir, "eli_$(radiation_names[i])_radiation_analysis.png")
    savefig(p, output_path)
    println("Saved plot for $(radiation_names[i]) radiation to: $output_path")
end

# Save PLS X weights to text file
weights_file_path = joinpath(eli_plots_dir, "pls_x_weights.txt")
open(weights_file_path, "w") do file
    println(file, "PLS X-Weights for ELI-Radiation Analysis")
    println(file, "=" ^ 50)
    println(file, "Time period: $(time_period[1]) to $(time_period[2])")
    println(file, "ELI Lags analyzed: -6, -3, 0, 3, 6 months")
    println(file, "Component: 1 (first PLS component)")
    println(file, "")
    
    for (rad_type, weights) in pls_weights
        println(file, "$(rad_type):")
        println(file, "  ELI Lag -6: $(round(weights[1], digits=4))")
        println(file, "  ELI Lag -3: $(round(weights[2], digits=4))")
        println(file, "  ELI Lag  0: $(round(weights[3], digits=4))")
        println(file, "  ELI Lag  3: $(round(weights[4], digits=4))")
        println(file, "  ELI Lag  6: $(round(weights[5], digits=4))")
        println(file, "")
    end
    
    println(file, "Note: X-weights indicate the relative importance and direction")
    println(file, "of each ELI lag in the PLS component for predicting radiation.")
end

println("Saved PLS X-weights to: $weights_file_path")

println("Analysis complete!")

"""
    plot_correlation_vs_lag()

Plot correlation between ELI at different lags and each radiation variable on a single axis.
"""
function plot_correlation_vs_lag()
    println("Calculating correlations across lag range...")
    
    # Define extended lag range
    extended_lags = collect(-24:24)
    eli_extended_columns = ["ELI_Lag$lag" for lag in extended_lags]
    
    # Load ELI data with extended lags
    eli_data_extended, _ = load_eli_data(time_period; lags=extended_lags)
    
    # Load CERES data
    ceres_data, ceres_coords = load_ceres_data(ceres_vars, time_period)
    time_points = Date.(ceres_coords["time"])
    
    # Initialize correlation storage
    correlations_matrix = zeros(length(extended_lags), length(ceres_vars))
    
    # Calculate correlations for each radiation variable and lag
    for (rad_idx, rad_var) in enumerate(ceres_vars)
        println("Processing $(radiation_labels[rad_idx])...")
        
        # Extract and preprocess radiation data
        radiation_data = ceres_data[rad_var]
        radiation_processed = preprocess_data(radiation_data, time_points)
        
        for (lag_idx, (lag, eli_col)) in enumerate(zip(extended_lags, eli_extended_columns))
            if haskey(eli_data_extended, eli_col)
                # Extract and preprocess ELI data
                eli_data_raw = eli_data_extended[eli_col]
                eli_processed = preprocess_data(eli_data_raw, time_points)
                
                # Calculate correlation
                corr_val = cor(eli_processed, radiation_processed)
                correlations_matrix[lag_idx, rad_idx] = corr_val
            else
                correlations_matrix[lag_idx, rad_idx] = NaN
            end
        end
    end
    
    # Create the correlation plot
    p = plot(size=(800, 600), dpi=300)
    
    colors = [:blue, :red, :green]
    line_styles = [:solid, :dash, :dot]
    
    for (rad_idx, (rad_label, rad_short)) in enumerate(zip(radiation_labels, radiation_short_labels))
        plot!(p, extended_lags, correlations_matrix[:, rad_idx],
              label=rad_label,
              color=colors[rad_idx],
              linestyle=line_styles[rad_idx],
              linewidth=2,
              marker=:circle,
              markersize=3)
    end
    
    # Add formatting
    plot!(p, xlabel="ELI Lag (months)",
          ylabel="Correlation with ELI",
          title="ELI-Radiation Correlations vs Lag",
          grid=true,
          legend=:topright,
          xlims=(-25, 25))
    
    # Add zero line for reference
    hline!(p, [0], color=:black, linestyle=:dashdot, alpha=0.5, label="")
    
    return p
end

# Create and save the correlation vs lag plot
correlation_plot = plot_correlation_vs_lag()
display(correlation_plot)

# Save the plot
correlation_plot_path = joinpath(eli_plots_dir, "eli_radiation_correlation_vs_lag.png")
savefig(correlation_plot, correlation_plot_path)
println("Saved correlation vs lag plot to: $correlation_plot_path")
