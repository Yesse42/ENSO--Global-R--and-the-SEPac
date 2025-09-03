using Plots, Statistics, StatsBase, Dates
using LinearAlgebra

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

"""
    analyze_sepac_radiation_effects()

Analyze the correlation between SEPac SST index at different lags (-6, -3, 0, 3, 6) and CERES radiation data.
Performs 1-component PLS regression and creates visualization plots.
"""
function analyze_sepac_radiation_effects()
    # Use time period defined in constants.jl
    # time_period is already defined as (Date(2000, 3), Date(2022, 4))
    
    # Define SEPac SST index lags to analyze
    sepac_lags = [-6, -3, 0, 3, 6]
    sepac_columns = ["SEPac_SST_Index_Lag$lag" for lag in sepac_lags]
    
    # Define CERES radiation variables
    ceres_vars = ["gtoa_net_all_mon", "global_net_sw", "gtoa_lw_all_mon"]
    radiation_labels = ["Net Radiation", "SW Radiation", "LW Radiation"]
    radiation_short_labels = ["net", "sw", "lw"]
    
    println("Loading SEPac SST index data...")
    # Load SEPac SST index data
    sepac_data, sepac_coords = load_sepac_sst_index(time_period; lags=sepac_lags)
    
    println("Loading CERES data...")
    # Load CERES global radiation data
    ceres_data, ceres_coords = load_ceres_data(ceres_vars, time_period)
    
    # Check data availability
    println("SEPac SST data keys: ", keys(sepac_data))
    println("CERES data keys: ", keys(ceres_data))
    println("SEPac SST time length: ", length(sepac_coords["time"]))
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
        
        # Create subplot layout (2 rows, 3 columns for 5 lags + PLS)
        subplot_layout = @layout [a b c; d e f]
        p = plot(layout=subplot_layout, size=(1200, 800))
        plot!(p, plot_title="$rad_label: SEPac SST Index Lag Correlations and PLS Analysis", 
              plot_titlefontsize=16)
        
        # Store correlations and SEPac SST data for PLS
        correlations = Float64[]
        sepac_matrix = zeros(length(time_points), length(sepac_lags))
        
        # Calculate correlations for each lag
        for (lag_idx, (lag, sepac_col)) in enumerate(zip(sepac_lags, sepac_columns))
            if haskey(sepac_data, sepac_col)
                # Extract and preprocess SEPac SST data
                sepac_data_raw = sepac_data[sepac_col]
                sepac_processed = preprocess_data(sepac_data_raw, time_points)
                
                # Calculate correlation
                corr_val = cor(sepac_processed, radiation_processed)
                push!(correlations, corr_val)
                
                # Store for PLS
                sepac_matrix[:, lag_idx] = sepac_processed
                
                # Create time series plot - both SEPac SST and radiation on same axis
                plot!(p[lag_idx], time_points, sepac_processed,
                      label="SEPac SST Lag $lag", color=:blue, linewidth=2,
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
        
        # Use all data for PLS
        X_pls = sepac_matrix
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
            
            # Plot X-scores and radiation as time series - both on same axis
            plot!(p[6], time_points, x_scores,
                  label="PLS X-Score", color=:green, linewidth=2,
                  xlabel="Time",
                  ylabel="Standardized Values",
                  title="PLS (r=$(round(pls_corr, digits=3)))")
            
            # Add radiation data on same axis
            plot!(p[6], time_points, Y_pls,
                  label=rad_short, color=:red, linewidth=2)
            
        catch e
            println("PLS failed for $rad_label: ", e)
            plot!(p[6], title="PLS Analysis (failed)")
        end
        
        push!(plots_list, p)
    end
    
    return plots_list, pls_weights_results
end

# Run the analysis
println("Starting SEPac SST Index-Radiation analysis...")
plots_list, pls_weights = analyze_sepac_radiation_effects()

# Create vis directory path and subdirectory for these plots
vis_dir = joinpath(@__DIR__, "../../vis/sepac_radiation_effects/")
sepac_plots_dir = joinpath(vis_dir, "global_rad")

# Create the subdirectory if it doesn't exist
if !isdir(sepac_plots_dir)
    mkpath(sepac_plots_dir)
end

# Display plots
for (i, p) in enumerate(plots_list)
    display(p)
    
    # Save plots in dedicated subdirectory
    radiation_names = ["net", "sw", "lw"]
    output_path = joinpath(sepac_plots_dir, "sepac_$(radiation_names[i])_radiation_analysis.png")
    savefig(p, output_path)
    println("Saved plot for $(radiation_names[i]) radiation to: $output_path")
end

# Save PLS X weights to text file
weights_file_path = joinpath(sepac_plots_dir, "pls_x_weights.txt")
open(weights_file_path, "w") do file
    println(file, "PLS X-Weights for SEPac SST Index-Radiation Analysis")
    println(file, "=" ^ 60)
    println(file, "Time period: $(time_period[1]) to $(time_period[2])")
    println(file, "SEPac SST Index Lags analyzed: -6, -3, 0, 3, 6 months")
    println(file, "Component: 1 (first PLS component)")
    println(file, "")
    
    for (rad_type, weights) in pls_weights
        println(file, "$(rad_type):")
        println(file, "  SEPac SST Lag -6: $(round(weights[1], digits=4))")
        println(file, "  SEPac SST Lag -3: $(round(weights[2], digits=4))")
        println(file, "  SEPac SST Lag  0: $(round(weights[3], digits=4))")
        println(file, "  SEPac SST Lag  3: $(round(weights[4], digits=4))")
        println(file, "  SEPac SST Lag  6: $(round(weights[5], digits=4))")
        println(file, "")
    end
    
    println(file, "Note: X-weights indicate the relative importance and direction")
    println(file, "of each SEPac SST Index lag in the PLS component for predicting radiation.")
end

println("Saved PLS X-weights to: $weights_file_path")

ceres_vars = ["gtoa_net_all_mon", "global_net_sw", "gtoa_lw_all_mon"]
radiation_labels = ["Net Radiation", "SW Radiation", "LW Radiation"]
radiation_short_labels = ["net", "sw", "lw"]

"""
    plot_sepac_radiation_correlation_vs_lag()

Plot correlation between SEPac SST index at different lags and each radiation variable on a single axis.
"""
function plot_sepac_radiation_correlation_vs_lag()
    println("Calculating SEPac SST-Radiation correlations across lag range...")
    
    # Define extended lag range for SEPac SST
    extended_lags = collect(-24:24)
    sepac_extended_columns = ["SEPac_SST_Index_Lag$lag" for lag in extended_lags]
    
    # Load SEPac SST data with extended lags
    sepac_data_extended, _ = load_sepac_sst_index(time_period; lags=extended_lags)
    
    # Load CERES data
    ceres_data, ceres_coords = load_ceres_data(ceres_vars, time_period)
    time_points = Date.(ceres_coords["time"])
    
    # Initialize correlation storage
    correlations_matrix = zeros(length(extended_lags), length(ceres_vars))
    
    # Calculate correlations for each radiation variable and SEPac SST lag
    for (rad_idx, rad_var) in enumerate(ceres_vars)
        println("Processing $(radiation_labels[rad_idx])...")
        
        # Extract and preprocess radiation data
        radiation_data = ceres_data[rad_var]
        radiation_processed = preprocess_data(radiation_data, time_points)
        
        for (lag_idx, (lag, sepac_col)) in enumerate(zip(extended_lags, sepac_extended_columns))
            if haskey(sepac_data_extended, sepac_col)
                # Extract and preprocess SEPac SST data
                sepac_data_raw = sepac_data_extended[sepac_col]
                sepac_processed = preprocess_data(sepac_data_raw, time_points)
                
                # Calculate correlation
                corr_val = cor(sepac_processed, radiation_processed)
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
    plot!(p, xlabel="SEPac SST Index Lag (months)",
          ylabel="Correlation with Radiation",
          title="SEPac SST Index-Radiation Correlations vs Lag",
          grid=true,
          legend=:topright,
          xlims=(-25, 25),
          ylims=(-0.5, 0.5))
    
    # Add zero line for reference
    hline!(p, [0], color=:black, linestyle=:dashdot, alpha=0.5, label="")
    
    return p
end

# Create and save the SEPac-Radiation correlation vs lag plot
sepac_radiation_correlation_plot = plot_sepac_radiation_correlation_vs_lag()
display(sepac_radiation_correlation_plot)

# Save the plot
sepac_radiation_correlation_plot_path = joinpath(sepac_plots_dir, "sepac_radiation_correlation_vs_lag.png")
savefig(sepac_radiation_correlation_plot, sepac_radiation_correlation_plot_path)
println("Saved SEPac-Radiation correlation vs lag plot to: $sepac_radiation_correlation_plot_path")

println("Analysis complete!")
