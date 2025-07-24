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
    analyze_sepac_enso_correlations()

Analyze the correlation between SEPac SST index at different lags (-6, -3, 0, 3, 6) and ENSO at lag 0.
Performs 1-component PLS regression and creates visualization plots.
"""
function analyze_sepac_enso_correlations()
    # Use time period defined in constants.jl
    # time_period is already defined as (Date(2000, 3), Date(2022, 4))
    
    # Define SEPac SST index lags to analyze
    sepac_lags = [-6, -3, 0, 3, 6]
    sepac_columns = ["SEPac_SST_Index_Lag$lag" for lag in sepac_lags]
    
    # Define ENSO at lag 0
    enso_lag = 0
    enso_column = "oni_lag_$enso_lag"
    
    println("Loading SEPac SST index data...")
    # Load SEPac SST index data
    sepac_data, sepac_coords = load_sepac_sst_index(time_period; lags=sepac_lags)
    
    println("Loading ENSO data...")
    # Load ENSO data at lag 0
    enso_data, enso_coords = load_enso_data(time_period; lags=[enso_lag])
    
    # Check data availability
    println("SEPac SST data keys: ", keys(sepac_data))
    println("ENSO data keys: ", keys(enso_data))
    println("SEPac SST time length: ", length(sepac_coords["time"]))
    println("ENSO time length: ", length(enso_coords["time"]))
    
    # Use ENSO times as reference since they should align with the time period
    time_points = Date.(enso_coords["time"])
    
    println("Time points for analysis: ", length(time_points))
    
    # Check if ENSO data exists
    if !haskey(enso_data, enso_column)
        error("ENSO data at lag $enso_lag not found")
    end
    
    # Extract and preprocess ENSO data
    enso_data_raw = enso_data[enso_column]
    enso_processed = preprocess_data(enso_data_raw, time_points)
    
    println("Processing SEPac SST Index vs ENSO correlations...")
    
    # Create subplot layout (2 rows, 3 columns for 5 lags + PLS)
    subplot_layout = @layout [a b c; d e f]
    p = plot(layout=subplot_layout, size=(1200, 800))
    plot!(p, plot_title="SEPac SST Index Lag Correlations with ENSO (Lag 0) and PLS Analysis", 
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
            corr_val = cor(sepac_processed, enso_processed)
            push!(correlations, corr_val)
            
            # Store for PLS
            sepac_matrix[:, lag_idx] = sepac_processed
            
            # Create time series plot - both SEPac SST and ENSO on same axis
            plot!(p[lag_idx], time_points, sepac_processed,
                  label="SEPac SST Lag $lag", color=:blue, linewidth=2,
                  xlabel="Time", 
                  ylabel="Standardized Values",
                  title="Lag $lag (r=$(round(corr_val, digits=3)))")
            
            # Add ENSO data on same axis
            plot!(p[lag_idx], time_points, enso_processed,
                  label="ENSO Lag 0", color=:red, linewidth=2)
        else
            push!(correlations, NaN)
            plot!(p[lag_idx], title="Lag $lag (no data)")
        end
    end
    
    # Perform 1-component PLS regression
    println("Performing PLS regression...")
    
    # Use all SEPac lags for PLS to predict ENSO
    X_pls = sepac_matrix
    Y_pls = enso_processed
    
    # Fit PLS model with 1 component
    try
        # Use make_pls_regressor - it will handle normalization internally
        pls_result = make_pls_regressor(X_pls, Y_pls, 1; print_updates=false)
        
        # Extract X-scores (first component)
        x_scores = pls_result.X_scores[:, 1]
        
        # Store X weights for later saving
        x_weights = pls_result.X_weights[:, 1]  # First component weights
        
        # Calculate and display correlation for PLS
        pls_corr = cor(x_scores, Y_pls)
        
        # Plot X-scores and ENSO as time series - both on same axis
        plot!(p[6], time_points, x_scores,
              label="PLS X-Score", color=:green, linewidth=2,
              xlabel="Time",
              ylabel="Standardized Values",
              title="PLS (r=$(round(pls_corr, digits=3)))")
        
        # Add ENSO data on same axis
        plot!(p[6], time_points, Y_pls,
              label="ENSO Lag 0", color=:red, linewidth=2)
        
        # Print correlation summary
        println("\nCorrelation Summary:")
        println("=" ^ 50)
        for (lag, corr) in zip(sepac_lags, correlations)
            println("SEPac SST Lag $lag vs ENSO Lag 0: $(round(corr, digits=4))")
        end
        println("PLS X-Score vs ENSO Lag 0: $(round(pls_corr, digits=4))")
        
        # Print PLS weights
        println("\nPLS X-Weights:")
        println("=" ^ 30)
        for (lag, weight) in zip(sepac_lags, x_weights)
            println("SEPac SST Lag $lag: $(round(weight, digits=4))")
        end
        
        return p, correlations, x_weights, pls_corr
        
    catch e
        println("PLS failed: ", e)
        plot!(p[6], title="PLS Analysis (failed)")
        return p, correlations, Float64[], NaN
    end
end

# Run the analysis
println("Starting SEPac SST Index vs ENSO correlation analysis...")
plot_result, correlations, pls_weights, pls_corr = analyze_sepac_enso_correlations()

# Create vis directory path and subdirectory for these plots
vis_dir = joinpath(@__DIR__, "../../vis/sepac_radiation_effects/")
sepac_enso_dir = joinpath(vis_dir, "sepac_enso_correlations")

# Create the subdirectory if it doesn't exist
if !isdir(sepac_enso_dir)
    mkpath(sepac_enso_dir)
end

# Display plot
display(plot_result)

# Save plot
output_path = joinpath(sepac_enso_dir, "sepac_sst_enso_correlation_analysis.png")
savefig(plot_result, output_path)
println("Saved correlation plot to: $output_path")

# Save correlation results and PLS weights to text file
results_file_path = joinpath(sepac_enso_dir, "correlation_results.txt")
open(results_file_path, "w") do file
    println(file, "SEPac SST Index vs ENSO Correlation Analysis")
    println(file, "=" ^ 50)
    println(file, "Time period: $(time_period[1]) to $(time_period[2])")
    println(file, "SEPac SST Index Lags analyzed: -6, -3, 0, 3, 6 months")
    println(file, "ENSO: Lag 0 (contemporary)")
    println(file, "")
    
    println(file, "Individual Correlations:")
    println(file, "-" ^ 25)
    sepac_lags = [-6, -3, 0, 3, 6]
    for (lag, corr) in zip(sepac_lags, correlations)
        println(file, "  SEPac SST Lag $(lpad(lag, 2)): $(round(corr, digits=4))")
    end
    println(file, "")
    
    if !isempty(pls_weights) && !isnan(pls_corr)
        println(file, "PLS Analysis (1 component):")
        println(file, "-" ^ 30)
        println(file, "  PLS X-Score vs ENSO correlation: $(round(pls_corr, digits=4))")
        println(file, "")
        println(file, "  PLS X-Weights:")
        for (lag, weight) in zip(sepac_lags, pls_weights)
            println(file, "    SEPac SST Lag $(lpad(lag, 2)): $(round(weight, digits=4))")
        end
        println(file, "")
        println(file, "Note: X-weights indicate the relative importance and direction")
        println(file, "of each SEPac SST Index lag in the PLS component for predicting ENSO.")
    else
        println(file, "PLS Analysis: Failed")
    end
end

println("Saved correlation results to: $results_file_path")

println("Analysis complete!")
