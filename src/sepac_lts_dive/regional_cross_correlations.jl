"""
Generate cross-correlation matrices for all stratocumulus regions.
Uses the same variables and time period as compare_surface_aloft.jl:
- SW and LW local and nonlocal radiation
- θ_1000 (surface potential temperature) 
- θ_700 (aloft potential temperature)
- LTS_1000 (lower tropospheric stability)
- Global net radiation
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using Statistics, DataFrames, JLD2, CSV, Dates, PythonCall
@py import matplotlib.pyplot as plt, numpy as np

# Load region masks to get all available regions
mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)
base_regions = collect(keys(region_data["regional_masks_ceres"]))

# Add the special ENSO-removed region
valid_regions = vcat(base_regions, ["enso_removed_SEPac_feedback_definition"])

# Define radiation variables and time period (same as compare_surface_aloft.jl)
rad_variables = ["gtoa_net_all_mon", "gtoa_net_lw_mon", "gtoa_net_sw_mon"]
date_range = (Date(2002, 3, 1), Date(2022, 3, 31))
is_analysis_time(t) = in_time_period(t, date_range)

# Load global radiation time series
global_rad_data, global_coords = load_new_ceres_data(rad_variables, date_range)

# Data directories
local_data_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/local_region_time_series"
nonlocal_data_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/nonlocal_radiation_time_series"

# Output directory
output_dir = "../../vis/lts_global_rad/cross_correlation_matrices"
mkpath(output_dir)

function calculate_regional_correlations(region::String)
    """
    Calculate cross-correlation matrix for a specific region.
    Returns a dictionary with variable data and correlation matrix.
    """
    
    println("Processing region: $region")
    
    # Load ERA5 data with potential temperature variables
    era5_file = joinpath(local_data_dir, "era5_region_avg_$(region).csv")
    era5_df = CSV.read(era5_file, DataFrame)
    
    # Load local and nonlocal radiation data
    local_rad_file = joinpath(nonlocal_data_dir, "$(region)_local_radiation.csv")
    nonlocal_rad_file = joinpath(nonlocal_data_dir, "$(region)_nonlocal_radiation.csv")
    
    # Handle special case for ENSO-removed region
    if region == "enso_removed_SEPac_feedback_definition"
        # Use SEPac_feedback_definition mask areas for the ENSO-removed case
        mask_region = "SEPac_feedback_definition"
        local_rad_file = joinpath(nonlocal_data_dir, "$(mask_region)_local_radiation.csv")
        nonlocal_rad_file = joinpath(nonlocal_data_dir, "$(mask_region)_nonlocal_radiation.csv")

        # Filter out non-lag-0 columns and rename lag-0 columns to plain names
        era5_columns = names(era5_df)
        keep_columns = String[]
        rename_dict = Dict{String, String}()

        for col in era5_columns
            if col == "date"
                push!(keep_columns, col)
            elseif endswith(col, "_lag_0")
                # Keep lag-0 columns and prepare to rename them
                push!(keep_columns, col)
                plain_name = replace(col, "_lag_0" => "")
                rename_dict[col] = plain_name
            end
            # Skip all other columns (non-lag-0)
        end

        # Select only the columns we want to keep
        era5_df = select(era5_df, keep_columns)

        # Rename lag-0 columns to plain names
        rename!(era5_df, rename_dict)
    else
        mask_region = region
    end
    
    local_rad_df = CSV.read(local_rad_file, DataFrame)
    nonlocal_rad_df = CSV.read(nonlocal_rad_file, DataFrame)
    
    # Filter to analysis time period
    filter!(row -> is_analysis_time(row.date), era5_df)
    filter!(row -> is_analysis_time(row.date), local_rad_df)
    filter!(row -> is_analysis_time(row.date), nonlocal_rad_df)
    
    # Prepare time variables for detrending/deseasonalizing
    analysis_times = era5_df.date
    float_times = calc_float_time.(analysis_times)
    months = month.(analysis_times)
    
    # Extract and process atmospheric variables
    theta_1000 = copy(era5_df[!, "θ_1000"])      # Surface potential temperature
    neg_theta_1000 = -theta_1000                 # Negative surface potential temperature (as in compare_surface_aloft.jl)
    theta_700 = copy(era5_df[!, "θ_700"])        # Aloft potential temperature
    lts_1000 = copy(era5_df[!, "LTS_1000"])      # Lower tropospheric stability
    
    # Extract radiation variables
    local_sw = copy(local_rad_df[!, "toa_net_sw_mon"])
    local_lw = copy(local_rad_df[!, "toa_net_lw_mon"])
    nonlocal_sw = copy(nonlocal_rad_df[!, "gtoa_net_sw_mon"])
    nonlocal_lw = copy(nonlocal_rad_df[!, "gtoa_net_lw_mon"])
    
    # Process global net radiation
    global_times = round.(global_coords["time"], Month(1), RoundDown)
    time_mask = [t in analysis_times for t in global_times]
    global_net = global_rad_data["gtoa_net_all_mon"][time_mask]
    
    # Create dictionary of all variables
    variables = Dict(
        "neg_θ₁₀₀₀" => neg_theta_1000,
        "θ₇₀₀" => theta_700, 
        "LTS₁₀₀₀" => lts_1000,
        "Local SW" => local_sw,
        "Local LW" => local_lw,
        "Nonlocal SW" => nonlocal_sw,
        "Nonlocal LW" => nonlocal_lw,
        "Global Net" => global_net
    )
    
    # Detrend and deseasonalize all variables
    for (var_name, var_data) in variables
        detrend_and_deseasonalize!(var_data, float_times, months)
        println("  Processed: $var_name")
    end
    
    # Create variable names list for matrix ordering
    var_names = ["neg_θ₁₀₀₀", "θ₇₀₀", "LTS₁₀₀₀", "Local SW", "Local LW", "Nonlocal SW", "Nonlocal LW", "Global Net"]
    
    # Calculate correlation matrix
    n_vars = length(var_names)
    corr_matrix = zeros(n_vars, n_vars)
    
    for i in 1:n_vars
        for j in 1:n_vars
            corr_matrix[i, j] = cor(variables[var_names[i]], variables[var_names[j]])
        end
    end
    
    println("  Calculated correlation matrix ($(n_vars)×$(n_vars))")
    
    return Dict(
        "region" => region,
        "variables" => variables,
        "var_names" => var_names,
        "correlation_matrix" => corr_matrix,
        "analysis_times" => analysis_times
    )
end

function plot_correlation_matrix(region_data::Dict, save_path::String)
    """
    Create and save a correlation matrix heatmap for a region.
    """
    
    region = region_data["region"]
    var_names = region_data["var_names"]
    corr_matrix = region_data["correlation_matrix"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    
    # Set ticks and labels
    ax.set_xticks(0:(length(var_names)-1))
    ax.set_yticks(0:(length(var_names)-1))
    ax.set_xticklabels(var_names, rotation=45, ha="right")
    ax.set_yticklabels(var_names)
    
    # Add correlation values as text
    for i in 1:length(var_names)
        for j in 1:length(var_names)
            text_color = abs(corr_matrix[i, j]) > 0.5 ? "white" : "black"
            ax.text(j-1, i-1, string(round(corr_matrix[i, j], digits=2)), 
                   ha="center", va="center", color=text_color, fontsize=9)
        end
    end
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation Coefficient", rotation=270, labelpad=20)
    
    # Set title
    ax.set_title("Cross-Correlation Matrix: $region\n$(date_range[1]) to $(date_range[2])", 
                 pad=20, fontsize=14, fontweight="bold")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    println("  Saved correlation matrix: $save_path")
end

function create_summary_comparison(all_results::Vector{Dict})
    """
    Create summary plots comparing key correlations across all regions.
    """
    
    println("Creating summary comparison plots...")
    
    # Extract data for comparison
    region_names = [r["region"] for r in all_results]
    n_regions = length(region_names)
    
    # Extract key correlation pairs
    var_names = all_results[1]["var_names"]
    
    # Define key variable pairs to compare across regions
    key_pairs = [
        ("LTS₁₀₀₀", "Global Net"),
        ("neg_θ₁₀₀₀", "Global Net"), 
        ("θ₇₀₀", "Global Net"),
        ("Local SW", "neg_θ₁₀₀₀"),
        ("Local LW", "neg_θ₁₀₀₀"),
        ("Nonlocal SW", "neg_θ₁₀₀₀"),
        ("Nonlocal LW", "neg_θ₁₀₀₀")
    ]
    
    # Create comparison bar plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for (idx, (var1, var2)) in enumerate(key_pairs)
        ax = axes[idx]
        
        # Get correlation values for this pair across all regions
        correlations = Float64[]
        for result in all_results
            var_idx1 = findfirst(x -> x == var1, result["var_names"])
            var_idx2 = findfirst(x -> x == var2, result["var_names"])
            push!(correlations, result["correlation_matrix"][var_idx1, var_idx2])
        end
        
        # Create bar plot
        bars = ax.bar(1:n_regions, correlations, alpha=0.7)
        
        # Color bars by magnitude
        for (i, bar) in enumerate(bars)
            if correlations[i] < 0
                bar.set_color("red")
            else
                bar.set_color("blue")
            end
        end
        
        ax.set_title("$var1 vs $var2", fontsize=11, fontweight="bold")
        ax.set_ylabel("Correlation", fontsize=10)
        ax.set_xticks(1:n_regions)
        ax.set_xticklabels(region_names, rotation=45, ha="right", fontsize=9)
        ax.grid(true, alpha=0.3)
        ax.set_ylim(-1, 1)
        
        # Add value labels on bars
        for (i, corr) in enumerate(correlations)
            va_position = corr > 0 ? "bottom" : "top"
            ax.text(i+1, corr + 0.05*sign(corr), "$(round(corr, digits=2))", 
                   ha="center", va=va_position, fontsize=8)
        end
    end
    
    # Hide the last subplot (we only have 7 pairs)
    axes[7].set_visible(false)
    
    plt.suptitle("Cross-Regional Correlation Comparison\nKey Variable Pairs", 
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_file = joinpath(output_dir, "regional_correlation_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    println("Saved regional comparison: $comparison_file")
end

function create_correlation_summary_table(all_results::Vector{Dict})
    """
    Create a summary table of key correlations for all regions.
    """
    
    println("Creating correlation summary table...")
    
    # Define key correlations to extract
    key_correlations = [
        ("LTS₁₀₀₀", "Global Net", "LTS-Global Net"),
        ("neg_θ₁₀₀₀", "Global Net", "Surface θ-Global Net"),
        ("θ₇₀₀", "Global Net", "Aloft θ-Global Net"),
        ("Local SW", "Local LW", "Local SW-LW"),
        ("Nonlocal SW", "Nonlocal LW", "Nonlocal SW-LW"),
        ("neg_θ₁₀₀₀", "θ₇₀₀", "Surface-Aloft θ"),
        ("LTS₁₀₀₀", "neg_θ₁₀₀₀", "LTS-Surface θ"),
        ("LTS₁₀₀₀", "θ₇₀₀", "LTS-Aloft θ")
    ]
    
    # Create summary DataFrame
    summary_data = Dict{String, Vector{Float64}}()
    
    for (var1, var2, label) in key_correlations
        correlations = Float64[]
        for result in all_results
            var_idx1 = findfirst(x -> x == var1, result["var_names"])
            var_idx2 = findfirst(x -> x == var2, result["var_names"])
            push!(correlations, result["correlation_matrix"][var_idx1, var_idx2])
        end
        summary_data[label] = correlations
    end
    
    # Add region names
    region_names = [r["region"] for r in all_results]
    
    # Create and save table
    summary_df = DataFrame(summary_data)
    insertcols!(summary_df, 1, :Region => region_names)
    
    # Save as CSV
    table_file = joinpath(output_dir, "regional_correlations_summary.csv")
    CSV.write(table_file, summary_df)
    
    println("Saved correlation summary table: $table_file")
    
    # Print summary statistics
    println("\\nSummary Statistics:")
    println("-" ^ 60)
    for (var1, var2, label) in key_correlations
        corrs = summary_data[label]
        println("$label:")
        println("  Mean: $(round(mean(corrs), digits=3))")
        println("  Std:  $(round(std(corrs), digits=3))")
        println("  Min:  $(round(minimum(corrs), digits=3)) ($(region_names[argmin(corrs)]))")
        println("  Max:  $(round(maximum(corrs), digits=3)) ($(region_names[argmax(corrs)]))")
        println()
    end
    
    return summary_df
end

# Main execution
function run_correlation_analysis()
    """
    Run cross-correlation analysis for all regions.
    """
    
    println("="^80)
    println("CROSS-CORRELATION ANALYSIS FOR STRATOCUMULUS REGIONS")
    println("="^80)
    println("Time period: $(date_range[1]) to $(date_range[2])")
    println("Variables: neg_θ₁₀₀₀, θ₇₀₀, LTS₁₀₀₀, Local SW/LW, Nonlocal SW/LW, Global Net")
    println("Regions: $(join(valid_regions, ", "))")
    println()
    
    # Process all regions
    all_results = Dict{String, Any}[]
    
    for region in valid_regions
        try
            result = calculate_regional_correlations(region)
            push!(all_results, result)
            
            # Create and save correlation matrix plot
            matrix_file = joinpath(output_dir, "$(region)_correlation_matrix.png")
            plot_correlation_matrix(result, matrix_file)
            
            println("  ✓ Completed: $region")
            println()
            
        catch e
            println("  ✗ Failed: $region")
            println("    Error: $e")
            println()
            continue
        end
    end
    
    # Create summary analyses
    if length(all_results) > 1
        create_summary_comparison(all_results)
        summary_table = create_correlation_summary_table(all_results)
        
        println("="^80)
        println("ANALYSIS COMPLETE")
        println("="^80)
        println("Processed $(length(all_results)) regions successfully")
        println("Output directory: $output_dir")
        println("- Individual correlation matrices: [region]_correlation_matrix.png")
        println("- Regional comparison plot: regional_correlation_comparison.png") 
        println("- Summary table: regional_correlations_summary.csv")
        
    else
        println("Insufficient successful regions for summary analysis")
    end
    
    return all_results
end

# Run the analysis
if true
    results = run_correlation_analysis()
end