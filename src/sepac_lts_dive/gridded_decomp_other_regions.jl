"""
Perform gridded decomposition analysis for different stratocumulus regions.
This script generalizes the decomposition analysis from gridded_effects.jl to work with multiple regions.
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")

using CSV, DataFrames, Dates, Dictionaries, PythonCall
@py import matplotlib.pyplot as plt, cartopy.crs as ccrs, matplotlib.colors as colors, cmasher as cmr

function skipmissing_corr(x,y)
    valid_data = .!ismissing.(x) .& .!ismissing.(y)
    if sum(valid_data) > 0
        return cor(x[valid_data], y[valid_data])
    else
        return NaN
    end
end

function perform_regional_decomposition(region::String; analysis_bounds = (Date(2000, 3), Date(2024, 3, 31)))
    """
    Perform the gridded decomposition analysis for a specific stratocumulus region.
    
    Parameters:
    - region: String name of the region (e.g., "NEPac", "SEAtl", "SEPac_feedback_definition")
    - analysis_bounds: Tuple of start and end dates for the analysis
    
    Returns:
    - Dictionary containing the analysis results and saves plots to disk
    """
    
    println("="^60)
    println("Processing region: $region")
    println("="^60)
    
    # Create output directory for this region
    region_vis_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/gridded_effects/regional_decomposition/$region"
    mkpath(region_vis_dir)
    
    # Load local time series data
    local_ts_dir = "../../data/sepac_lts_data/local_region_time_series"
    era5_local_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
    ceres_local_df = CSV.read(joinpath(local_ts_dir, "ceres_region_avg_lagged_$(region).csv"), DataFrame)
    
    # Load global CERES data
    ceres_global_df = CSV.read("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/lagged/global_ceres_lagged_detrended_deseasonalized.csv", DataFrame)
    
    # Load nonlocal radiation time series
    nonlocal_ts_dir = "../../data/sepac_lts_data/nonlocal_radiation_time_series"
    nonlocal_rad_df = CSV.read(joinpath(nonlocal_ts_dir, "$(region)_nonlocal_radiation.csv"), DataFrame)
    
    # Load gridded CERES data
    cre_names = []
    toa_rad_names = "toa_" .* ["net_all", "net_lw", "net_sw"] .* "_mon"
    ceres_varnames = vcat(cre_names, toa_rad_names)
    
    ceres_data, ceres_coords = load_new_ceres_data(ceres_varnames, analysis_bounds)
    ceres_lat = ceres_coords["latitude"]
    ceres_lon = ceres_coords["longitude"]
    ceres_time = round.(ceres_coords["time"], Month(1), RoundDown)
    ceres_time_valid = in_time_period.(ceres_time, Ref(analysis_bounds))
    ceres_time = ceres_time[ceres_time_valid]
    ceres_float_time = calc_float_time.(ceres_time)
    ceres_precalculated_month_groups = groupfind(month, ceres_time)
    
    # Detrend and deseasonalize gridded data
    for var in ceres_varnames
        ceres_data[var] = ceres_data[var][:, :, ceres_time_valid]
        detrend_and_deseasonalize_precalculated_groups!.(eachslice(ceres_data[var]; dims = (1,2)), Ref(ceres_float_time), Ref(ceres_precalculated_month_groups))
    end
    
    # Process time series data
    era5_local_df[!, :date] = Date.(era5_local_df[!, :date])
    ceres_local_df[!, :date] = Date.(ceres_local_df[!, :date])
    nonlocal_rad_df[!, :date] = Date.(nonlocal_rad_df[!, :date])
    ceres_global_df[!, :date] = Date.(ceres_global_df[!, :date])
    
    # Filter to analysis bounds
    era5_local_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], era5_local_df)
    ceres_local_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], ceres_local_df)
    nonlocal_rad_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], nonlocal_rad_df)
    ceres_global_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], ceres_global_df)
    
    # Join all dataframes
    local_df = DataFrames.innerjoin(era5_local_df, ceres_local_df, nonlocal_rad_df, ceres_global_df, on = :date)
    
    # Extract key variables
    net_rad = ceres_global_df[!, "gtoa_net_all_mon_lag_0"]
    lts_1000 = local_df[!, "LTS_1000_lag_0"]
    
    # Calculate total correlation
    total_corr = skipmissing_corr(lts_1000, net_rad)
    println("Total correlation between $region LTS and global net radiation: $total_corr")
    
    # Calculate area-weighted gridded data
    area_weighted_net_rad = ceres_data["toa_net_all_mon"] .* cosd.(ceres_lat')
    
    # Calculate gridded correlations and weights
    gridded_net_rad_corr = calculate_corrfunc_grid(area_weighted_net_rad, lts_1000; corrfunc = skipmissing_corr)
    
    net_rad_from_sum = vec(sum(area_weighted_net_rad; dims = (1,2)))
    gridded_net_rad_std = mapslices(std, area_weighted_net_rad; dims = 3)[:,:, 1]
    net_corr_sum_weights = gridded_net_rad_std ./ std(net_rad_from_sum)
    
    weighted_corr = gridded_net_rad_corr .* net_corr_sum_weights
    
    # Verify decomposition
    if !isapprox(sum(skipmissing(vec(weighted_corr))), total_corr; rtol=1e-2)
        @warn "Weighted correlation does not match total correlation within tolerance for region $region!"
    end
    
    # Create 3-panel decomposition plot
    fig, axs = plt.subplots(1, 3, 
                           figsize=(18, 4),
                           subplot_kw=Dict("projection" => ccrs.Sinusoidal(central_longitude=-160)),
                           layout="compressed")

    plot_data = [gridded_net_rad_corr, net_corr_sum_weights ./ cosd.(ceres_lat'), weighted_corr ./ cosd.(ceres_lat')]
    plot_titles = ["Raw Corr", "Weights", "Weighted Corr"]
    
    for (i, (data, title)) in enumerate(zip(plot_data, plot_titles))
        ax = axs[i-1] # Adjust for 0-based indexing in Python
        ax.set_global()
        ax.coastlines()
        ax.set_title(title)
        
        # Calculate color normalization for each plot individually
        absmax = max(abs(minimum(data)), abs(maximum(data)))
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
        
        # Plot the data
        c = ax.contourf(ceres_lon, ceres_lat, data', 
                      transform=ccrs.PlateCarree(), 
                      cmap=cmr.prinsenvlag.reversed(), 
                      levels=21, 
                      norm=colornorm)
        
        # Add individual colorbar for this subplot
        plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, label=title)
    end
    
    fig.suptitle("$region LTS - Global Net Radiation Correlation Decomposition: Total Corr = $(round(total_corr, digits=3))")
    
    # Save the figure
    fig.savefig(joinpath(region_vis_dir, "$(region)_lts_global_net_rad_decomposition.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Component decomposition analysis
    gridded_sw = ceres_data["toa_net_sw_mon"] .* cosd.(ceres_lat')
    gridded_lw = ceres_data["toa_net_lw_mon"] .* cosd.(ceres_lat')
    
    gridded_sw_std = mapslices(std, gridded_sw; dims = 3)
    gridded_lw_std = mapslices(std, gridded_lw; dims = 3)
    
    local_neg_theta_1000 = -1 .* local_df[!, "θ_1000_lag_0"]
    local_theta_700 = local_df[!, "θ_700_lag_0"]
    
    neg_theta_1000_std = std(local_neg_theta_1000)
    theta_700_std = std(local_theta_700)
    LTS_std = std(lts_1000)
    
    # Calculate correlation maps and weights
    sw_theta_1000_corr = calculate_corrfunc_grid(gridded_sw, local_neg_theta_1000; corrfunc = skipmissing_corr)
    sw_theta_1000_weights = @. gridded_sw_std * neg_theta_1000_std / (LTS_std * gridded_net_rad_std)
    
    sw_theta_700_corr = calculate_corrfunc_grid(gridded_sw, local_theta_700; corrfunc = skipmissing_corr)
    sw_theta_700_weights = @. gridded_sw_std * theta_700_std / (LTS_std * gridded_net_rad_std)
    
    lw_theta_1000_corr = calculate_corrfunc_grid(gridded_lw, local_neg_theta_1000; corrfunc = skipmissing_corr)
    lw_theta_1000_weights = @. gridded_lw_std * neg_theta_1000_std / (LTS_std * gridded_net_rad_std)
    
    lw_theta_700_corr = calculate_corrfunc_grid(gridded_lw, local_theta_700; corrfunc = skipmissing_corr)
    lw_theta_700_weights = @. gridded_lw_std * theta_700_std / (LTS_std * gridded_net_rad_std)
    
    # Create 5-panel component decomposition plot
    decomp_corrs = [gridded_net_rad_corr, sw_theta_1000_corr .* sw_theta_1000_weights, sw_theta_700_corr .* sw_theta_700_weights, 
                   lw_theta_1000_corr .* lw_theta_1000_weights, lw_theta_700_corr .* lw_theta_700_weights]
    
    # Handle dimension issues
    decomp_corrs = [if ndims(comp) == 2 comp else dropdims(comp; dims=3) end for comp in decomp_corrs]

    decomp_corrs = [comp ./ cosd.(ceres_lat') .* net_corr_sum_weights for comp in decomp_corrs]
    
    decomp_subtitles = ["Total Local Net Rad Corr", "SW × θ₁₀₀₀ Component", "SW × θ₇₀₀ Component", 
                       "LW × θ₁₀₀₀ Component", "LW × θ₇₀₀ Component"]
    
    # Use plot_multiple_levels to create a single shared colorbar
    fig = plot_multiple_levels(ceres_lat, ceres_lon, decomp_corrs, (1, 5); 
                              subtitles=decomp_subtitles, 
                              colorbar_label="Weighted Correlation Components",
                              proj = ccrs.Sinusoidal(central_longitude=-160))
    
    fig.suptitle("$region LTS - Global Net Radiation Decomposition into SW/LW and θ Components")
    
    # Save the decomposition figure
    fig.savefig(joinpath(region_vis_dir, "$(region)_lts_radiation_component_decomposition.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Return results dictionary
    results = Dict(
        "region" => region,
        "total_correlation" => total_corr,
        "gridded_net_rad_corr" => gridded_net_rad_corr,
        "net_corr_sum_weights" => net_corr_sum_weights,
        "weighted_corr" => weighted_corr,
        "decomp_components" => decomp_corrs,
        "output_dir" => region_vis_dir
    )
    
    println("Completed analysis for region: $region")
    println("Output saved to: $region_vis_dir")
    println()
    
    return results
end

# Main execution: Run analysis for all available regions
function run_all_regional_analyses()
    """
    Run the decomposition analysis for all available stratocumulus regions.
    """
    
    # Define the available regions based on the file listing
    regions = ["NEPac", "SEAtl", "SEPac", "SEPac_feedback_definition", "SEPac_feedback_only"]
    
    # Storage for all results
    all_results = Dict{String, Any}()
    
    # Analysis bounds
    analysis_bounds = (Date(2000, 3), Date(2024, 3, 31))
    
    println("Starting regional decomposition analysis for $(length(regions)) regions...")
    println("Analysis period: $(analysis_bounds[1]) to $(analysis_bounds[2])")
    println()
    
    # Run analysis for each region
    for region in regions
        try
            results = perform_regional_decomposition(region; analysis_bounds = analysis_bounds)
            all_results[region] = results
        catch e
            println("ERROR: Failed to process region $region")
            println("Error: $e")
            println()
            continue
        end
    end
    
    # Create summary comparison plot
    create_regional_comparison_summary(all_results)
    
    println("="^60)
    println("SUMMARY OF REGIONAL ANALYSIS")
    println("="^60)
    
    for (region, results) in all_results
        if haskey(results, "total_correlation")
            println("$region: Total correlation = $(round(results["total_correlation"], digits=4))")
        else
            println("$region: Analysis failed")
        end
    end
    
    println("="^60)
    println("All regional analyses completed!")
    
    return all_results
end

function create_regional_comparison_summary(all_results::Dict)
    """
    Create a summary plot comparing total correlations across all regions.
    """
    
    # Extract successful results
    successful_regions = String[]
    correlations = Float64[]
    
    for (region, results) in all_results
        if haskey(results, "total_correlation") && !isnan(results["total_correlation"])
            push!(successful_regions, region)
            push!(correlations, results["total_correlation"])
        end
    end
    
    if length(successful_regions) == 0
        println("No successful regional analyses to summarize")
        return
    end
    
    # Create summary directory
    summary_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/gridded_effects/regional_decomposition"
    mkpath(summary_dir)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(successful_regions, correlations, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_ylabel("Total Correlation")
    ax.set_title("Regional LTS - Global Net Radiation Correlations")
    ax.grid(true, alpha=0.3)
    
    # Add value labels on bars
    for (bar, corr) in zip(bars, correlations)
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               "$(round(corr, digits=3))", ha="center", va="bottom")
    end
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save summary plot
    summary_file = joinpath(summary_dir, "regional_correlation_summary.png")
    fig.savefig(summary_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    println("Regional comparison summary saved to: $summary_file")
end

# Run the analysis
all_results = run_all_regional_analyses()
