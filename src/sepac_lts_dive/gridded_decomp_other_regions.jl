"""
Perform gridded decomposition analysis for different stratocumulus regions.
This script generalizes the decomposition analysis from gridded_effects.jl to work with multiple regions.
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")

using CSV, DataFrames, Dates, Dictionaries, PythonCall, JLD2, Statistics
@py import matplotlib.pyplot as plt, cartopy.crs as ccrs, matplotlib.colors as colors, cmasher as cmr, matplotlib as mpl

function skipmissing_corr(x,y)
    valid_data = .!ismissing.(x) .& .!ismissing.(y)
    if sum(valid_data) > 0
        return cor(x[valid_data], y[valid_data])
    else
        return NaN
    end
end

regions = ["NEPac", "SEAtl", "SEPac_feedback_definition", "SEPac", "SEPac_feedback_only"]

region_extents_km = [2000, 2000, 3000, 3000, 3000]

region_extend_dict = Dict(zip(regions, region_extents_km))

function perform_regional_decomposition(region::String; analysis_bounds = (Date(2002, 3), Date(2023, 2, 28)))
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
    region_vis_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/regional_gridded_decomp/regional_decomposition/$region"
    mkpath(region_vis_dir)
    
    # Load local time series data
    local_ts_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/local_region_time_series"
    era5_local_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_enso_removed_$(region).csv"), DataFrame)
    ceres_local_df = CSV.read(joinpath(local_ts_dir, "ceres_region_avg_lagged_$(region).csv"), DataFrame)

    era5_lsm, _ = load_era5_data(["lsm"], analysis_bounds)
    era5_lsm = era5_lsm["lsm"][:,:, 1] .== 0

    #Now convert from era5 grid to ceres grid for masking
    ceres_coords_maps = jldopen("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/era5_ceres_coordinate_mapping.jld2")
    era5_to_ceres = CartesianIndex.(ceres_coords_maps["era5_to_ceres_indices"])
    lsm_ceres_grid = getindex.(Ref(era5_lsm), era5_to_ceres)
    
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
        deseasonalize_and_detrend_precalculated_groups_twice!.(eachslice(ceres_data[var]; dims = (1,2)), Ref(ceres_float_time), Ref(ceres_precalculated_month_groups))
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
    net_rad = local_df[!, "gtoa_net_all_mon_lag_0"]
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

    total_rad_check = vec(sum(area_weighted_net_rad; dims=(1,2)))
    rad_check_corr = skipmissing_corr(lts_1000, total_rad_check)
    println("  Verification correlation (from summed gridded data): $rad_check_corr")
    
    # Verify decomposition
    if !isapprox(sum(skipmissing(vec(weighted_corr))), total_corr; rtol=1e-2)
        @warn "Weighted correlation does not match total correlation within tolerance for region $(region)!"
    end

    region_mask_dict = jldopen("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2")
    region_mask = region_mask_dict["regional_masks_ceres"][region]
    
    # Create 3-panel decomposition plot
    fig, axs = plt.subplots(1, 3, 
                           figsize=(18, 4),
                           subplot_kw=Dict("projection" => ccrs.EckertIV(central_longitude=-160)),
                           layout="compressed")

    plot_data = [gridded_net_rad_corr, net_corr_sum_weights ./ cosd.(ceres_lat'), weighted_corr ./ cosd.(ceres_lat')]
    plot_titles = ["Raw Corr", "Weights", "Weighted Corr"]

    #Check the total correlation from the weighted correlation map
    weighted_corr_total = sum(skipmissing(vec(weighted_corr)))
    println("  Total correlation from weighted correlation map: $weighted_corr_total")
    
    for (i, (data, title)) in enumerate(zip(plot_data, plot_titles))
        ax = axs[i-1] # Adjust for 0-based indexing in Python
        ax.set_global()
        ax.coastlines()
        ax.set_title(title)
        
        # Calculate color normalization for each plot individually
        absmax = max(abs(minimum(data)), abs(maximum(data)))
        colornorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
        
        # Plot the data
        c = ax.contourf(ceres_lon, ceres_lat, data', 
                      transform=ccrs.PlateCarree(), 
                      cmap=cmr.prinsenvlag.reversed(), 
                      levels=21, 
                      norm=colornorm)
        
        # Plot the mask contour overlay
        mask_contour = ax.contour(ceres_lon, ceres_lat, Float64.(region_mask)', 
                                transform=ccrs.PlateCarree(), 
                                levels=[0.5], 
                                colors="black", 
                                linewidths=2)
        
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
                              proj = ccrs.EckertIV(central_longitude=-160))
    
    # Add mask contour to all subplots
    for (i, ax) in enumerate(fig.axes)  # Only the first 5 axes contain our plots
        i > 5 && break
        ax.contour(ceres_lon, ceres_lat, Float64.(region_mask)', 
                  transform=ccrs.PlateCarree(), 
                  levels=[0.5], 
                  colors="black", 
                  linewidths=2)
    end
    
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

    cos_weighted_corr = weighted_corr
    results["cos_weighted_corr"] = cos_weighted_corr
    
    #Plotting spatial structure of weighted correlation
    
    #To do this sum the weighted correlation as a function of distance from the region center
    #First find the region center in the most naive way possible: the median lat/lon of the masked region
    region_mask_indices = findall(region_mask)
    # More efficient coordinate extraction using broadcasting
    region_coords = [(ceres_lon[i], ceres_lat[j]) for (i, j) in Tuple.(region_mask_indices)]
    region_lons = first.(region_coords)
    region_lats = last.(region_coords)
    region_center_lon = median(region_lons)
    region_center_lat = median(region_lats)

    # Optimized haversine distance calculation
    function haversine(lon1, lat1, lon2, lat2)
        R = 6371.0 # Earth radius in km
        dlon = deg2rad(lon2 - lon1)
        dlat = deg2rad(lat2 - lat1)
        a = sin(dlat/2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)^2
        c = 2 * atan(sqrt(a), sqrt(1 - a))
        return R * c
    end

    # Calculate distances more efficiently using broadcasting
    distances_shapeful = haversine.(ceres_lon, ceres_lat', region_center_lon, region_center_lat)
    distances = vec(distances_shapeful)

    # Bin distances and sum weighted correlations in each bin
    # Define distance bins using Julia's range function
    bin_width = 100.0  # km
    max_distance = ceil(maximum(distances) / bin_width) * bin_width
    bin_edges = 0:bin_width:max_distance
    bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
    
    # Use digitize for efficient binning (similar to numpy's digitize)
    cos_weighted_corr_vec = vec(cos_weighted_corr)
    bin_indices = searchsortedfirst.(Ref(bin_edges[2:end]), distances)
    # Clamp bin indices to valid range
    bin_indices = clamp.(bin_indices, 1, length(bin_centers))
    
    # Initialize binned results
    n_bins = length(bin_centers)
    binned_results = Dict(
        "total_weighted_corr" => zeros(n_bins),
        "abs_weighted_corr" => zeros(n_bins),
        "positive_weighted_corr" => zeros(n_bins),
        "negative_weighted_corr" => zeros(n_bins)
    )

    # Single pass efficient binning using accumulate pattern
    for (bin_idx, corr_val) in zip(bin_indices, cos_weighted_corr_vec)
        if !ismissing(corr_val) && 1 <= bin_idx <= n_bins
            binned_results["total_weighted_corr"][bin_idx] += corr_val
            binned_results["abs_weighted_corr"][bin_idx] += abs(corr_val)
            if corr_val > 0
                binned_results["positive_weighted_corr"][bin_idx] += corr_val
            elseif corr_val < 0
                binned_results["negative_weighted_corr"][bin_idx] += corr_val
            end
        end
    end

    results["binned_results"] = Dict("bin_centers" => collect(bin_centers), "data" => binned_results)

    # Plot cumulative sum and raw total weighted correlation vs distance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=true)

    # Plot raw total weighted correlation
    ax1.plot(bin_centers, binned_results["total_weighted_corr"], "o-", linewidth=2, markersize=4, color="steelblue")
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Weighted Correlation")
    ax1.set_title("$region: Weighted Correlation vs Distance from Region Center")
    ax1.grid(true, alpha=0.3)

    # Plot cumulative sum of total weighted correlation  
    cumsum_weighted_corr = cumsum(binned_results["total_weighted_corr"])
    ax2.plot(bin_centers, cumsum_weighted_corr, "o-", linewidth=2, markersize=4, color="darkred")
    ax2.axhline(y=total_corr, color="black", linestyle="--", alpha=0.7, label="Total Correlation")
    ax2.set_ylabel("Cumulative Weighted Correlation")
    ax2.set_xlabel("Distance from Region Center (km)")
    ax2.set_title("Cumulative Sum of Weighted Correlation")
    ax2.grid(true, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save the distance analysis plot
    fig.savefig(joinpath(region_vis_dir, "$(region)_weighted_correlation_vs_distance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot positive and negative components on a single plot (negative values multiplied by -1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot positive components
    ax.plot(bin_centers, binned_results["positive_weighted_corr"], "o-", linewidth=2, markersize=4, color="red", label="Positive")
    
    # Plot negative components with sign flipped (multiply by -1)
    negative_flipped = -1 .* binned_results["negative_weighted_corr"]
    ax.plot(bin_centers, negative_flipped, "o-", linewidth=2, markersize=4, color="blue", label="Negative (×-1)")
    
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Weighted Correlation Magnitude")
    ax.set_xlabel("Distance from Region Center (km)")
    ax.set_title("$region: Positive vs Negative Weighted Correlation Components vs Distance")
    ax.grid(true, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save the combined positive/negative plot
    fig.savefig(joinpath(region_vis_dir, "$(region)_combined_positive_negative_vs_distance.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create distance contour plot on Robinson projection centered on region longitude
    fig, ax = plt.subplots(1, 1, figsize=(12, 8),
                          subplot_kw=Dict("projection" => ccrs.EckertIV(central_longitude=region_center_lon)))
    
    ax.set_global()
    ax.coastlines()
    ax.set_title("$region: Distance Contours from Region Center (every 5000 km)")
    
    # Plot the weighted correlation as background
    unweighted_mat = cos_weighted_corr ./ cosd.(ceres_lat')
    absmax = max(abs(minimum(unweighted_mat)), abs(maximum(unweighted_mat)))
    colornorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
    
    c = ax.contourf(ceres_lon, ceres_lat, unweighted_mat', 
                   transform=ccrs.PlateCarree(), 
                   cmap=cmr.prinsenvlag.reversed(), 
                   levels=21, 
                   norm=colornorm,
                   alpha=0.8)
    
    # Add colorbar for weighted correlation
    cbar = plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, 
                       label="Weighted Correlation (sums to correlation with global rad)", shrink=0.8)
    
    # Plot distance contours every 5000 km
    distance_levels = collect(2500:2500:20000)  # Contours every 5000 km up to 30000 km
    contours = ax.contour(ceres_lon, ceres_lat, distances_shapeful', 
                         transform=ccrs.PlateCarree(),
                         levels=distance_levels,
                         colors="green",
                         linewidths=1.5,
                         alpha=0.9)
    
    # Add contour labels
    ax.clabel(contours, inline=true, fontsize=10, fmt="%d km")
    
    # Plot the region mask contour
    mask_contour = ax.contour(ceres_lon, ceres_lat, Float64.(region_mask)', 
                            transform=ccrs.PlateCarree(), 
                            levels=[0.5], 
                            colors="black", 
                            linewidths=3)
    
    # Mark the region center
    ax.plot(region_center_lon, region_center_lat, "r*", 
           transform=ccrs.PlateCarree(), 
           markersize=15, 
           markeredgecolor="white", 
           markeredgewidth=1,
           label="Region Center")
    
    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
    
    # Save the distance contour plot
    fig.savefig(joinpath(region_vis_dir, "$(region)_distance_contours_robinson.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Distance binning analysis for decomposition components
    # Apply cosine weighting to each component for consistent comparison
    sw_theta_1000_weighted = (sw_theta_1000_corr .* sw_theta_1000_weights ./ cosd.(ceres_lat') .* net_corr_sum_weights) .* cosd.(ceres_lat')
    sw_theta_700_weighted = (sw_theta_700_corr .* sw_theta_700_weights ./ cosd.(ceres_lat') .* net_corr_sum_weights) .* cosd.(ceres_lat')
    lw_theta_1000_weighted = (lw_theta_1000_corr .* lw_theta_1000_weights ./ cosd.(ceres_lat') .* net_corr_sum_weights) .* cosd.(ceres_lat')
    lw_theta_700_weighted = (lw_theta_700_corr .* lw_theta_700_weights ./ cosd.(ceres_lat') .* net_corr_sum_weights) .* cosd.(ceres_lat')

    component_names = ["SW×θ₁₀₀₀", "SW×θ₇₀₀", "LW×θ₁₀₀₀", "LW×θ₇₀₀"]
    component_data = [sw_theta_1000_weighted, sw_theta_700_weighted, lw_theta_1000_weighted, lw_theta_700_weighted]
    component_results = Dict()

    for (comp_name, comp_data) in zip(component_names, component_data)
        # Flatten the component data
        comp_data_vec = vec(comp_data)
        
        # Use same distance binning as before
        comp_binned = Dict(
            "total_weighted_corr" => zeros(length(bin_centers)),
            "abs_weighted_corr" => zeros(length(bin_centers)),
            "positive_weighted_corr" => zeros(length(bin_centers)),
            "negative_weighted_corr" => zeros(length(bin_centers))
        )

        # Single pass efficient binning
        for (bin_idx, corr_val) in zip(bin_indices, comp_data_vec)
            if !ismissing(corr_val) && 1 <= bin_idx <= length(bin_centers)
                comp_binned["total_weighted_corr"][bin_idx] += corr_val
                comp_binned["abs_weighted_corr"][bin_idx] += abs(corr_val)
                if corr_val > 0
                    comp_binned["positive_weighted_corr"][bin_idx] += corr_val
                elseif corr_val < 0
                    comp_binned["negative_weighted_corr"][bin_idx] += corr_val
                end
            end
        end

        comp_cumsum = cumsum(comp_binned["total_weighted_corr"])
        
        # Plot component distance analysis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=true)

        # Plot raw total weighted correlation
        ax1.plot(bin_centers, comp_binned["total_weighted_corr"], "o-", linewidth=2, markersize=4, color="steelblue")
        ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Weighted Correlation")
        ax1.set_title("$region $comp_name: Weighted Correlation vs Distance from Region Center")
        ax1.grid(true, alpha=0.3)

        # Plot cumulative sum
        ax2.plot(bin_centers, comp_cumsum, "o-", linewidth=2, markersize=4, color="darkred")
        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.7)
        ax2.set_ylabel("Cumulative Weighted Correlation")
        ax2.set_xlabel("Distance from Region Center (km)")
        ax2.set_title("Cumulative Sum of $comp_name Component")
        ax2.grid(true, alpha=0.3)

        plt.tight_layout()

        # Save the component distance analysis plot
        comp_safe_name = replace(comp_name, "×" => "x")  # Replace × with x for filename safety
        fig.savefig(joinpath(region_vis_dir, "$(region)_$(comp_safe_name)_distance_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Store component results
        component_results[comp_name] = Dict(
            "binned_results" => comp_binned,
            "cumsum" => comp_cumsum
        )
    end

    # Create combined plot with all components plus total
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=true)

    # Define colors for each component
    colors = ["red", "orange", "blue", "purple", "black"]
    labels = [component_names..., "Total"]
    
    # Plot raw weighted correlations
    for (i, (comp_name, comp_data)) in enumerate(zip(component_names, component_data))
        comp_binned = component_results[comp_name]["binned_results"]
        ax1.plot(bin_centers, comp_binned["total_weighted_corr"], "o-", 
                linewidth=2, markersize=3, color=colors[i], label=comp_name, alpha=0.8)
    end
    # Add total correlation
    ax1.plot(bin_centers, binned_results["total_weighted_corr"], "o-", 
            linewidth=3, markersize=4, color=colors[end], label="Total")
    
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Weighted Correlation")
    ax1.set_title("$region: All Components vs Distance from Region Center")
    ax1.grid(true, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot cumulative sums
    for (i, (comp_name, comp_data)) in enumerate(zip(component_names, component_data))
        comp_cumsum = component_results[comp_name]["cumsum"]
        ax2.plot(bin_centers, comp_cumsum, "o-", 
                linewidth=2, markersize=3, color=colors[i], label=comp_name, alpha=0.8)
    end
    # Add total cumsum
    ax2.plot(bin_centers, cumsum_weighted_corr, "o-", 
            linewidth=3, markersize=4, color=colors[end], label="Total")
    ax2.axhline(y=total_corr, color="gray", linestyle="--", alpha=0.7, label="Expected Total")
    
    ax2.set_ylabel("Cumulative Weighted Correlation")
    ax2.set_xlabel("Distance from Region Center (km)")
    ax2.set_title("Cumulative Sums of All Components")
    ax2.grid(true, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save the combined components plot
    fig.savefig(joinpath(region_vis_dir, "$(region)_all_components_distance_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    results["distance_analysis"] = Dict("bin_centers" => bin_centers, "binned_results" => binned_results, "cumsum_weighted_corr" => cumsum_weighted_corr, "region_center" => (region_center_lon, region_center_lat), "distance_levels" => distance_levels)
    results["component_distance_analysis"] = component_results

    # Latitude binning analysis
    lat_bins = -90:5:90
    lat_centers = (lat_bins[1:end-1] .+ lat_bins[2:end]) ./ 2
    
    # Function to bin data by latitude
    function bin_by_latitude(data_matrix, lat_grid, lat_bins)
        binned_data = zeros(length(lat_bins) - 1)
        for (i, lat_val) in enumerate(lat_grid)
            bin_idx = searchsortedfirst(lat_bins[2:end], lat_val)
            bin_idx = clamp(bin_idx, 1, length(lat_bins) - 1)
            
            # Sum across all longitudes for this latitude
            for j in 1:size(data_matrix, 1)  # longitude dimension
                if !ismissing(data_matrix[j, i])
                    binned_data[bin_idx] += data_matrix[j, i]
                end
            end
        end
        return binned_data
    end

    lonlat_arr = tuple.(ceres_lon, ceres_lat')
    mat_lats = last.(lonlat_arr)
    region_lats = mat_lats[region_mask .== true]
    region_lat_extrema = extrema(region_lats)



    
    # Bin the total weighted correlation and components by latitude
    lat_binned_total = bin_by_latitude(cos_weighted_corr, ceres_lat, lat_bins)
    lat_binned_sw_theta_1000 = bin_by_latitude(sw_theta_1000_weighted, ceres_lat, lat_bins)
    lat_binned_sw_theta_700 = bin_by_latitude(sw_theta_700_weighted, ceres_lat, lat_bins)
    lat_binned_lw_theta_1000 = bin_by_latitude(lw_theta_1000_weighted, ceres_lat, lat_bins)
    lat_binned_lw_theta_700 = bin_by_latitude(lw_theta_700_weighted, ceres_lat, lat_bins)
    
    # Create latitude binning plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    
    # Plot components
    ax.plot(lat_binned_sw_theta_1000, lat_centers, "o-", label="SW×θ₁₀₀₀", linewidth=2, markersize=3, color="red")
    ax.plot(lat_binned_sw_theta_700, lat_centers, "o-", label="SW×θ₇₀₀", linewidth=2, markersize=3, color="orange")
    ax.plot(lat_binned_lw_theta_1000, lat_centers, "o-", label="LW×θ₁₀₀₀", linewidth=2, markersize=3, color="blue")
    ax.plot(lat_binned_lw_theta_700, lat_centers, "o-", label="LW×θ₇₀₀", linewidth=2, markersize=3, color="purple")
    ax.plot(lat_binned_total, lat_centers, "o-", label="Total", linewidth=3, markersize=4, color="black")

    #Now plot horizontal lines indicating the region latitude extent
    ax.axhline(y=region_lat_extrema[1], color="gray", linestyle=":", alpha=0.7, label="Region Lat Extrema")
    ax.axhline(y=region_lat_extrema[2], color="gray", linestyle=":", alpha=0.7)
    
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Summed Weighted Correlation")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("$region: Latitude-Binned Weighted Correlation Components")
    ax.grid(true, alpha=0.3)
    ax.legend()
    ax.set_ylim(-90, 90)
    
    plt.tight_layout()
    
    # Save the latitude binning plot
    fig.savefig(joinpath(region_vis_dir, "$(region)_latitude_binned_weighted_correlation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Store latitude binning results
    results["latitude_binning"] = Dict(
        "lat_centers" => collect(lat_centers),
        "total" => lat_binned_total,
        "sw_theta_1000" => lat_binned_sw_theta_1000,
        "sw_theta_700" => lat_binned_sw_theta_700,
        "lw_theta_1000" => lat_binned_lw_theta_1000,
        "lw_theta_700" => lat_binned_lw_theta_700
    )

    # Longitude binning analysis
    lon_bins = 0:5:360
    lon_centers = (lon_bins[1:end-1] .+ lon_bins[2:end]) ./ 2
    
    # Function to bin data by longitude
    function bin_by_longitude(data_matrix, lon_grid, lon_bins)
        binned_data = zeros(length(lon_bins) - 1)
        for (i, lon_val) in enumerate(lon_grid)
            bin_idx = searchsortedfirst(lon_bins[2:end], lon_val)
            bin_idx = clamp(bin_idx, 1, length(lon_bins) - 1)
            
            # Sum across all latitudes for this longitude
            for j in 1:size(data_matrix, 2)  # latitude dimension
                if !ismissing(data_matrix[i, j])
                    binned_data[bin_idx] += data_matrix[i, j]
                end
            end
        end
        return binned_data
    end

    lonlat_arr = tuple.(ceres_lon, ceres_lat')
    mat_lons = first.(lonlat_arr)
    region_lons = mat_lons[region_mask .== true]
    region_lon_extrema = extrema(region_lons)
    
    # Bin the total weighted correlation and components by longitude
    lon_binned_total = bin_by_longitude(cos_weighted_corr, ceres_lon, lon_bins)
    lon_binned_sw_theta_1000 = bin_by_longitude(sw_theta_1000_weighted, ceres_lon, lon_bins)
    lon_binned_sw_theta_700 = bin_by_longitude(sw_theta_700_weighted, ceres_lon, lon_bins)
    lon_binned_lw_theta_1000 = bin_by_longitude(lw_theta_1000_weighted, ceres_lon, lon_bins)
    lon_binned_lw_theta_700 = bin_by_longitude(lw_theta_700_weighted, ceres_lon, lon_bins)
    
    # Create longitude binning plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot components
    ax.plot(lon_centers, lon_binned_sw_theta_1000, "o-", label="SW×θ₁₀₀₀", linewidth=2, markersize=3, color="red")
    ax.plot(lon_centers, lon_binned_sw_theta_700, "o-", label="SW×θ₇₀₀", linewidth=2, markersize=3, color="orange")
    ax.plot(lon_centers, lon_binned_lw_theta_1000, "o-", label="LW×θ₁₀₀₀", linewidth=2, markersize=3, color="blue")
    ax.plot(lon_centers, lon_binned_lw_theta_700, "o-", label="LW×θ₇₀₀", linewidth=2, markersize=3, color="purple")
    ax.plot(lon_centers, lon_binned_total, "o-", label="Total", linewidth=3, markersize=4, color="black")

    # Now plot vertical lines indicating the region longitude extent
    ax.axvline(x=region_lon_extrema[1], color="gray", linestyle=":", alpha=0.7, label="Region Lon Extrema")
    ax.axvline(x=region_lon_extrema[2], color="gray", linestyle=":", alpha=0.7)
    
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Summed Weighted Correlation")
    ax.set_xlabel("Longitude (°)")
    ax.set_title("$region: Longitude-Binned Weighted Correlation Components")
    ax.grid(true, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the longitude binning plot
    fig.savefig(joinpath(region_vis_dir, "$(region)_longitude_binned_weighted_correlation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Store longitude binning results
    results["longitude_binning"] = Dict(
        "lon_centers" => collect(lon_centers),
        "total" => lon_binned_total,
        "sw_theta_1000" => lon_binned_sw_theta_1000,
        "sw_theta_700" => lon_binned_sw_theta_700,
        "lw_theta_1000" => lon_binned_lw_theta_1000,
        "lw_theta_700" => lon_binned_lw_theta_700
    )

    #Now perform latitude binning, but excluding the stratcomulus regions from the sum
    dist_thresh_km = region_extend_dict[region]

    in_stratocum_dist = distances_shapeful .<= dist_thresh_km
    in_stratocum_corr_thresh = cos_weighted_corr .< 0
    in_stratocum_region = in_stratocum_dist .& in_stratocum_corr_thresh .& lsm_ceres_grid

    #Sum the correlation for the stratocumulus region
    stratocum_region_sum = sum(cos_weighted_corr[in_stratocum_region .== true])

    #Now bin the non stratocumulus region by latitude as before
    # Function to bin data by latitude excluding stratocumulus region
    function bin_by_latitude_excluding_mask(data_matrix, lat_grid, lat_bins, exclusion_mask)
        binned_data = zeros(length(lat_bins) - 1)
        for (i, lat_val) in enumerate(lat_grid)
            bin_idx = searchsortedfirst(lat_bins[2:end], lat_val)
            bin_idx = clamp(bin_idx, 1, length(lat_bins) - 1)
            
            # Sum across all longitudes for this latitude, excluding masked regions
            for j in 1:size(data_matrix, 1)  # longitude dimension
                if !ismissing(data_matrix[j, i]) && !exclusion_mask[j, i]
                    binned_data[bin_idx] += data_matrix[j, i]
                end
            end
        end
        return binned_data
    end

    # Bin the total weighted correlation and components by latitude excluding stratocumulus region
    lat_binned_total_excl = bin_by_latitude_excluding_mask(cos_weighted_corr, ceres_lat, lat_bins, in_stratocum_region)
    lat_binned_sw_theta_1000_excl = bin_by_latitude_excluding_mask(sw_theta_1000_weighted, ceres_lat, lat_bins, in_stratocum_region)
    lat_binned_sw_theta_700_excl = bin_by_latitude_excluding_mask(sw_theta_700_weighted, ceres_lat, lat_bins, in_stratocum_region)
    lat_binned_lw_theta_1000_excl = bin_by_latitude_excluding_mask(lw_theta_1000_weighted, ceres_lat, lat_bins, in_stratocum_region)
    lat_binned_lw_theta_700_excl = bin_by_latitude_excluding_mask(lw_theta_700_weighted, ceres_lat, lat_bins, in_stratocum_region)

    # Create 2-panel plots for each region showing total and components
    # Define component names and colors
    comp_names_plot = ["SW×θ₁₀₀₀", "SW×θ₇₀₀", "LW×θ₁₀₀₀", "LW×θ₇₀₀"]
    comp_colors_plot = ["red", "orange", "blue", "purple"]
    comp_data_excl = [lat_binned_sw_theta_1000_excl, lat_binned_sw_theta_700_excl, 
                      lat_binned_lw_theta_1000_excl, lat_binned_lw_theta_700_excl]

    # Create the 2-panel plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=true)

    # Panel 1: Total correlation by latitude (excluded vs full)
    ax1.plot(lat_binned_total, lat_centers, "o-", label="Full Global", 
            linewidth=2.5, markersize=4, color="black")
    ax1.plot(lat_binned_total_excl, lat_centers, "o-", label="Stratocum Excluded", 
            linewidth=2.5, markersize=4, color="gray", linestyle="--")
    
    # Add horizontal lines indicating region latitude extent
    ax1.axhline(y=region_lat_extrema[1], color="lightgray", linestyle=":", alpha=0.7)
    ax1.axhline(y=region_lat_extrema[2], color="lightgray", linestyle=":", alpha=0.7)
    
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Summed Weighted Correlation", fontsize=12)
    ax1.set_ylabel("Latitude (°)", fontsize=12)
    ax1.set_title("Total Correlation by Latitude", fontsize=13, fontweight="bold")
    ax1.grid(true, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim(-90, 90)
    
    # Add subtitle showing stratocumulus contribution
    ax1.text(0.5, -0.12, "Stratocum Region Contribution: $(round(stratocum_region_sum, digits=2))", 
            transform=ax1.transAxes, ha="center", fontsize=10, style="italic")

    # Panel 2: Four components (stratocum excluded only)
    for (comp_name, comp_color, comp_data) in zip(comp_names_plot, comp_colors_plot, comp_data_excl)
        ax2.plot(comp_data, lat_centers, "o-", label=comp_name, 
                linewidth=2, markersize=3, color=comp_color)
    end
    
    # Add horizontal lines indicating region latitude extent
    ax2.axhline(y=region_lat_extrema[1], color="lightgray", linestyle=":", alpha=0.7)
    ax2.axhline(y=region_lat_extrema[2], color="lightgray", linestyle=":", alpha=0.7)
    
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Summed Weighted Correlation", fontsize=12)
    ax2.set_title("Components (Stratocum Excluded)", fontsize=13, fontweight="bold")
    ax2.grid(true, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Add subtitle showing stratocumulus contribution
    ax2.text(0.5, -0.12, "Stratocum Region Contribution: $(round(stratocum_region_sum, digits=2))", 
            transform=ax2.transAxes, ha="center", fontsize=10, style="italic")

    # Overall title
    fig.suptitle("$region: Latitude-Binned Analysis (Stratocumulus Region Excluded)", 
                fontsize=15, fontweight="bold")
    
    plt.tight_layout()
    
    # Save the stratocumulus-excluded latitude binning plot
    fig.savefig(joinpath(region_vis_dir, "$(region)_latitude_binned_stratocum_excluded.png"), 
               dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Store stratocumulus-excluded latitude binning results
    results["latitude_binning_stratocum_excluded"] = Dict(
        "lat_centers" => collect(lat_centers),
        "total" => lat_binned_total_excl,
        "sw_theta_1000" => lat_binned_sw_theta_1000_excl,
        "sw_theta_700" => lat_binned_sw_theta_700_excl,
        "lw_theta_1000" => lat_binned_lw_theta_1000_excl,
        "lw_theta_700" => lat_binned_lw_theta_700_excl,
        "stratocum_region_sum" => stratocum_region_sum,
        "distance_threshold_km" => dist_thresh_km
    )

    println("Stratocumulus-excluded latitude binning completed. Region contribution: $(round(stratocum_region_sum, digits=4))")

    # Create cartographic plot showing excluded region with weighted correlation background
    fig, ax = plt.subplots(1, 1, figsize=(12, 8),
                          subplot_kw=Dict("projection" => ccrs.EckertIV(central_longitude=region_center_lon)))
    
    ax.set_global()
    ax.coastlines()
    ax.set_title("$region: Excluded Stratocumulus Region (Dist ≤ $(dist_thresh_km) km, Corr < 0)", 
                fontsize=14, fontweight="bold")
    
    # Plot the weighted correlation as background
    unweighted_mat = cos_weighted_corr ./ cosd.(ceres_lat')
    absmax = max(abs(minimum(unweighted_mat)), abs(maximum(unweighted_mat)))
    colornorm = mpl.colors.Normalize(vmin=-absmax, vmax=absmax)
    
    c = ax.contourf(ceres_lon, ceres_lat, unweighted_mat', 
                   transform=ccrs.PlateCarree(), 
                   cmap=cmr.prinsenvlag.reversed(), 
                   levels=21, 
                   norm=colornorm,
                   alpha=0.85)
    
    # Add colorbar for weighted correlation
    cbar = plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, 
                       label="Weighted Correlation (sums to correlation with global rad)", shrink=0.8)
    
    # Plot the excluded stratocumulus region contour (thick, prominent)
    exclusion_contour = ax.contour(ceres_lon, ceres_lat, Float64.(in_stratocum_region)', 
                                  transform=ccrs.PlateCarree(), 
                                  levels=[0.5], 
                                  colors="yellow", 
                                  linewidths=3,
                                  linestyles="solid")
    
    # Add contour label
    ax.clabel(exclusion_contour, inline=true, fontsize=11, fmt="Excluded", manual=false)
    
    # Plot the original region mask contour (for reference)
    mask_contour = ax.contour(ceres_lon, ceres_lat, Float64.(region_mask)', 
                            transform=ccrs.PlateCarree(), 
                            levels=[0.5], 
                            colors="black", 
                            linewidths=2.5,
                            linestyles="--")
    
    # Mark the region center
    ax.plot(region_center_lon, region_center_lat, "r*", 
           transform=ccrs.PlateCarree(), 
           markersize=15, 
           markeredgecolor="white", 
           markeredgewidth=1.5,
           label="Region Center")
    
    # Add text annotation showing contribution
    ax.text(0.02, 0.98, "Excluded Region Contribution: $(round(stratocum_region_sum, digits=2))", 
           transform=ax.transAxes, 
           ha="left", va="top",
           bbox=Dict("boxstyle" => "round,pad=0.5", "facecolor" => "white", "alpha" => 0.8),
           fontsize=11, fontweight="bold")
    
    # Create custom legend
    matplotlib_patches = pyimport("matplotlib.patches")
    matplotlib_lines = pyimport("matplotlib.lines")
    Patch = matplotlib_patches.Patch
    Line2D = matplotlib_lines.Line2D
    
    legend_elements = pylist([
        Line2D([0], [0], color="yellow", linewidth=3, label="Excluded Region"),
        Line2D([0], [0], color="black", linewidth=2.5, linestyle="--", label="Original Region Mask"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="r", 
               markersize=12, markeredgecolor="white", markeredgewidth=1.5, label="Region Center")
    ])
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)
    
    # Save the cartographic excluded region plot
    fig.savefig(joinpath(region_vis_dir, "$(region)_excluded_region_cartographic.png"), 
               dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    println("Cartographic excluded region plot saved.")

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
    analysis_bounds = (Date(2002, 3), Date(2022, 2, 28))
    
    println("Starting regional decomposition analysis for $(length(regions)) regions...")
    println("Analysis period: $(analysis_bounds[1]) to $(analysis_bounds[2])")
    println()
    
    # Run analysis for each region
    for region in regions
        results = perform_regional_decomposition(region; analysis_bounds = analysis_bounds)
        all_results[region] = results        
    end
    
    # Create summary comparison plot
    create_regional_comparison_summary(all_results)
    
    # Create 3x3 stacked decomposition plot
    create_stacked_3x3_decomposition_plot(all_results)
    
    # Create latitude binning comparison plot
    create_latitude_binning_comparison(all_results)
    
    # Create longitude binning comparison plot
    create_longitude_binning_comparison(all_results)
    
    # Create distance cumsum comparison plot
    create_distance_cumsum_comparison(all_results)
    
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

function create_stacked_3x3_decomposition_plot(all_results::Dict)
    """
    Create a 3x3 stacked plot with SEPac, NEPac, and SEAtl regions (rows)
    and Raw Corr, Weights, Weighted Corr (columns) with shared colorbars per column.
    """
    
    # Define the regions to include (in order)
    target_regions = ["SEPac", "NEPac", "SEAtl"]
    
    # Check if all target regions are available
    available_regions = [region for region in target_regions if haskey(all_results, region)]
    if length(available_regions) < 3
        println("Warning: Not all target regions available for 3x3 plot. Available: $available_regions")
        return
    end
    
    # Load masks for contour overlays
    region_mask_dict = jldopen("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2")
    
    # Create output directory
    summary_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/regional_gridded_decomp/regional_decomposition"
    mkpath(summary_dir)
    
    # Get coordinates (load once and reuse)
    lat = get_ceres_lat()
    lon = get_ceres_lon()
    
    # Create 3x3 subplot figure
    fig, axs = plt.subplots(3, 3, 
                           figsize=(18, 12),
                           subplot_kw=Dict("projection" => ccrs.EckertIV(central_longitude=-160)),
                           layout="compressed")
    
    # Column titles
    column_titles = ["Raw Corr", "Weights", "Weighted Corr"]
    
    # Store data for each column to compute shared colorbars
    all_column_data = [[] for _ in 1:3]  # Three columns
    
    # Collect all data first to compute shared color ranges
    for region in available_regions
        results = all_results[region]
        
        # Prepare the three data types for this region
        plot_data = [
            results["gridded_net_rad_corr"],
            results["net_corr_sum_weights"] ./ cosd.(lat'),
            results["weighted_corr"] ./ cosd.(lat')
        ]
        
        for (col_idx, data) in enumerate(plot_data)
            push!(all_column_data[col_idx], data)
        end
    end
    
    # Calculate shared color normalization for each column
    column_norms = []
    for col_data in all_column_data
        all_vals = vcat([vec(data) for data in col_data]...)
        absmax = max(abs(minimum(all_vals)), abs(maximum(all_vals)))
        push!(column_norms, colors.Normalize(vmin=-absmax, vmax=absmax))
    end
    
    # Now plot everything
    cbar_mappables = []  # Store mappables for colorbars
    
    for (row_idx, region) in enumerate(available_regions)
        results = all_results[region]
        region_mask = region_mask_dict["regional_masks_ceres"][region]
        
        # Prepare the three data types for this region
        plot_data = [
            results["gridded_net_rad_corr"],
            results["net_corr_sum_weights"] ./ cosd.(lat'),
            results["weighted_corr"] ./ cosd.(lat')
        ]
        
        for (col_idx, (data, title)) in enumerate(zip(plot_data, column_titles))
            ax = axs[row_idx-1, col_idx-1]
            ax.set_global()
            ax.coastlines()
            
            # Set title only for top row
            if row_idx == 1  # Julia 1-based indexing
                ax.set_title(title, fontsize=14, fontweight="bold")
            end
            
            # Set region label only for first column
            if col_idx == 1  # Julia 1-based indexing
                ax.text(-0.1, 0.5, region, transform=ax.transAxes, 
                       rotation=90, ha="center", va="center", 
                       fontsize=14, fontweight="bold")
            end
            
            # Plot the data using shared color normalization
            c = ax.contourf(lon, lat, data', 
                          transform=ccrs.PlateCarree(), 
                          cmap=cmr.prinsenvlag.reversed(), 
                          levels=21, 
                          norm=column_norms[col_idx])
            
            # Plot the mask contour overlay
            ax.contour(lon, lat, Float64.(region_mask)', 
                      transform=ccrs.PlateCarree(), 
                      levels=[0.5], 
                      colors="black", 
                      linewidths=2)
            
            # Store mappable for bottom row (for colorbars)
            if row_idx == 3  # Bottom row (Julia 1-based: 1, 2, 3)
                if col_idx <= length(cbar_mappables)
                    push!(cbar_mappables, c)
                end
            end
        end
    end
    
    # Add shared colorbars for each column at the bottom
    for (col_idx, (mappable, title)) in enumerate(zip(cbar_mappables, column_titles))
        # Get all axes for this column (convert to Python 0-based indexing)
        col_axes = [axs[i-1, col_idx-1] for i in 1:3]
        cbar = plt.colorbar(mappable, ax=col_axes, orientation="horizontal", 
                           pad=0.05, label=title, shrink=0.8)
    end
    
    # Set overall title
    fig.suptitle("Regional LTS - Global Net Radiation Decomposition Comparison", 
                fontsize=16, fontweight="bold")
    
    # Save the figure
    output_file = joinpath(summary_dir, "stacked_3x3_regional_decomposition.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    
    
    # Close the JLD2 file
    close(region_mask_dict)
    
    println("3x3 stacked decomposition plot saved to: $output_file")
end

# Helper function to get coordinates (we'll need to modify this to access the coordinates)
function get_ceres_lat()
    # Load coordinates from CERES data - this is a simplified version
    # In practice, you might want to pass coordinates or load them here
    ceres_data, ceres_coords = load_new_ceres_data(["toa_net_all_mon"], (Date(2000, 3), Date(2000, 4)))
    return ceres_coords["latitude"]
end

function get_ceres_lon()
    # Load coordinates from CERES data - this is a simplified version
    ceres_data, ceres_coords = load_new_ceres_data(["toa_net_all_mon"], (Date(2000, 3), Date(2000, 4)))
    return ceres_coords["longitude"]
end

function create_latitude_binning_comparison(all_results::Dict)
    """
    Create comparison plots of latitude-binned weighted correlations for all regions.
    """
    
    # Define the regions to include
    target_regions = ["SEPac", "NEPac", "SEAtl"]
    available_regions = [region for region in target_regions if haskey(all_results, region) && haskey(all_results[region], "latitude_binning")]
    
    if length(available_regions) == 0
        println("No regions with latitude binning data available")
        return
    end
    
    # Create output directory
    summary_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/regional_gridded_decomp/regional_decomposition"
    mkpath(summary_dir)
    
    # Get latitude centers (should be the same for all regions)
    lat_centers = all_results[available_regions[1]]["latitude_binning"]["lat_centers"]
    
    # Component names and colors
    component_names = ["sw_theta_1000", "sw_theta_700", "lw_theta_1000", "lw_theta_700"]
    component_labels = ["SW×θ₁₀₀₀", "SW×θ₇₀₀", "LW×θ₁₀₀₀", "LW×θ₇₀₀"]
    component_colors = ["red", "orange", "blue", "purple"]
    
    # Create subplot for each region
    n_regions = length(available_regions)
    fig, axes = plt.subplots(1, n_regions, figsize=(6*n_regions, 8), sharey=true)
    
    if n_regions == 1
        axes = [axes]  # Ensure axes is always iterable
    end
    
    for (i, region) in enumerate(available_regions)
        ax = axes[i-1]
        lat_data = all_results[region]["latitude_binning"]
        
        # Plot components
        for (comp_name, comp_label, color) in zip(component_names, component_labels, component_colors)
            ax.plot(lat_data[comp_name], lat_centers, "o-", 
                   label=comp_label, linewidth=2, markersize=3, color=color)
        end
        
        # Plot total
        ax.plot(lat_data["total"], lat_centers, "o-", 
               label="Total", linewidth=3, markersize=4, color="black")
        
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Summed Weighted Correlation")
        ax.set_title(region)
        ax.grid(true, alpha=0.3)
        ax.set_ylim(-90, 90)
        
        if i == 0  # Only set ylabel for leftmost plot
            ax.set_ylabel("Latitude (°)")
        end
        
        if i == n_regions - 1  # Only show legend on rightmost plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        end
    end
    
    plt.suptitle("Regional Latitude-Binned Weighted Correlation Components", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save the comparison plot
    output_file = joinpath(summary_dir, "regional_latitude_binning_comparison.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Create combined plot with all regions on one axis for total correlations
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    
    region_colors = ["red", "blue", "green", "purple", "orange"]
    
    for (i, region) in enumerate(available_regions)
        lat_data = all_results[region]["latitude_binning"]
        ax.plot(lat_data["total"], lat_centers, "o-", 
               label=region, linewidth=2, markersize=3, 
               color=region_colors[i % length(region_colors)])
    end
    
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Summed Weighted Correlation")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("Total Weighted Correlation by Latitude - All Regions")
    ax.grid(true, alpha=0.3)
    ax.legend()
    ax.set_ylim(-90, 90)
    
    plt.tight_layout()
    
    # Save the combined total plot
    combined_output_file = joinpath(summary_dir, "combined_total_latitude_binning.png")
    fig.savefig(combined_output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    println("Latitude binning comparison plots saved to:")
    println("  Individual components: $output_file")
    println("  Combined totals: $combined_output_file")
end

function create_longitude_binning_comparison(all_results::Dict)
    """
    Create comparison plots of longitude-binned weighted correlations for all regions.
    """
    
    # Define the regions to include
    target_regions = ["SEPac", "NEPac", "SEAtl"]
    available_regions = [region for region in target_regions if haskey(all_results, region) && haskey(all_results[region], "longitude_binning")]
    
    if length(available_regions) == 0
        println("No regions with longitude binning data available")
        return
    end
    
    # Create output directory
    summary_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/regional_gridded_decomp/regional_decomposition"
    mkpath(summary_dir)
    
    # Get longitude centers (should be the same for all regions)
    lon_centers = all_results[available_regions[1]]["longitude_binning"]["lon_centers"]
    
    # Component names and colors
    component_names = ["sw_theta_1000", "sw_theta_700", "lw_theta_1000", "lw_theta_700"]
    component_labels = ["SW×θ₁₀₀₀", "SW×θ₇₀₀", "LW×θ₁₀₀₀", "LW×θ₇₀₀"]
    component_colors = ["red", "orange", "blue", "purple"]
    
    # Create subplot for each region
    n_regions = length(available_regions)
    fig, axes = plt.subplots(n_regions, 1, figsize=(12, 6*n_regions), sharex=true)
    
    if n_regions == 1
        axes = [axes]  # Ensure axes is always iterable
    end
    
    for (i, region) in enumerate(available_regions)
        ax = axes[i-1]
        lon_data = all_results[region]["longitude_binning"]
        
        # Plot components
        for (comp_name, comp_label, color) in zip(component_names, component_labels, component_colors)
            ax.plot(lon_centers, lon_data[comp_name], "o-", 
                   label=comp_label, linewidth=2, markersize=3, color=color)
        end
        
        # Plot total
        ax.plot(lon_centers, lon_data["total"], "o-", 
               label="Total", linewidth=3, markersize=4, color="black")
        
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Summed Weighted Correlation")
        ax.set_title(region)
        ax.grid(true, alpha=0.3)
        
        if i == n_regions - 1  # Only set xlabel for bottom plot
            ax.set_xlabel("Longitude (°)")
        end
        
        if i == n_regions - 1  # Only show legend on bottom plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        end
    end
    
    plt.suptitle("Regional Longitude-Binned Weighted Correlation Components", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save the comparison plot
    output_file = joinpath(summary_dir, "regional_longitude_binning_comparison.png")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Create combined plot with all regions on one axis for total correlations
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    region_colors = ["red", "blue", "green", "purple", "orange"]
    
    for (i, region) in enumerate(available_regions)
        lon_data = all_results[region]["longitude_binning"]
        ax.plot(lon_centers, lon_data["total"], "o-", 
               label=region, linewidth=2, markersize=3, 
               color=region_colors[i % length(region_colors)])
    end
    
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Summed Weighted Correlation")
    ax.set_xlabel("Longitude (°)")
    ax.set_title("Total Weighted Correlation by Longitude - All Regions")
    ax.grid(true, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    # Save the combined total plot
    combined_output_file = joinpath(summary_dir, "combined_total_longitude_binning.png")
    fig.savefig(combined_output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    println("Longitude binning comparison plots saved to:")
    println("  Individual components: $output_file")
    println("  Combined totals: $combined_output_file")
end

function create_distance_cumsum_comparison(all_results::Dict)
    """
    Create comparison plots of cumulative summed weighted correlations by distance for all regions.
    """
    
    # Define the regions to include
    target_regions = ["SEPac", "NEPac", "SEAtl"]
    available_regions = [region for region in target_regions if haskey(all_results, region) && haskey(all_results[region], "distance_analysis")]
    
    if length(available_regions) == 0
        println("No regions with distance analysis data available")
        return
    end
    
    # Create output directory
    summary_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/regional_gridded_decomp/regional_decomposition"
    mkpath(summary_dir)
    
    # Create individual subplot for each region showing raw and cumsum
    n_regions = length(available_regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(15, 5*n_regions), sharex="col")
    
    if n_regions == 1
        axes = reshape([axes], 1, 2)  # Ensure axes is 2D for consistent indexing
    end
    
    region_colors = ["red", "blue", "green", "purple", "orange"]
    
    for (i, region) in enumerate(available_regions)
        distance_data = all_results[region]["distance_analysis"]
        bin_centers = distance_data["bin_centers"]
        binned_results = distance_data["binned_results"]
        cumsum_data = distance_data["cumsum_weighted_corr"]
        
        color = region_colors[i % length(region_colors)]
        
        # Plot raw weighted correlation
        ax1 = axes[i-1, 1-1]
        ax1.plot(bin_centers, binned_results["total_weighted_corr"], "o-", 
                linewidth=2, markersize=3, color=color, label=region)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Weighted Correlation")
        ax1.set_title("$region - Raw Weighted Correlation")
        ax1.grid(true, alpha=0.3)
        ax1.legend()
        
        # Plot cumulative sum
        ax2 = axes[i-1, 2-1]
        ax2.plot(bin_centers, cumsum_data, "o-", 
                linewidth=2, markersize=3, color=color, label=region)
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Cumulative Weighted Correlation")
        ax2.set_title("$region - Cumulative Sum")
        ax2.grid(true, alpha=0.3)
        ax2.legend()
        
        # Set xlabel for bottom row
        if i == n_regions
            ax1.set_xlabel("Distance from Region Center (km)")
            ax2.set_xlabel("Distance from Region Center (km)")
        end
    end
    
    plt.suptitle("Regional Distance Analysis: Raw vs Cumulative", fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    # Save individual comparison plot
    individual_output_file = joinpath(summary_dir, "regional_distance_individual_comparison.png")
    fig.savefig(individual_output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Create combined cumsum comparison plot - all regions on same axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for (i, region) in enumerate(available_regions)
        distance_data = all_results[region]["distance_analysis"]
        bin_centers = distance_data["bin_centers"]
        binned_results = distance_data["binned_results"]
        cumsum_data = distance_data["cumsum_weighted_corr"]
        
        color = region_colors[i % length(region_colors)]
        
        # Plot raw weighted correlations together
        ax1.plot(bin_centers, binned_results["total_weighted_corr"], "o-", 
                linewidth=2, markersize=3, color=color, label=region, alpha=0.8)
        
        # Plot cumulative sums together
        ax2.plot(bin_centers, cumsum_data, "o-", 
                linewidth=2, markersize=3, color=color, label=region, alpha=0.8)
    end
    
    # Format raw correlation plot
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Distance from Region Center (km)")
    ax1.set_ylabel("Weighted Correlation")
    ax1.set_title("Raw Weighted Correlations - All Regions")
    ax1.grid(true, alpha=0.3)
    ax1.legend()
    
    # Format cumulative sum plot
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Distance from Region Center (km)")
    ax2.set_ylabel("Cumulative Weighted Correlation")
    ax2.set_title("Cumulative Weighted Correlations - All Regions")
    ax2.grid(true, alpha=0.3)
    ax2.legend()
    
    plt.suptitle("Core Region Distance Analysis Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save combined comparison plot
    combined_output_file = joinpath(summary_dir, "combined_distance_cumsum_comparison.png")
    fig.savefig(combined_output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Create cumsum-only comparison plot for cleaner visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for (i, region) in enumerate(available_regions)
        distance_data = all_results[region]["distance_analysis"]
        bin_centers = distance_data["bin_centers"]
        cumsum_data = distance_data["cumsum_weighted_corr"]
        
        color = region_colors[i % length(region_colors)]
        ax.plot(bin_centers, cumsum_data, "o-", 
               linewidth=3, markersize=4, color=color, label=region)
    end
    
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Distance from Region Center (km)")
    ax.set_ylabel("Cumulative Weighted Correlation")
    ax.set_title("Cumulative Weighted Correlation by Distance - Regional Comparison")
    ax.grid(true, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save cumsum-only plot
    cumsum_only_file = joinpath(summary_dir, "cumsum_only_distance_comparison.png")
    fig.savefig(cumsum_only_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    println("Distance cumsum comparison plots saved to:")
    println("  Individual analysis: $individual_output_file")
    println("  Combined raw+cumsum: $combined_output_file")
    println("  Cumsum only: $cumsum_only_file")
end

# Run the analysis
all_results = run_all_regional_analyses()
