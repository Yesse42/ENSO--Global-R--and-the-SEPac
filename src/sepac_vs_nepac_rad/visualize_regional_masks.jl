"""
    visualize_regional_masks.jl

This script loads and visualizes the regional masks created by determine_cre_regions.jl 
to verify that the upscaling/downscaling algorithms are working correctly.

Plots both the original CERES grid masks and the upscaled ERA5 grid masks for comparison,
and saves the plots to vis/compare_stratocumulus_regions/region_upscaled_correctly/
"""

using JLD2, PythonCall
@py import matplotlib.pyplot as plt
@py import matplotlib.patches as patches
@py import cartopy.crs as ccrs

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")

# Define paths
mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/compare_stratocumulus_regions/region_upscaled_correctly"

# Create output directory if it doesn't exist
mkpath(visdir)

# Load the mask data
println("Loading regional mask data...")
mask_data = JLD2.load(mask_file)

# Extract the data
ceres_masks = mask_data["regional_masks_ceres"]
era5_masks = mask_data["regional_masks_era5"] 
ceres_lon = mask_data["longitude_ceres"]
ceres_lat = mask_data["latitude_ceres"]
era5_lon = mask_data["longitude_era5"]
era5_lat = mask_data["latitude_era5"]
bounds = mask_data["bounds"]

println("Loaded masks for regions: ", keys(ceres_masks))

# Define colors for each region
colors_dict = Dict(
    "SEPac" => "red",
    "NEPac" => "blue", 
    "SEAtl" => "green",
    "SEPac_feedback_definition" => "orange",
    "SEPac_feedback_only" => "purple"
)

# Function to create a mask visualization
function plot_mask_comparison(region_name, ceres_mask, era5_mask, ceres_lon, ceres_lat, era5_lon, era5_lat; save_path=nothing)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                  subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))
    
    # Plot 1: CERES grid mask
    ax1.set_global()
    ax1.coastlines()
    ax1.set_title("$region_name - CERES Grid (Original)")
    
    # Convert mask to plottable data (NaN where false, 1 where true)
    ceres_mask_data = Float64.(ceres_mask)
    ceres_mask_data[.!ceres_mask] .= NaN
    
    # Plot the mask
    if any(ceres_mask)  # Only plot if mask has any true values
        c1 = ax1.contourf(ceres_lon, ceres_lat, ceres_mask_data', 
                         levels=[0.5, 1.5], colors=[colors_dict[region_name]], 
                         alpha=0.8, transform=ccrs.PlateCarree())
    end
    
    # Add gridlines for reference
    ax1.gridlines(draw_labels=true, alpha=0.3)
    
    # Plot 2: ERA5 grid mask  
    ax2.set_global()
    ax2.coastlines()
    ax2.set_title("$region_name - ERA5 Grid (Upscaled)")
    
    # Convert mask to plottable data
    era5_mask_data = Float64.(era5_mask)
    era5_mask_data[.!era5_mask] .= NaN
    
    # Plot the mask
    if any(era5_mask)  # Only plot if mask has any true values
        c2 = ax2.contourf(era5_lon, era5_lat, era5_mask_data',
                         levels=[0.5, 1.5], colors=[colors_dict[region_name]],
                         alpha=0.8, transform=ccrs.PlateCarree())
    end
    
    # Add gridlines for reference  
    ax2.gridlines(draw_labels=true, alpha=0.3)
    
    # Add region bounds if available (excluding placeholder bounds)
    if haskey(bounds, region_name) && bounds[region_name]["lat_min"] != 0
        region_bounds = bounds[region_name]
        lon_min = region_bounds["lon_min"]
        lon_max = region_bounds["lon_max"] 
        lat_min = region_bounds["lat_min"]
        lat_max = region_bounds["lat_max"]
        
        # Add rectangle to both plots
        for ax in [ax1, ax2]
            rect = patches.Rectangle((lon_min, lat_min),
                                   lon_max - lon_min,
                                   lat_max - lat_min,
                                   linewidth=2,
                                   edgecolor="black",
                                   facecolor="none",
                                   transform=ccrs.PlateCarree(),
                                   linestyle="--")
            ax.add_patch(rect)
        end
    end
    
    plt.tight_layout()
    
    if !isnothing(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        println("Saved: $save_path")
    end
    
    return fig
end

# Create individual plots for each region
println("\nCreating individual region plots...")
for (region_name, ceres_mask) in ceres_masks
    if haskey(era5_masks, region_name)
        era5_mask = era5_masks[region_name]
        
        # Skip if both masks are empty
        if !any(ceres_mask) && !any(era5_mask)
            println("Skipping $region_name - both masks are empty")
            continue
        end
        
        save_path = joinpath(visdir, "$(region_name)_mask_comparison.png")
        fig = plot_mask_comparison(region_name, ceres_mask, era5_mask, 
                                 ceres_lon, ceres_lat, era5_lon, era5_lat;
                                 save_path=save_path)
        plt.close(fig)
        
        # Print some statistics
        ceres_count = sum(ceres_mask)
        era5_count = sum(era5_mask)
        println("$region_name: CERES grid has $ceres_count points, ERA5 grid has $era5_count points")
    end
end

# Create a combined overview plot showing all regions on both grids
println("\nCreating combined overview plots...")

# CERES grid overview
fig1 = plot_global_heatmap(ceres_lat, ceres_lon, zeros(size(ceres_masks["SEPac"])); 
                          title="All Regional Masks - CERES Grid", 
                          colorbar_label="Region ID", central_longitude=180)
ax1 = fig1.axes[0]

# ERA5 grid overview  
fig2 = plot_global_heatmap(era5_lat, era5_lon, zeros(size(era5_masks["SEPac"]));
                          title="All Regional Masks - ERA5 Grid",
                          colorbar_label="Region ID", central_longitude=180)
ax2 = fig2.axes[0]

# Add all regions to both plots
legend_elements = []
for (i, (region_name, ceres_mask)) in enumerate(ceres_masks)
    if haskey(era5_masks, region_name) && (any(ceres_mask) || any(era5_masks[region_name]))
        color = get(colors_dict, region_name, "gray")
        
        # CERES plot
        if any(ceres_mask)
            ceres_mask_data = Float64.(ceres_mask)
            ceres_mask_data[.!ceres_mask] .= NaN
            ax1.contourf(ceres_lon, ceres_lat, ceres_mask_data',
                        levels=[0.5, 1.5], colors=[color],
                        alpha=0.7, transform=ccrs.PlateCarree())
        end
        
        # ERA5 plot
        era5_mask = era5_masks[region_name]
        if any(era5_mask)
            era5_mask_data = Float64.(era5_mask)
            era5_mask_data[.!era5_mask] .= NaN
            ax2.contourf(era5_lon, era5_lat, era5_mask_data',
                        levels=[0.5, 1.5], colors=[color],
                        alpha=0.7, transform=ccrs.PlateCarree())
        end
        
        # Add to legend
        push!(legend_elements, patches.Patch(color=color, alpha=0.7, label=region_name))
    end
end

# Add legends
ax1.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.02, 0.98))
ax2.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.02, 0.98))

# Save combined plots
fig1.savefig(joinpath(visdir, "all_regions_ceres_grid.png"), dpi=300, bbox_inches="tight")
fig2.savefig(joinpath(visdir, "all_regions_era5_grid.png"), dpi=300, bbox_inches="tight")

plt.close(fig1)
plt.close(fig2)

println("Saved combined overview plots")

# Create a side-by-side comparison for key regions
println("\nCreating side-by-side comparison for main stratocumulus regions...")

main_regions = ["SEPac", "NEPac", "SEAtl"]
valid_main_regions = [r for r in main_regions if haskey(ceres_masks, r) && haskey(era5_masks, r)]

if !isempty(valid_main_regions)
    n_regions = length(valid_main_regions)
    fig, axs = plt.subplots(2, n_regions, figsize=(5*n_regions, 10),
                           subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))
    
    if n_regions == 1
        axs = reshape([axs[1], axs[2]], 2, 1)
    end
    
    for (i, region_name) in enumerate(valid_main_regions)
        color = colors_dict[region_name]
        
        # CERES grid (top row)
        ax_ceres = axs[0, i-1]
        ax_ceres.set_global()
        ax_ceres.coastlines()
        ax_ceres.set_title("$region_name\nCERES Grid")
        
        ceres_mask = ceres_masks[region_name]
        if any(ceres_mask)
            ceres_mask_data = Float64.(ceres_mask)
            ceres_mask_data[.!ceres_mask] .= NaN
            ax_ceres.contourf(ceres_lon, ceres_lat, ceres_mask_data',
                            levels=[0.5, 1.5], colors=[color],
                            alpha=0.8, transform=ccrs.PlateCarree())
        end
        
        # ERA5 grid (bottom row)
        ax_era5 = axs[1, i-1]
        ax_era5.set_global()
        ax_era5.coastlines()
        ax_era5.set_title("$region_name\nERA5 Grid")
        
        era5_mask = era5_masks[region_name]
        if any(era5_mask)
            era5_mask_data = Float64.(era5_mask)
            era5_mask_data[.!era5_mask] .= NaN
            ax_era5.contourf(era5_lon, era5_lat, era5_mask_data',
                           levels=[0.5, 1.5], colors=[color],
                           alpha=0.8, transform=ccrs.PlateCarree())
        end
        
        # Add bounding box if available
        if haskey(bounds, region_name) && bounds[region_name]["lat_min"] != 0
            region_bounds = bounds[region_name]
            lon_min = region_bounds["lon_min"]
            lon_max = region_bounds["lon_max"]
            lat_min = region_bounds["lat_min"]
            lat_max = region_bounds["lat_max"]
            
            for ax in [ax_ceres, ax_era5]
                rect = patches.Rectangle((lon_min, lat_min),
                                       lon_max - lon_min,
                                       lat_max - lat_min,
                                       linewidth=1.5,
                                       edgecolor="black",
                                       facecolor="none",
                                       transform=ccrs.PlateCarree(),
                                       linestyle="--")
                ax.add_patch(rect)
            end
        end
    end
    
    plt.tight_layout()
    fig.savefig(joinpath(visdir, "main_stratocumulus_regions_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    println("Saved main stratocumulus regions comparison")
end

# Print summary statistics
println("\n" * "="^50)
println("SUMMARY STATISTICS")  
println("="^50)

for region_name in keys(ceres_masks)
    if haskey(era5_masks, region_name)
        ceres_mask = ceres_masks[region_name]
        era5_mask = era5_masks[region_name]
        
        ceres_count = sum(ceres_mask)
        era5_count = sum(era5_mask)
        
        ceres_total = length(ceres_mask)
        era5_total = length(era5_mask)
        
        ceres_pct = round(100 * ceres_count / ceres_total, digits=2)
        era5_pct = round(100 * era5_count / era5_total, digits=2)
        
        println("$region_name:")
        println("  CERES: $ceres_count/$ceres_total points ($ceres_pct%)")
        println("  ERA5:  $era5_count/$era5_total points ($era5_pct%)")
        
        if ceres_count > 0 && era5_count > 0
            ratio = era5_count / ceres_count
            println("  ERA5/CERES ratio: $(round(ratio, digits=2))")
        end
        println()
    end
end

println("All visualization plots saved to: $visdir")
