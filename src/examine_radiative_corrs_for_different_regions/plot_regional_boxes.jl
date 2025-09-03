using PythonCall
@py import cartopy.crs as ccrs, matplotlib.pyplot as plt, matplotlib.patches as patches

cd(@__DIR__)  # Change to script directory

include("regions.jl")

"""
    plot_regional_boxes(;save_path=nothing, show_plot=true)

Plot the regional boxes defined in regions.jl on a global map using cartopy.
Handles longitude wraparound properly.

# Arguments
- `save_path`: Optional path to save the figure
- `show_plot`: Whether to display the plot (default: true)
"""
function plot_regional_boxes(; save_path=nothing, show_plot=true)
    # Create figure with Robinson projection
    fig, ax = plt.subplots(figsize=(12, 8), 
                          subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))
    
    # Set up the map
    ax.set_global()
    ax.coastlines()
    ax.gridlines(draw_labels=true, alpha=0.5)
    ax.set_title("Regional Analysis Boxes", fontsize=16, fontweight="bold")
    
    # Define colors for each region
    colors = ["red", "blue", "green", "orange", "purple"]
    
    # Define the regions and their coordinates
    regions = [
        ("sepac", sepac_lon, sepac_lat),
        ("nepac", nepac_lon, nepac_lat), 
        ("weqpac", weqpac_lon, weqpac_lat),
        ("kuroshio", kuroshio_lon, kuroshio_lat),
        ("rockies", rockies_lon, rockies_lat)
    ]
    
    # Plot each region
    for (i, (name, lon_range, lat_range)) in enumerate(regions)
        color = colors[i]
        
        # Handle longitude wraparound
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        
        # Convert longitude to -180 to 180 range for plotting
        if lon_min > 180
            lon_min_plot = lon_min - 360
        else
            lon_min_plot = lon_min
        end
        
        if lon_max > 180
            lon_max_plot = lon_max - 360
        else
            lon_max_plot = lon_max
        end
        
        # Check if we need to handle wraparound
        if lon_min_plot > lon_max_plot
            # Case where region crosses the dateline
            # Plot two rectangles: one from lon_min to 180, another from -180 to lon_max
            
            # Rectangle 1: from lon_min to 180
            rect1 = patches.Rectangle(
                (lon_min_plot, lat_min), 
                180 - lon_min_plot, 
                lat_max - lat_min,
                linewidth=2, 
                edgecolor=color, 
                facecolor="none",
                transform=ccrs.PlateCarree(),
                label=name
            )
            ax.add_patch(rect1)
            
            # Rectangle 2: from -180 to lon_max  
            rect2 = patches.Rectangle(
                (-180, lat_min),
                lon_max_plot - (-180),
                lat_max - lat_min,
                linewidth=2,
                edgecolor=color,
                facecolor="none", 
                transform=ccrs.PlateCarree()
            )
            ax.add_patch(rect2)
            
        else
            # Normal case - single rectangle
            rect = patches.Rectangle(
                (lon_min_plot, lat_min),
                lon_max_plot - lon_min_plot,
                lat_max - lat_min,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                transform=ccrs.PlateCarree(),
                label=name
            )
            ax.add_patch(rect)
        end
        
        # Add region label at the center
        center_lon = (lon_min + lon_max) / 2
        center_lat = (lat_min + lat_max) / 2
        
        # Convert center longitude for plotting
        if center_lon > 180
            center_lon_plot = center_lon - 360
        else
            center_lon_plot = center_lon
        end
        
        ax.text(center_lon_plot, center_lat, uppercase(name), 
                transform=ccrs.PlateCarree(),
                fontsize=10, fontweight="bold", color=color,
                ha="center", va="center",
                bbox=Dict("boxstyle" => "round,pad=0.3", "facecolor" => "white", "alpha" => 0.8))
    end
    
    # Add legend
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))
    
    # Save if requested
    if save_path !== nothing
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        println("Figure saved to: $save_path")
    end
    
    # Show plot if requested
    if show_plot
        plt.show()
    end
    
    return fig, ax
end

"""
    print_region_info()

Print information about each region including their coordinate ranges.
"""
function print_region_info()
    println("Regional Analysis Boxes:")
    println("=" ^ 50)
    
    regions = [
        ("SEPac (Southeast Pacific)", sepac_lon, sepac_lat),
        ("NEPac (Northeast Pacific)", nepac_lon, nepac_lat),
        ("WEqPac (Western Equatorial Pacific)", weqpac_lon, weqpac_lat),
        ("Kuroshio", kuroshio_lon, kuroshio_lat),
        ("Rockies", rockies_lon, rockies_lat)
    ]
    
    for (name, lon_range, lat_range) in regions
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        
        # Show both original and -180 to 180 coordinates
        if lon_min > 180 || lon_max > 180
            lon_min_180 = lon_min > 180 ? lon_min - 360 : lon_min
            lon_max_180 = lon_max > 180 ? lon_max - 360 : lon_max
            println("$name:")
            println("  Original: Lon $(lon_min)° to $(lon_max)°, Lat $(lat_min)° to $(lat_max)°")
            println("  -180/180: Lon $(lon_min_180)° to $(lon_max_180)°, Lat $(lat_min)° to $(lat_max)°")
        else
            println("$name:")
            println("  Lon $(lon_min)° to $(lon_max)°, Lat $(lat_min)° to $(lat_max)°")
        end
        println()
    end
end

# Example usage
if true
    println("Plotting regional boxes...")
    print_region_info()
    
    # Create the plot
    fig, ax = plot_regional_boxes(save_path="../../vis/radiative_corrs_different_regions/regional_boxes.png")
    
    println("Done!")
end
