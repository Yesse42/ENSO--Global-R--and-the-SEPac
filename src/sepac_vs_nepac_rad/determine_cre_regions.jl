"""
    This script plots the climatology of cloud radiative effects averaged over the year using the CERES data, and then displays latlon polygons onto that climatology map to determine where the SEPac, NEPac, and Namibian stratocumulus decks are located.
"""
using GMT, JLD2, Interpolations

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")

"""
    is_in_lonlat_bounds(lon_point, lat_point, lon_min, lon_max, lat_min, lat_max)

Check if a point (lon_point, lat_point) is within the specified longitude and latitude bounds.
Handles longitude wraparound from 0° to 360° and Greenwich meridian crossing.

# Arguments
- `lon_point`: Longitude of the point to test
- `lat_point`: Latitude of the point to test  
- `lon_min`: Minimum longitude of bounding box
- `lon_max`: Maximum longitude of bounding box
- `lat_min`: Minimum latitude of bounding box
- `lat_max`: Maximum latitude of bounding box

# Returns
- `Bool`: true if point is within bounds, false otherwise

# Notes
- Handles both [0, 360] and [-180, 180] longitude formats by normalizing to [0, 360]
- Handles cases where bounding box crosses the 0°/360° meridian (e.g., -15° to 15°)
- Latitude bounds are straightforward as they don't wrap around
"""
function is_in_lonlat_bounds(lon_point, lat_point, lon_min, lon_max, lat_min, lat_max)
    # Check latitude bounds (straightforward, no wraparound)
    lat_in_bounds = lat_min <= lat_point <= lat_max
    
    # Normalize longitudes to [0, 360] for consistent comparison
    lon_point_norm = mod(lon_point, 360)
    lon_min_norm = mod(lon_min, 360)
    lon_max_norm = mod(lon_max, 360)
    
    # Check longitude bounds (handle wraparound)
    if lon_min_norm <= lon_max_norm
        # Normal case: bounding box doesn't cross 0°/360° meridian
        lon_in_bounds = lon_min_norm <= lon_point_norm <= lon_max_norm
    else
        # Wraparound case: bounding box crosses 0°/360° meridian
        # Point is in bounds if it's either:
        # 1. Greater than lon_min_norm (in the "high" part, e.g., 350°-360°)
        # 2. Less than lon_max_norm (in the "low" part, e.g., 0°-10°)
        lon_in_bounds = (lon_point_norm >= lon_min_norm) || (lon_point_norm <= lon_max_norm)
    end
    
    return lat_in_bounds && lon_in_bounds
end

"""
    create_bbox_mask(lon_grid, lat_grid, lon_min, lon_max, lat_min, lat_max)

Create a boolean mask for points within the specified longitude/latitude bounding box.
Handles longitude wraparound from 0° to 360°.

# Arguments
- `lon_grid`: 2D array of longitude values
- `lat_grid`: 2D array of latitude values
- `lon_min`, `lon_max`: Longitude bounds
- `lat_min`, `lat_max`: Latitude bounds

# Returns
- `BitArray`: Boolean mask with same dimensions as input grids
"""
function create_bbox_mask(lon_grid, lat_grid, lon_min, lon_max, lat_min, lat_max)
    mask = BitArray(undef, size(lon_grid))
    
    for i in eachindex(lon_grid)
        mask[i] = is_in_lonlat_bounds(lon_grid[i], lat_grid[i], lon_min, lon_max, lat_min, lat_max)
    end
    
    return mask
end

"""
    haversine_distance(lat1, lon1, lat2, lon2)

Calculate the great-circle distance between two points on Earth using the Haversine formula.

# Arguments
- `lat1`, `lon1`: Latitude and longitude of first point (in degrees)
- `lat2`, `lon2`: Latitude and longitude of second point (in degrees)

# Returns
- Angular distance in radians between the two points
"""
function haversine_distance(lon1, lat1, lon2, lat2)
    # Convert degrees to radians
    φ1 = deg2rad(lat1)
    φ2 = deg2rad(lat2)
    Δφ = deg2rad(lat2 - lat1)
    Δλ = deg2rad(lon2 - lon1)
    
    # Haversine formula
    a = sin(Δφ/2)^2 + cos(φ1) * cos(φ2) * sin(Δλ/2)^2
    a = clamp(a, 0, 1)  # Ensure a is within [0, 1] to avoid domain errors
    c = 2 * atan(sqrt(a), sqrt(1-a))
    
    # Return angular distance in radians
    return c
end

haversine_distance(lonlat1, lonlat2) = haversine_distance(lonlat1[1], lonlat1[2], lonlat2[1], lonlat2[2])

function nearest_neighbor_interpolate_grid(original_lon_lat, new_lon_lat; metric_func = haversine_distance)

    function min_dist_idx(target_point)
        return argmin(metric_func(target_point, original_grid_point) for original_grid_point in original_lon_lat)
    end

    return min_dist_idx.(new_lon_lat)
end

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/compare_stratocumulus_regions"
savedir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison"

# Create directories if they don't exist
mkpath(visdir)
mkpath(savedir)

net_cre_var = "toa_cre_net_mon"
time_period = (Date(2000, 3), Date(2025, 4)) # in time period does not include the right endpoint

ceres_data, ceres_coords = load_new_ceres_data([net_cre_var], time_period)

mean_cre = mapslices(mean, ceres_data[net_cre_var], dims=(3,))[:,:,1]  # mean over time dimension
lat = ceres_coords["latitude"]
lon = ceres_coords["longitude"]

ceres_lat = lat
ceres_lon = lon

fig = plot_global_heatmap(lat, lon, mean_cre; 
                          title="Climatological Mean Cloud Radiative Effect (2000-2025)", 
                          colorbar_label="W/m²", central_longitude = 160)

ax = fig.axes[0]

cre_thresh = -36

ax.contour(lon, lat, mean_cre', levels=[cre_thresh], colors="k", linestyles="--", transform=ccrs.PlateCarree())

# Save the first figure (bounding boxes)
plt.savefig(joinpath(visdir, "cre_climatology_with_bounding_boxes.png"), dpi=300, bbox_inches="tight")
println("Saved figure: cre_climatology_with_bounding_boxes.png")

bounds = Dict{String, Dict{String, Float64}}()

# SEPac bounds: sepac_lon, sepac_lat = ((-110, -69.3) .+ 360, (-40, 0))
bounds["SEPac"] = Dict("lat_min"=>-34, "lat_max"=>0, "lon_min"=>250, "lon_max"=>290.7)
bounds["NEPac"] = Dict("lat_min"=>15, "lat_max"=>38, "lon_min"=>210, "lon_max"=>260)
bounds["SEAtl"] = Dict("lat_min"=>-30, "lat_max"=>-7, "lon_min"=>-15, "lon_max"=>15)

# Plot the bounds as rectangles on the map
@py import matplotlib.patches as patches

# Define colors for each region
colorsdict = Dict("SEPac" => "red", "NEPac" => "blue", "SEAtl" => "green")

for (region_name, region_bounds) in bounds
    # Create rectangle coordinates
    lon_min = region_bounds["lon_min"]
    lon_max = region_bounds["lon_max"]
    lat_min = region_bounds["lat_min"]
    lat_max = region_bounds["lat_max"]
    
    # Create rectangle patch
    rect = patches.Rectangle((lon_min, lat_min), 
                           lon_max - lon_min, 
                           lat_max - lat_min,
                           linewidth=2, 
                           edgecolor=colorsdict[region_name], 
                           facecolor="none",
                           transform=ccrs.PlateCarree(),
                           label=region_name)
    
    ax.add_patch(rect)
end

# Add legend
ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))
fig.savefig(joinpath(visdir, "cre_climatology_with_bounding_boxes.png"), dpi=300, bbox_inches="tight")

plt.close(fig)

# Create masks for each region using CRE threshold, bounding boxes, and land/ocean test
println("Creating regional masks...")

# Create coordinate meshgrid (original data is lon x lat)
lon_grid = repeat(reshape(lon, :, 1), 1, length(lat))  # lon x lat
lat_grid = repeat(reshape(lat, 1, :), length(lon), 1)  # lon x lat

# Flatten for GMT processing
lon_flat = vec(lon_grid)
lat_flat = vec(lat_grid)
points_data = [lon_flat lat_flat]

# Use GMT to determine which points are over ocean (not land)
println("Testing points for land/ocean using GMT...")
# Alternative approach: Create a land mask grid and interpolate
# First, create a land mask at the resolution of our data
lon_range = (minimum(lon), maximum(lon))
lat_range = (minimum(lat), maximum(lat))
land_mask_grid = grdlandmask(region=(lon_range[1], lon_range[2], lat_range[1], lat_range[2]), 
                           inc=(lon[2]-lon[1], lat[2]-lat[1]), 
                           res=:crude)

# Extract the land mask values and interpolate to our grid points
# grdlandmask returns 1 for land, 0 for ocean, NaN for lakes
land_values = land_mask_grid.z'
ocean_mask = land_values .== 0  # True for ocean points

# Create CRE threshold mask
cre_mask = mean_cre .< cre_thresh

# Create regional masks
regional_masks = Dict()

for (region_name, region_bounds) in bounds
    println("Creating mask for $region_name...")
    
    # Bounding box mask using wraparound-aware function
    bbox_mask = create_bbox_mask(lon_grid, lat_grid, 
                                region_bounds["lon_min"], region_bounds["lon_max"],
                                region_bounds["lat_min"], region_bounds["lat_max"])
    
    # Combine all conditions: in bounding box AND over ocean AND meets CRE threshold
    regional_masks[region_name] = bbox_mask .& ocean_mask .& cre_mask
end

# Create fresh plot with masked regions
println("Creating plot with regional masks...")
fig = plot_global_heatmap(lat, lon, mean_cre; 
                          title="Cloud Radiative Effect with Stratocumulus Regions", 
                          colorbar_label="W/m²", central_longitude = 160)

ax = fig.axes[0]

# Add CRE threshold contour
ax.contour(lon, lat, mean_cre', levels=[cre_thresh], colors="k", linestyles="--", transform=ccrs.PlateCarree())

# Plot the masked regions
for (region_name, mask) in regional_masks
    # Convert mask to plottable data (NaN where false, 1 where true)
    mask_data = Float64.(mask)
    mask_data[.!mask] .= NaN
    
    # Plot the mask as filled contours
    ax.contourf(lon, lat, mask_data', levels=[0.5, 1.5], 
               colors=[colorsdict[region_name]], alpha=0.7, 
               transform=ccrs.PlateCarree())
end

# Add legend
legend_elements = [patches.Patch(color=colorsdict[name], alpha=0.7, label=name) for name in keys(bounds)]
ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.02, 0.98))

# Save the second figure (with masks)
plt.savefig(joinpath(visdir, "cre_climatology_with_stratocumulus_masks.png"), dpi=300, bbox_inches="tight")
println("Saved figure: cre_climatology_with_stratocumulus_masks.png")

plt.close(fig)

era5_data, era5_coords = load_era5_data(["lsm"], (Date(2000,3), Date(2025,4)))

eralat = era5_coords["latitude"]
eralon = era5_coords["longitude"]

# Upscale masks to ERA5 grid using nearest neighbor interpolation
println("Upscaling masks to ERA5 grid using nearest neighbor interpolation...")
upscaled_masks = Dict()

println("Upscaling masks")

era_lonlat_grid = tuple.(eralon, eralat')
ceres_lonlat_grid = tuple.(lon, lat')

ceres_to_era5_indices = nearest_neighbor_interpolate_grid(ceres_lonlat_grid, era_lonlat_grid; metric_func=haversine_distance)

era5_lsm = era5_data["lsm"][:,:,1]  # Take first time slice
era5_ocean_mask = era5_lsm .< 0.5  # ERA5 LSM: 0=ocean, 1=land

println("Downscaling Masks")

era5_to_ceres_indices = nearest_neighbor_interpolate_grid(era_lonlat_grid, ceres_lonlat_grid; metric_func=haversine_distance)

# Save the coordinate mapping indices
println("Saving coordinate mapping indices...")
coord_mapping_data = Dict(
    :era5_to_ceres_indices => era5_to_ceres_indices,
    :ceres_to_era5_indices => ceres_to_era5_indices,
    :era5_longitude => eralon,
    :era5_latitude => eralat,
    :ceres_longitude => lon,
    :ceres_latitude => lat,
    :description => "Nearest neighbor mapping indices between ERA5 and CERES grids"
)

coord_mapping_file = joinpath(savedir, "era5_ceres_coordinate_mapping.jld2")
jldsave(coord_mapping_file; coord_mapping_data...)
println("Saved coordinate mapping indices to: $coord_mapping_file")

for (region_name, mask) in regional_masks
    println("Upscaling $region_name mask...")
    
    # Apply ERA5 land-sea mask (keep only ocean points)
    upscaled_mask = mask[ceres_to_era5_indices]

    # Combine upscaled mask with ERA5 ocean mask
    upscaled_masks[region_name] = upscaled_mask .& era5_ocean_mask
    
    println("  Original mask size: $(size(mask))")
    println("  Upscaled mask size: $(size(upscaled_masks[region_name]))")
    println("  Points in upscaled mask: $(sum(upscaled_masks[region_name]))")
end

#Now load in the other SEPac's mask
other_region_name = "SEPac_feedback_definition"
path_to_mask = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/SEPac_SST/sepac_sst_mask_and_weights.jld2"
other_region_data = JLD2.load(path_to_mask)
other_region_mask = other_region_data["final_mask"] .> 0.5  # Convert to boolean

#Convert this grid from its native era5 to ceres grid
# Downscale other_region_mask from ERA5 grid to CERES grid using haversine distance
println("Downscaling $other_region_name mask from ERA5 to CERES grid using haversine distance...")

# Perform nearest neighbor interpolation from ERA5 to CERES grid
downscaled_other_mask = other_region_mask[era5_to_ceres_indices]

# Add to regional_masks dictionary for consistency
regional_masks[other_region_name] = downscaled_other_mask

println("  Original ERA5 mask size: $(size(other_region_mask))")
println("  Downscaled CERES mask size: $(size(downscaled_other_mask))")
println("  Points in downscaled mask: $(sum(downscaled_other_mask))")

# Save the masks and coordinates to JLD2 file
println("Saving masks and coordinates...")
regional_masks[other_region_name] = downscaled_other_mask  # Add the downscaled mask to the dictionary
upscaled_masks[other_region_name] = other_region_mask  # Add the original ERA5 mask to the upscaled dictionary
bounds[other_region_name] = Dict("lat_min"=>0, "lat_max"=>0, "lon_min"=>0, "lon_max"=>0)  # Placeholder bounds

#Lastly save the region which is in the feedback definition but not in the cre definition
sepac_cre_mask = upscaled_masks["SEPac"]
high_res_mask = @. other_region_mask && !sepac_cre_mask

sepac_low_res_cre_mask = regional_masks["SEPac"]
low_res_mask = @. downscaled_other_mask && !sepac_low_res_cre_mask

#Now push these masks to the dicts too
region_name = "SEPac_feedback_only"
regional_masks[region_name] = low_res_mask
upscaled_masks[region_name] = high_res_mask
bounds[region_name] = Dict("lat_min"=>0, "lat_max"=>0, "lon_min"=>0, "lon_max"=>0)  # Placeholder bounds

# Save comprehensive dataset with all masks
mask_data = Dict(
    :regional_masks_ceres => regional_masks,           # Original CERES grid masks
    :regional_masks_era5 => upscaled_masks,           # Upscaled ERA5 grid masks
    :longitude_ceres => lon,                          # CERES longitude
    :latitude_ceres => lat,                           # CERES latitude
    :longitude_era5 => eralon,                        # ERA5 longitude
    :latitude_era5 => eralat,                         # ERA5 latitude
    :cre_threshold => cre_thresh,
    :bounds => bounds,
    :description => "Stratocumulus region masks: original CERES grid and upscaled ERA5 grid versions"
)

mask_file = joinpath(savedir, "stratocumulus_region_masks.jld2")
jldsave(mask_file; mask_data...)
println("Saved comprehensive masks to: $mask_file")
