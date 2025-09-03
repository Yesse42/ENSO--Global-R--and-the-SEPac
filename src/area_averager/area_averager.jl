"""
General purpose area averaging functions for spatial data over prescribed masks.

This module provides functions to:
1. Average spatial data over a spatial mask
2. Handle different lat/lon grids between data and mask
3. Calculate area-weighted averages using cosine latitude weights
4. Interpolate masks between different grids
5. Return results in lon x lat or lon x lat x time format
"""

using Statistics, Interpolations

"""
    average_over_mask(data, lat_data, lon_data, mask, lat_mask, lon_mask; 
                     interpolation_method=:nearest, cosine_weights=true)

Average spatial data over a prescribed spatial mask, with optional grid interpolation.

# Arguments
- `data`: 2D array (lon, lat) or 3D array (lon, lat, time)
- `lat_data`: Vector of latitude coordinates for data grid
- `lon_data`: Vector of longitude coordinates for data grid  
- `mask`: 2D array (lon, lat) with 1s where data should be included, 0s elsewhere
- `lat_mask`: Vector of latitude coordinates for mask grid
- `lon_mask`: Vector of longitude coordinates for mask grid

# Keywords
- `interpolation_method::Symbol=:nearest`: Method to interpolate mask (:nearest, :linear)
- `cosine_weights::Bool=true`: Whether to apply cosine latitude weighting
- `mask_threshold::Float64=0.5`: Threshold for binary mask after interpolation

# Returns
For 2D input (lon, lat):
- `averaged_value`: Single area-weighted average value
- `interpolated_mask`: The mask interpolated to the data grid (lon, lat)
- `weights`: The weights used for averaging (lon, lat)

For 3D input (lon, lat, time):
- `averaged_series`: Vector of area-weighted averages over time
- `interpolated_mask`: The mask interpolated to the data grid (lon, lat)
- `weights`: The weights used for averaging (lon, lat)

# Example
```julia
# Load your data and mask
data = rand(72, 36, 100)  # 72 lons, 36 lats, 100 time steps  
lat_data = range(-87.5, 87.5, length=36)
lon_data = range(0, 357.5, length=72)
mask = create_box_mask(lat_data, lon_data, lat_bounds=(-10, 10), lon_bounds=(160, 280))

# Average the data
avg_series, interp_mask, weights = average_over_mask(
    data, lat_data, lon_data, mask, lat_data, lon_data
)
```
"""
function average_over_mask(data, lat_data, lon_data, mask, lat_mask, lon_mask; 
                          interpolation_method::Symbol=:nearest, 
                          cosine_weights::Bool=true, mask_threshold::Float64=0.5)
    
    # Determine if data is 2D or 3D
    data_dims = ndims(data)
    if data_dims == 2
        n_lon, n_lat = size(data)
        has_time = false
    elseif data_dims == 3
        n_lon, n_lat, n_time = size(data)
        has_time = true
    else
        error("Data must be 2D (lon, lat) or 3D (lon, lat, time)")
    end
    
    # Check if grids are the same
    same_grid = (length(lat_data) == length(lat_mask) && 
                 length(lon_data) == length(lon_mask) &&
                 isapprox(lat_data, lat_mask, atol=1e-6) && 
                 isapprox(lon_data, lon_mask, atol=1e-6))
    
    if same_grid
        println("Data and mask on same grid - using mask directly")
        interpolated_mask = Float64.(mask)
    else
        println("Interpolating mask from $(length(lon_mask))×$(length(lat_mask)) to $(length(lon_data))×$(length(lat_data)) grid")
        interpolated_mask = interpolate_mask(mask, lat_mask, lon_mask, 
                                           lat_data, lon_data, interpolation_method)
    end
    
    # Convert to binary mask
    binary_mask = interpolated_mask .>= mask_threshold
    
    # Calculate cosine weights if requested
    if cosine_weights
        cos_lat_weights = cosd.(lat_data)  # Vector of cosine weights
        # Broadcast to match data grid (lon, lat)
        area_weights = binary_mask .* cos_lat_weights'  # (lon, lat) .* (1, lat) -> (lon, lat)
    else
        area_weights = Float64.(binary_mask)  # (lon, lat)
    end
    
    # Calculate total weight for normalization
    total_weight = sum(area_weights)
    
    if total_weight == 0
        error("No valid points in mask after interpolation")
    end
    
    println("Averaging over $(sum(binary_mask)) grid points with total weight $(round(total_weight, digits=3))")
    
    if !has_time
        # 2D case: return single average value
        # Handle NaN values
        valid_data = .!isnan.(data)
        combined_weights = area_weights .* valid_data  # (lon, lat)
        combined_weight_sum = sum(combined_weights)
        
        if combined_weight_sum > 0
            averaged_value = sum(data .* combined_weights) / combined_weight_sum
        else
            averaged_value = NaN
        end
        
        return averaged_value, interpolated_mask, area_weights
    else
        # 3D case: return time series
        averaged_series = Vector{Float64}(undef, n_time)
        
        for t in 1:n_time
            data_slice = data[:, :, t]  # (lon, lat)
            
            # Handle NaN values
            valid_data = .!isnan.(data_slice)
            combined_weights = area_weights .* valid_data  # (lon, lat)
            combined_weight_sum = sum(combined_weights)
            
            if combined_weight_sum > 0
                averaged_series[t] = sum(data_slice .* combined_weights) / combined_weight_sum
            else
                averaged_series[t] = NaN
            end
        end
        
        return averaged_series, interpolated_mask, area_weights
    end
end

"""
    interpolate_mask(mask, lat_mask, lon_mask, lat_target, lon_target, method=:nearest)

Interpolate a 2D mask from one lat/lon grid to another.

# Arguments
- `mask`: 2D array (lon, lat) to interpolate
- `lat_mask, lon_mask`: Source grid coordinates
- `lat_target, lon_target`: Target grid coordinates  
- `method`: Interpolation method (:nearest or :linear)

# Returns
- Interpolated mask on target grid (lon, lat)
"""
function interpolate_mask(mask, lat_mask, lon_mask, lat_target, lon_target, method=:nearest)
    
    if method == :nearest
        return interpolate_mask_nearest(mask, lat_mask, lon_mask, lat_target, lon_target)
    elseif method == :linear
        return interpolate_mask_linear(mask, lat_mask, lon_mask, lat_target, lon_target)
    else
        error("Unknown interpolation method: $method. Use :nearest or :linear")
    end
end

"""
Nearest neighbor interpolation for mask
"""
function interpolate_mask_nearest(mask, lat_mask, lon_mask, lat_target, lon_target)
    
    n_lon_target = length(lon_target)
    n_lat_target = length(lat_target)
    interpolated = zeros(Float64, n_lon_target, n_lat_target)
    
    # For each target grid point, find nearest source point
    for (i, target_lon) in enumerate(lon_target)
        for (j, target_lat) in enumerate(lat_target)
            
            # Find nearest longitude index (handle periodicity)
            lon_diffs = abs.(lon_mask .- target_lon)
            # Handle longitude wraparound
            lon_diffs_wrapped = min.(lon_diffs, abs.(lon_diffs .- 360))
            lon_idx = argmin(lon_diffs_wrapped)
            
            # Find nearest latitude index
            lat_idx = argmin(abs.(lat_mask .- target_lat))
            
            interpolated[i, j] = mask[lon_idx, lat_idx]
        end
    end
    
    return interpolated
end

"""
Linear interpolation for mask (bilinear)
"""
function interpolate_mask_linear(mask, lat_mask, lon_mask, lat_target, lon_target)
    
    # Create interpolation object
    # Note: Interpolations.jl expects (x, y) = (lon, lat) order
    # mask is (lon, lat), so we don't need to transpose
    itp = LinearInterpolation((lon_mask, lat_mask), mask, extrapolation_bc=Flat())
    
    n_lon_target = length(lon_target)
    n_lat_target = length(lat_target)
    interpolated = zeros(Float64, n_lon_target, n_lat_target)
    
    for (i, target_lon) in enumerate(lon_target)
        for (j, target_lat) in enumerate(lat_target)
            # Handle longitude wraparound if needed
            adj_lon = target_lon
            if target_lon < minimum(lon_mask)
                adj_lon = target_lon + 360
            elseif target_lon > maximum(lon_mask)
                adj_lon = target_lon - 360
            end
            
            interpolated[i, j] = itp(adj_lon, target_lat)
        end
    end
    
    return interpolated
end

"""
    create_box_mask(lat, lon; lat_bounds=nothing, lon_bounds=nothing)

Create a rectangular mask on a lat/lon grid.

# Arguments
- `lat`: Vector of latitude coordinates
- `lon`: Vector of longitude coordinates
- `lat_bounds`: Tuple (lat_min, lat_max) or nothing for all latitudes
- `lon_bounds`: Tuple (lon_min, lon_max) or nothing for all longitudes

# Returns
- 2D binary mask array (lon, lat)

# Example
```julia
# Create Pacific box mask
lat = range(-90, 90, length=36)
lon = range(0, 360, length=72)
mask = create_box_mask(lat, lon, lat_bounds=(-20, 20), lon_bounds=(160, 280))
```
"""
function create_box_mask(lat, lon; lat_bounds=nothing, lon_bounds=nothing)
    
    n_lat = length(lat)
    n_lon = length(lon)
    mask = ones(Int, n_lon, n_lat)
    
    if lat_bounds !== nothing
        lat_min, lat_max = lat_bounds
        lat_mask = (lat .>= lat_min) .& (lat .<= lat_max)
        # Broadcast lat_mask to (lon, lat) format
        mask = mask .* lat_mask'
    end
    
    if lon_bounds !== nothing
        lon_min, lon_max = lon_bounds
        
        if lon_max > lon_min  # Normal case
            lon_mask = (lon .>= lon_min) .& (lon .<= lon_max)
        else  # Wraparound case (e.g., 350° to 20°)
            lon_mask = (lon .>= lon_min) .| (lon .<= lon_max)
        end
        
        # Broadcast lon_mask to (lon, lat) format
        mask = mask .* lon_mask
    end
    
    return mask
end

"""
    create_circular_mask(lat, lon, center_lat, center_lon, radius_deg)

Create a circular mask centered at given coordinates.

# Arguments
- `lat, lon`: Grid coordinates
- `center_lat, center_lon`: Center of circle
- `radius_deg`: Radius in degrees

# Returns
- 2D binary mask array (lon, lat)
"""
function create_circular_mask(lat, lon, center_lat, center_lon, radius_deg)
    
    n_lat = length(lat)
    n_lon = length(lon)
    mask = zeros(Int, n_lon, n_lat)
    
    for (i, grid_lon) in enumerate(lon)
        for (j, grid_lat) in enumerate(lat)
            # Calculate great circle distance
            dist = haversine_distance(center_lat, center_lon, grid_lat, grid_lon)
            if dist <= radius_deg * 111.32  # Convert degrees to km (approximate)
                mask[i, j] = 1
            end
        end
    end
    
    return mask
end

"""
    haversine_distance(lat1, lon1, lat2, lon2)

Calculate great circle distance between two points in km.
"""
function haversine_distance(lat1, lon1, lat2, lon2)
    R = 6371.0  # Earth radius in km
    
    φ1, φ2 = deg2rad(lat1), deg2rad(lat2)
    Δφ = deg2rad(lat2 - lat1)
    Δλ = deg2rad(lon2 - lon1)
    
    a = sin(Δφ/2)^2 + cos(φ1) * cos(φ2) * sin(Δλ/2)^2
    c = 2 * atan(sqrt(a), sqrt(1-a))
    
    return R * c
end

"""
    validate_mask_coverage(data, mask, lat_data, lon_data, lat_mask, lon_mask; 
                          min_coverage=0.1)

Validate that the mask provides sufficient coverage of the data domain.

# Arguments
- `data`: Data array (lon, lat) or (lon, lat, time)
- `lat_data`, `lon_data`: Data coordinates
- `mask`: Mask array (lon, lat)
- `lat_mask`, `lon_mask`: Mask coordinates
- `min_coverage`: Minimum fraction of domain that should be covered

# Returns
- Boolean indicating whether coverage is sufficient
"""
function validate_mask_coverage(data, mask, lat_data, lon_data, lat_mask, lon_mask; 
                               min_coverage=0.1)
    
    # Interpolate mask to data grid
    interpolated_mask = interpolate_mask(mask, lat_mask, lon_mask, lat_data, lon_data, :nearest)
    
    # Calculate coverage
    total_points = length(lon_data) * length(lat_data)
    covered_points = sum(interpolated_mask .>= 0.5)
    coverage = covered_points / total_points
    
    println("Mask covers $(covered_points)/$(total_points) points ($(round(coverage*100, digits=1))%)")
    
    return coverage >= min_coverage
end

# Export main functions
export average_over_mask, interpolate_mask, create_box_mask, create_circular_mask, validate_mask_coverage
