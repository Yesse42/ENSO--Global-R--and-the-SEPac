"""
Example script demonstrating the use of area_averager.jl functions.

This script shows how to:
1. Load data and create masks
2. Average time series over different spatial regions
3. Handle different grids between data and masks
4. Create and use various mask types
"""

using Dates, Statistics, NCDatasets, CSV, DataFrames
using JLD2  # For loading mask files

# Include the area averaging functions
include("area_averager.jl")

# Include utilities for data loading
include("../utils/load_funcs.jl")
include("../utils/constants.jl")

"""
Example 1: Average CERES data over a simple rectangular box
"""
function example_rectangular_box()
    println("\n=== Example 1: Rectangular Box Averaging ===")
    
    # Load some CERES data
    time_period = (Date(2001, 1, 1), Date(2010, 1, 1))
    ceres_variables = ["toa_net_all_mon"]
    ceres_data, ceres_coords = load_ceres_data(ceres_variables, time_period)
    
    # Extract data and coordinates
    data = ceres_data["toa_net_all_mon"]  # (time, lat, lon)
    lat = ceres_coords["latitude"]
    lon = ceres_coords["longitude"]
    time_axis = ceres_coords["time"]
    
    println("Loaded data with dimensions: ", size(data))
    
    # Create a tropical Pacific box mask
    mask = create_box_mask(lat, lon, 
                          lat_bounds=(-20, 20), 
                          lon_bounds=(160, 280))
    
    # Average the data over this region
    avg_series, interp_mask, weights = average_over_mask(
        data, lat, lon, mask, lat, lon,  # Same grid for data and mask
        time_dim=1, cosine_weights=true
    )
    
    println("Average tropical Pacific TOA net radiation:")
    println("  Mean: $(round(mean(avg_series), digits=2)) W/m²")
    println("  Std:  $(round(std(avg_series), digits=2)) W/m²")
    
    return avg_series, time_axis, mask
end

"""
Example 2: Use an existing mask file and interpolate to different grid
"""
function example_existing_mask()
    println("\n=== Example 2: Using Existing Mask File ===")
    
    # Load CERES data (different grid from ERA5)
    time_period = (Date(2001, 1, 1), Date(2005, 1, 1))
    ceres_variables = ["toa_net_all_mon"]
    ceres_data, ceres_coords = load_ceres_data(ceres_variables, time_period)
    
    data = ceres_data["toa_net_all_mon"]
    ceres_lat = ceres_coords["latitude"]
    ceres_lon = ceres_coords["longitude"]
    time_axis = ceres_coords["time"]
    
    # Load an existing mask (e.g., SEPac SST mask)
    mask_file = "../../data/SEPac_SST/sepac_sst_mask_and_weights.jld2"
    
    if isfile(mask_file)
        mask_data = load(mask_file)
        era5_mask = mask_data["final_mask"]
        era5_lat = mask_data["latitude"]
        era5_lon = mask_data["longitude"]
        
        println("Loaded ERA5 mask: $(size(era5_mask))")
        println("CERES grid: $(length(ceres_lon)) × $(length(ceres_lat))")
        
        # Average using different grids (mask will be interpolated)
        avg_series, interp_mask, weights = average_over_mask(
            data, ceres_lat, ceres_lon,      # Data grid
            era5_mask, era5_lat, era5_lon,   # Mask grid  
            time_dim=1, 
            interpolation_method=:nearest,
            cosine_weights=true
        )
        
        println("SEPac region CERES TOA net radiation:")
        println("  Mean: $(round(mean(avg_series), digits=2)) W/m²")
        println("  Std:  $(round(std(avg_series), digits=2)) W/m²")
        
        return avg_series, time_axis, interp_mask
    else
        println("Mask file not found: $mask_file")
        return nothing, nothing, nothing
    end
end

"""
Example 3: Create and use a circular mask
"""
function example_circular_mask()
    println("\n=== Example 3: Circular Mask ===")
    
    # Load data
    time_period = (Date(2001, 1, 1), Date(2003, 1, 1))
    ceres_variables = ["toa_net_all_mon"]
    ceres_data, ceres_coords = load_ceres_data(ceres_variables, time_period)
    
    data = ceres_data["toa_net_all_mon"]
    lat = ceres_coords["latitude"] 
    lon = ceres_coords["longitude"]
    time_axis = ceres_coords["time"]
    
    # Create circular mask centered on Niño 3.4 region
    center_lat = 0.0    # Equator
    center_lon = 190.0  # Central Pacific
    radius_deg = 15.0   # 15-degree radius
    
    mask = create_circular_mask(lat, lon, center_lat, center_lon, radius_deg)
    
    # Average over circular region
    avg_series, interp_mask, weights = average_over_mask(
        data, lat, lon, mask, lat, lon,
        time_dim=1, cosine_weights=true
    )
    
    println("Circular region ($(center_lat)°N, $(center_lon)°E, $(radius_deg)° radius):")
    println("  Mean: $(round(mean(avg_series), digits=2)) W/m²")
    println("  Std:  $(round(std(avg_series), digits=2)) W/m²")
    println("  Mask covers $(sum(mask)) grid points")
    
    return avg_series, time_axis, mask
end

"""
Example 4: Multiple regions comparison
"""
function example_multiple_regions()
    println("\n=== Example 4: Multiple Regions Comparison ===")
    
    # Load data
    time_period = (Date(2001, 1, 1), Date(2010, 1, 1))
    ceres_variables = ["toa_net_all_mon"]
    ceres_data, ceres_coords = load_ceres_data(ceres_variables, time_period)
    
    data = ceres_data["toa_net_all_mon"]
    lat = ceres_coords["latitude"]
    lon = ceres_coords["longitude"]
    time_axis = ceres_coords["time"]
    
    # Define multiple regions
    regions = Dict(
        "Niño3.4" => create_box_mask(lat, lon, lat_bounds=(-5, 5), lon_bounds=(190, 240)),
        "Tropical_Atlantic" => create_box_mask(lat, lon, lat_bounds=(-20, 20), lon_bounds=(300, 360)),
        "Tropical_Indian" => create_box_mask(lat, lon, lat_bounds=(-20, 20), lon_bounds=(40, 120)),
        "Global_Tropics" => create_box_mask(lat, lon, lat_bounds=(-30, 30), lon_bounds=nothing)
    )
    
    results = Dict()
    
    for (name, mask) in regions
        avg_series, _, _ = average_over_mask(
            data, lat, lon, mask, lat, lon,
            time_dim=1, cosine_weights=true
        )
        
        results[name] = avg_series
        
        println("$name region:")
        println("  Mean: $(round(mean(avg_series), digits=2)) W/m²")
        println("  Std:  $(round(std(avg_series), digits=2)) W/m²")
        println("  Grid points: $(sum(mask))")
    end
    
    return results, time_axis, regions
end

"""
Main execution function
"""
function main()
    println("Area Averager Examples")
    println("======================")
    
    try
        # Run examples
        avg1, time1, mask1 = example_rectangular_box()
        avg2, time2, mask2 = example_existing_mask()
        avg3, time3, mask3 = example_circular_mask()
        results4, time4, masks4 = example_multiple_regions()
        
        # Create a simple output
        if !isnothing(avg1) && !isnothing(time1)
            df = DataFrame(
                Date = time1,
                Tropical_Pacific = avg1
            )
            
            if !isnothing(avg2)
                df.SEPac_Region = avg2[1:min(length(avg2), nrow(df))]
            end
            
            if !isnothing(avg3)
                df.Circular_Region = avg3[1:min(length(avg3), nrow(df))]
            end
            
            # Save results
            outdir = "../../data/area_averager_examples"
            if !isdir(outdir)
                mkpath(outdir)
            end
            
            CSV.write(joinpath(outdir, "example_averages.csv"), df)
            println("\nResults saved to: $(joinpath(outdir, "example_averages.csv"))")
        end
        
        println("\nAll examples completed successfully!")
        
    catch e
        println("Error running examples: $e")
        println("This may be due to missing data files - check that CERES data is available")
    end
end

# Run examples if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
