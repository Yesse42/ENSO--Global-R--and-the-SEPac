"""
Functions for comparing local and non-local surface and aloft variables with lag correlations.
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")

using Statistics, DataFrames, Dictionaries, JLD2, CSV, Dates

"""
    calculate_lag_correlations(reference_data, lagged_data_dict; lags=-24:24)

Calculate correlation as a function of lag between reference data and lagged variables.

# Arguments
- `reference_data`: Vector of reference values (e.g., radiation data)
- `lagged_data_dict`: Dictionary mapping lag values to corresponding lagged data vectors
- `lags`: Range of lags to calculate correlations for (default: -24:24)

# Returns
- `Dict{Int, Float64}`: Sorted dictionary mapping lags to correlation values
"""
function calculate_lag_correlations(reference_data, lagged_data_dict; lags)
    
    correlations = Dict{Int, Float64}()
    
    for lag in lags
        # Get lagged data for this lag
        if haskey(lagged_data_dict, lag)
            lagged_data = lagged_data_dict[lag]
            
            # Find valid indices (non-missing values)
            valid_indices = .!(ismissing.(reference_data) .| ismissing.(lagged_data))
            
            if sum(valid_indices) > 0
                # Calculate correlation
                correlations[lag] = cor(reference_data[valid_indices], lagged_data[valid_indices])
            else
                correlations[lag] = NaN
            end
        else
            correlations[lag] = NaN
        end
    end
    
    # Return sorted dictionary
    return sortkeys!(correlations)
end

# Load mask area calculations
mask_areas = JLD2.load("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/mask_area_calculations.jld2")

valid_regions = keys(mask_areas["area_results_ceres"])

# Define radiation variables and time period
rad_variables = ["gtoa_net_all_mon", "gtoa_net_lw_mon", "gtoa_net_sw_mon"]
date_range = (Date(2000, 3), Date(2024, 3, 31))

# Load global radiation time series
global_rad_data, global_coords = load_new_ceres_data(rad_variables, date_range)

global_coords["time"] = round.(global_coords["time"], Dates.Month(1), RoundDown)

nonlocal_savedir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/nonlocal_radiation_time_series"

# Create save directory if it doesn't exist
mkpath(nonlocal_savedir)

for region in valid_regions
    println("Processing region: $region")
    
    # Load local radiation CSV for lag 0
    local_rad_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/local_region_time_series/ceres_region_avg_$(region).csv"
    local_rad_df = CSV.read(local_rad_file, DataFrame)

    display(local_rad_df)

    filter!(row -> in(row.date, global_coords["time"]), local_rad_df)

    display(local_rad_df)
    
    # Get area weights from mask calculations
    local_area = mask_areas["area_results_ceres"][region]["masked_area"]
    total_area = mask_areas["total_areas"]["ceres_total_area"]
    global_area = total_area - local_area  # Complement area
    
    # Weight the radiation time series by their angular areas
    weighted_global_rad = Dict{String, Vector}()
    weighted_local_rad = Dict{String, Vector}()
    nonlocal_rad = Dict{String, Vector}()
    
    for (i, var_name) in enumerate(rad_variables)
        # Extract variable base name (remove 'g' prefix for local matching)
        local_var_name = replace(var_name, "g" => "")
        
        # Weight by areas
        weighted_global_rad[var_name] = global_rad_data[var_name] .* global_area
        weighted_local_rad[local_var_name] = local_rad_df[!, local_var_name] .* local_area
        
        # Create residual nonlocal radiation time series
        # nonlocal = weighted_global - weighted_local
        nonlocal_rad[var_name] = weighted_global_rad[var_name] .- weighted_local_rad[local_var_name]
        
        println("  Created nonlocal time series for $var_name")
    end
    
    # Store results for this region (can be used later)
    println("  Completed processing for $region")
    println("  Local area: $(round(local_area, digits=2))")
    println("  Global (complement) area: $(round(global_area, digits=2))")
    
    # Create DataFrames for saving
    # Global radiation DataFrame
    global_df = DataFrame(date = global_coords["time"])
    for (var_name, data) in weighted_global_rad
        global_df[!, var_name] = data
    end
    
    # Local radiation DataFrame  
    local_df = DataFrame(date = global_coords["time"])
    for (var_name, data) in weighted_local_rad
        local_df[!, var_name] = data
    end
    
    # Nonlocal radiation DataFrame
    nonlocal_df = DataFrame(date = global_coords["time"])
    for (var_name, data) in nonlocal_rad
        nonlocal_df[!, var_name] = data
    end
    
    # Save to CSV files
    CSV.write(joinpath(nonlocal_savedir, "$(region)_global_radiation.csv"), global_df)
    CSV.write(joinpath(nonlocal_savedir, "$(region)_local_radiation.csv"), local_df)
    CSV.write(joinpath(nonlocal_savedir, "$(region)_nonlocal_radiation.csv"), nonlocal_df)
    
    println("  Saved radiation data to CSV files for $region")
    
end

