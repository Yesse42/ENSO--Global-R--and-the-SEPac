"""
This script cleans and processes the new CERES dataset. It will help me avoid annoying sign errors lol
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
using NCDatasets, StatsBase, Dates

ceres_data_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/NEW"

# Variable names for reading from datasets
gridded_lw_var = "toa_lw_all_mon"
global_lw_var = "gtoa_lw_all_mon"

gridded_sw_var = "toa_sw_all_mon"
global_sw_var = "gtoa_sw_all_mon"

gridded_solar_var = "solar_mon"
global_solar_var = "gsolar_mon"

# Multiply the gridded longwave and global lw variables by -1 to make them positive downwards, calling them net longwave
println("Calculating net longwave variables...")
gridded_net_lw = Dataset(joinpath(ceres_data_dir, "ceres_gridded.nc"), "r") do ds
    -1 * ds[gridded_lw_var][:]
end
global_net_lw = Dataset(joinpath(ceres_data_dir, "ceres_global.nc"), "r") do ds
    -1 * ds[global_lw_var][:]
end

# Make a net shortwave variable by subtracting the sw_var from the solar var for both gridded and global datasets, calling them net shortwave
println("Calculating net shortwave variables...")
gridded_net_sw = Dataset(joinpath(ceres_data_dir, "ceres_gridded.nc"), "r") do ds
    ds[gridded_solar_var][:] - ds[gridded_sw_var][:]
end
global_net_sw = Dataset(joinpath(ceres_data_dir, "ceres_global.nc"), "r") do ds
    ds[global_solar_var][:] - ds[global_sw_var][:]
end

# Check if the net sw and lw vars already exist in their respective netcdfs. If they do not, create the variables. Then, regardless of if the variables already existed or not, set the values of the variables to the newly calculated net sw and lw values.
gridded_lw_save_var = "toa_net_lw_mon"
global_lw_save_var = "gtoa_net_lw_mon"

gridded_sw_save_var = "toa_net_sw_mon"
global_sw_save_var = "gtoa_net_sw_mon"

# Process gridded dataset
println("Processing gridded dataset...")
Dataset(joinpath(ceres_data_dir, "ceres_gridded.nc"), "a") do ds
    # Handle net longwave
    if haskey(ds, gridded_lw_save_var)
        println("Updating existing gridded net longwave variable...")
        ds[gridded_lw_save_var][:] = gridded_net_lw
    else
        println("Creating new gridded net longwave variable...")
        # Get dimensions and attributes from original variable
        orig_var = ds[gridded_lw_var]
        defVar(ds, gridded_lw_save_var, eltype(gridded_net_lw), dimnames(orig_var),
               attrib = Dict(
                   "long_name" => "TOA Net Longwave Flux, Monthly Mean (positive downward)",
                   "units" => "W m-2",
                   "standard_name" => "toa_net_longwave_flux"
               ))
        ds[gridded_lw_save_var][:] = gridded_net_lw
    end
    
    # Handle net shortwave
    if haskey(ds, gridded_sw_save_var)
        println("Updating existing gridded net shortwave variable...")
        ds[gridded_sw_save_var][:] = gridded_net_sw
    else
        println("Creating new gridded net shortwave variable...")
        # Get dimensions and attributes from original variable
        orig_var = ds[gridded_sw_var]
        defVar(ds, gridded_sw_save_var, eltype(gridded_net_sw), dimnames(orig_var),
               attrib = Dict(
                   "long_name" => "TOA Net Shortwave Flux, Monthly Mean (positive downward)",
                   "units" => "W m-2",
                   "standard_name" => "toa_net_shortwave_flux"
               ))
        ds[gridded_sw_save_var][:] = gridded_net_sw
    end
end

# Process global dataset
println("Processing global dataset...")
Dataset(joinpath(ceres_data_dir, "ceres_global.nc"), "a") do ds
    # Handle net longwave
    if haskey(ds, global_lw_save_var)
        println("Updating existing global net longwave variable...")
        ds[global_lw_save_var][:] = global_net_lw
    else
        println("Creating new global net longwave variable...")
        # Get dimensions and attributes from original variable
        orig_var = ds[global_lw_var]
        defVar(ds, global_lw_save_var, eltype(global_net_lw), dimnames(orig_var),
               attrib = Dict(
                   "long_name" => "Global TOA Net Longwave Flux, Monthly Mean (positive downward)",
                   "units" => "W m-2",
                   "standard_name" => "global_toa_net_longwave_flux"
               ))
        ds[global_lw_save_var][:] = global_net_lw
    end
    
    # Handle net shortwave
    if haskey(ds, global_sw_save_var)
        println("Updating existing global net shortwave variable...")
        ds[global_sw_save_var][:] = global_net_sw
    else
        println("Creating new global net shortwave variable...")
        # Get dimensions and attributes from original variable
        orig_var = ds[global_sw_var]
        defVar(ds, global_sw_save_var, eltype(global_net_sw), dimnames(orig_var),
               attrib = Dict(
                   "long_name" => "Global TOA Net Shortwave Flux, Monthly Mean (positive downward)",
                   "units" => "W m-2",
                   "standard_name" => "global_toa_net_shortwave_flux"
               ))
        ds[global_sw_save_var][:] = global_net_sw
    end
end

println("CERES data cleaning completed successfully!")
println("Created/updated variables:")
println("  Gridded: $gridded_lw_save_var, $gridded_sw_save_var")
println("  Global: $global_lw_save_var, $global_sw_save_var")

