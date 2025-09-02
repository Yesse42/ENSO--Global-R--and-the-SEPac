using JLD2, Statistics, StatsBase, Dates, SplitApplyCombine, CSV, DataFrames, NCDatasets
using Plots

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")

"""
This script generates time series analogous to the SEPac SST index for surface fluxes.
It loads the mask and weights from the SEPac SST analysis and applies them to calculate
weighted averages for net surface SW flux, net surface LW flux, latent heat flux, 
and sensible heat flux over the same domain.
"""

# Load the mask and weights from the SEPac SST analysis
println("Loading SEPac SST mask and weights...")
mask_data = jldopen("../../data/SEPac_SST/sepac_sst_mask_and_weights.jld2", "r")
area_weights = mask_data["area_weights"]
lat = mask_data["latitude"]
lon = mask_data["longitude"]
close(mask_data)

# Define the flux variables we want to extract
# Based on ERA5 flux file inspection:
# - slhf: Surface latent heat net flux
# - sshf: Surface sensible heat net flux  
# - ssr: Surface net short-wave (solar) radiation
# - str: Surface net long-wave (thermal) radiation
flux_variables = ["slhf", "sshf", "ssr", "str"]
flux_descriptions = [
    "Surface Latent Heat Net Flux",
    "Surface Sensible Heat Net Flux", 
    "Surface Net Short-wave Radiation",
    "Surface Net Long-wave Radiation"
]

# Time period for analysis
idx_time_period = (Date(0), Date(10000000, 12, 31))

println("Loading ERA5 flux data...")

# Load data from fluxes_1.nc (contains all our needed variables)
flux_file = "../../data/ERA5/fluxes_1.nc"

# Initialize variables at the correct scope
flux_data = Dict()
flux_times = nothing

NCDatasets.Dataset(flux_file, "r") do ds
    # Get time coordinates and filter by time period
    era5_times = ds["valid_time"][:]
    time_mask = in_time_period.(era5_times, Ref(idx_time_period))
    global flux_times = era5_times[time_mask]
    
    # Verify that lat/lon match what we expect from the mask
    file_lat = ds["latitude"][:]
    file_lon = ds["longitude"][:]
    
    if !isapprox(file_lat, lat, atol=1e-6) || !isapprox(file_lon, lon, atol=1e-6)
        error("Latitude/longitude coordinates in flux file don't match SEPac mask coordinates")
    end
    
    # Load each flux variable
    for var in flux_variables
        if haskey(ds, var)
            println("  Loading $var...")
            raw_data = ds[var][:, :, time_mask]
            # Convert to Float32 and handle any missing values
            flux_data[var] = Array{Float32}(raw_data)
            # Replace any missing values with 0
            flux_data[var][ismissing.(flux_data[var])] .= 0.0f0
        else
            error("Variable $var not found in $flux_file")
        end
    end
end

println("Calculating weighted flux time series...")

# Calculate weighted averages for each flux variable
flux_time_series = Dict()
total_weight = sum(area_weights)

for (i, var) in enumerate(flux_variables)
    println("  Processing $(flux_descriptions[i])...")
    
    # Calculate weighted average time series
    var_data = flux_data[var]
    weighted_sum = vec(sum(var_data .* area_weights, dims=(1,2)))
    display(size(var_data))
    display(size(area_weights))
    flux_time_series[var] = weighted_sum ./ total_weight
end

# Create output DataFrame
println("Creating output DataFrame...")

# Debug: Check dimensions
println("  flux_times length: $(length(flux_times))")
for (i, var) in enumerate(flux_variables)
    println("  $(var) time series length: $(length(flux_time_series[var]))")
end

df = DataFrame(Date = flux_times)

# Add each flux time series to the DataFrame with descriptive names
column_names = [
    "SEPac_Latent_Heat_Flux",
    "SEPac_Sensible_Heat_Flux", 
    "SEPac_Net_SW_Radiation",
    "SEPac_Net_LW_Radiation"
]

for (i, var) in enumerate(flux_variables)
    df[!, Symbol(column_names[i])] = flux_time_series[var]
end

# Skip time lag generation as requested

# Plot each time series
println("\nPlotting flux time series...")
for (i, var) in enumerate(flux_variables)
    desc = flux_descriptions[i]
    col_name = column_names[i]
    values = flux_time_series[var]
    
    # Create a plot for this time series
    p = plot(flux_times, values, 
             linewidth=1.5, 
             color=:blue,
             xlabel="Date", 
             ylabel="Flux (W/m²)",
             title="SEPac $desc Time Series",
             label = "",
             grid=true,
             gridalpha=0.3,
             size=(800, 400))

    # Add statistics annotation
    stats_text = "Mean: $(round(mean(values), digits=1))\nStd: $(round(std(values), digits=1))"
    annotate!(p, (0.02, 0.98), text(stats_text, :left, :top, 8, :black), 
              subplot=1, coordinate_system=:relative)

    display(p)
end

# Display summary information
println("\nSummary of generated flux time series:")
println("Time period: $(first(flux_times)) to $(last(flux_times))")
println("Number of time points: $(length(flux_times))")
println("Variables generated:")
for (i, var) in enumerate(flux_variables)
    desc = flux_descriptions[i]
    col_name = column_names[i]
    values = flux_time_series[var]
    println("  $desc ($col_name):")
    println("    Mean: $(round(mean(values), digits=3))")
    println("    Std:  $(round(std(values), digits=3))")
    println("    Min:  $(round(minimum(values), digits=3))")
    println("    Max:  $(round(maximum(values), digits=3))")
end

# Save the results
output_file = "../../data/SEPac_SST/sepac_flux_time_series.csv"
println("\nSaving results to $output_file...")
CSV.write(output_file, df)

println("Done! Flux time series saved successfully.")
