"""
Generate lagged, detrended, and deseasonalized time series of global CERES radiation data.
Based on the logic from generate_regional_time_series.jl but for global CERES data.
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using Statistics, DataFrames, Dictionaries, JLD2, CSV, Dates

# Define radiation variables and time period
rad_variables = ["gtoa_net_all_mon", "gtoa_net_lw_mon", "gtoa_net_sw_mon"]
date_range = (Date(2000, 3), Date(2024, 3, 31))

# Load global radiation time series
println("Loading global CERES radiation data...")
global_rad_data, global_coords = load_new_ceres_data(rad_variables, date_range)

# Round time coordinates to monthly resolution
global_coords["time"] = round.(global_coords["time"], Dates.Month(1), RoundDown)
float_times = calc_float_time.(global_coords["time"])
months = month.(global_coords["time"])

println("Loaded data for $(length(global_coords["time"])) time steps")
println("Variables: $(join(rad_variables, ", "))")

# Define lags to generate (in months)
lags = -24:24

# Create output directory
output_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/lagged"
mkpath(output_dir)

# First, process all variables and collect data
println("\nProcessing each radiation variable...")

df = DataFrame(date = global_coords["time"])

for var_name in rad_variables
    println("Processing variable: $var_name")
    
    # Get the time series data
    raw_data = global_rad_data[var_name]
    
    # Detrend and deseasonalize the data
    println("  Detrending and deseasonalizing...")
    detrend_and_deseasonalize!(raw_data, float_times, months)
    processed_data = raw_data
    
    # Generate lagged versions
    println("  Generating lagged versions...")
    
    for lag in lags
        lagged_data = time_lag(processed_data, lag)
        col_name = Symbol("$(var_name)_lag_$(lag)")
        df[!, col_name] = lagged_data
    end
end

CSV.write(joinpath(output_dir, "global_ceres_lagged_detrended_deseasonalized.csv"), df)
