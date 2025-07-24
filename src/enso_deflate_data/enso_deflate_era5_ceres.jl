using NCDatasets, Dates, StatsBase, Plots, SplitApplyCombine, LinearAlgebra, Dictionaries
using CSV, DataFrames

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../pls_regressor/pls_functions.jl")
include("../utils/utilfuncs.jl")
include("../utils/constants.jl")

# Use time period from constants.jl
println("Analyzing period: $(time_period[1]) to $(time_period[2])")

# Define variables to deflate
era5_vars = ["t", "z", "t2m", "msl"]
ceres_vars = ["gtoa_lw_all_mon", "gtoa_net_all_mon", "gtoa_sw_all_mon"]
println("ERA5 variables to deflate: $(join(era5_vars, ", "))")
println("CERES variables to deflate: $(join(ceres_vars, ", "))")

# Define ENSO lags to use as predictors
enso_lags = [-6, -3, 0, 3, 6]
println("ENSO lags to use as predictors: $(join(enso_lags, ", "))")

# Load ENSO data with specified lags
println("Loading ENSO data...")
enso_data, enso_coords = load_enso_data(time_period; lags=enso_lags)

# Load ERA5 data 
println("Loading ERA5 data...")
era5_data, era5_coords = load_era5_data(era5_vars, time_period)

# Load CERES data
println("Loading CERES data...")
ceres_data, ceres_coords = load_ceres_data(ceres_vars, time_period)

# Get coordinate information
lat = era5_coords["latitude"]
lon = era5_coords["longitude"] 
if haskey(era5_coords, "pressure_level")
    press = era5_coords["pressure_level"]
else
    press = nothing
end

lat_size = length(lat)
lon_size = length(lon)
press_size = press === nothing ? 0 : length(press)

println("ERA5 spatial dimensions: lat=$lat_size, lon=$lon_size")
if press_size > 0
    println("Pressure levels: $press_size")
end

# Use common time from any dataset (they should all be the same after filtering)
common_time = era5_coords["time"]
println("Using $(length(common_time)) time points")

# Use the reshape_and_concatenate function for efficient processing
println("Using reshape_and_concatenate function...")

# Prepare data arrays and names for concatenation
all_arrays = []
all_names = []
time_indices = []

# Add ERA5 variables
for var in era5_vars
    push!(all_arrays, era5_data[var])
    push!(all_names, var)
    push!(time_indices, ndims(era5_data[var]))  # Time is last dimension
end

# Add CERES variables  
for var in ceres_vars
    push!(all_arrays, ceres_data[var])
    push!(all_names, var)
    push!(time_indices, ndims(ceres_data[var]))  # Time is last dimension
end

# Use the concatenation function to create the combined matrix
combined_matrix, var_indices, orig_shapes = reshape_and_concatenate(all_arrays, all_names, time_indices)
combined_matrix = Float32.(combined_matrix)  # Convert to Float32 for memory efficiency

println("Combined data dimensions: $(size(combined_matrix))")

# Extract ENSO predictor matrix (time x lags)
enso_predictors = hcat([enso_data["oni_lag_$(lag)"] for lag in enso_lags]...)
println("ENSO predictors shape: $(size(enso_predictors))")

# Prepare time information for detrending and deseasonalizing
months = month.(common_time)
float_times = calc_float_time.(common_time)
month_groups = groupfind(months)

println("Detrending and deseasonalizing data...")

# Check for missing data
has_missing_data = any(ismissing, combined_matrix)
if has_missing_data
    println("Warning: Combined matrix has missing data")
end

# Detrend and deseasonalize all data
detrend_and_deseasonalize_precalculated_groups!.(eachcol(combined_matrix), Ref(float_times), Ref(month_groups))

# Detrend and deseasonalize ENSO predictors
detrend_and_deseasonalize_precalculated_groups!.(eachcol(enso_predictors), Ref(float_times), Ref(month_groups))

# Prepare data for PLS regression
# X = Combined ERA5 + CERES data (predictands), Y = ENSO data (predictors) 
# Note: This is opposite to typical notation since we want to deflate using ENSO
Y = combined_matrix  # What we want to deflate
X = Float32.(enso_predictors)  # What we use to deflate

println("PLS regression setup:")
println("  X (data to deflate): $(size(X))")
println("  Y (ENSO predictors): $(size(Y))")

# Run PLS regression
n_components = min(size(X, 2), 5)  # Use number of ENSO lags or 5, whichever is smaller
println("Running PLS regression with $n_components components...")

my_pls = make_pls_regressor(X, Y, n_components; make_copies = NoCopy())

println("PLS regression completed successfully!")

# Extract the deflated data
deflated_data = my_pls.Y
println("Deflated data dimensions: $(size(deflated_data))")

# Use the deconcatenate_and_reshape function to restore original shapes
println("Using deconcatenate_and_reshape function...")
restored_arrays = deconcatenate_and_reshape(deflated_data, var_indices, orig_shapes, time_indices)

println("Unpacking deflated data...")
deflated_era5_vars = Dictionary{String, Array}()
deflated_ceres_vars = Dictionary{String, Array}()

# Separate ERA5 and CERES variables
for (i, var) in enumerate(all_names)
    if var in era5_vars
        set!(deflated_era5_vars, var, restored_arrays[var])
        println("  ERA5 $var: $(size(restored_arrays[var]))")
    elseif var in ceres_vars
        set!(deflated_ceres_vars, var, restored_arrays[var])
        println("  CERES $var: $(size(restored_arrays[var]))")
    end
end

# Save deflated data to NetCDF
output_dir = "../../data/ENSO_Deflated"
output_file = joinpath(output_dir, "era5_ceres_enso_deflated.nc")
println("Saving deflated data to $output_file")

ds_out = NCDataset(output_file, "c")

# Define dimensions
defDim(ds_out, "time", length(common_time))
defDim(ds_out, "latitude", lat_size)
defDim(ds_out, "longitude", lon_size)
if press_size > 0
    defDim(ds_out, "pressure_level", press_size)
end

# Define coordinate variables
defVar(ds_out, "time", common_time, ("time",))
defVar(ds_out, "latitude", lat, ("latitude",))
defVar(ds_out, "longitude", lon, ("longitude",))
if press_size > 0
    defVar(ds_out, "pressure_level", press, ("pressure_level",))
end

# Save deflated ERA5 variables
for (var, data) in pairs(deflated_era5_vars)
    if ndims(data) == 3  # Single level: (lon, lat, time)
        defVar(ds_out, "deflated_$(var)", data, ("longitude", "latitude", "time"))
    elseif ndims(data) == 4  # Pressure level: (lon, lat, pressure, time)
        defVar(ds_out, "deflated_$(var)", data, ("longitude", "latitude", "pressure_level", "time"))
    end
end

# Save deflated CERES variables
for (var, data) in pairs(deflated_ceres_vars)
    if ndims(data) == 1  # Global: (time,)
        defVar(ds_out, "deflated_$(var)", data, ("time",))
    end
end

# Save original ENSO predictors for reference
for (i, lag) in enumerate(enso_lags)
    defVar(ds_out, "oni_lag_$(lag)", enso_predictors[:, i], ("time",))
end

# Add metadata
ds_out.attrib["title"] = "ENSO-deflated ERA5 atmospheric and CERES radiation data"
ds_out.attrib["description"] = "ERA5 and CERES data deflated using PLS regression with ONI lags as predictors"
ds_out.attrib["era5_variables"] = join(era5_vars, ", ")
ds_out.attrib["ceres_variables"] = join(ceres_vars, ", ")
ds_out.attrib["enso_lags"] = join(enso_lags, ", ")
ds_out.attrib["n_components"] = n_components
ds_out.attrib["time_period"] = "$(time_period[1]) to $(time_period[2])"
ds_out.attrib["creation_date"] = string(now())

close(ds_out)

println("Successfully saved deflated data to $output_file")

println("ENSO deflation analysis complete!")
