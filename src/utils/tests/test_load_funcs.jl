# Example usage of the data loading functions in load_funcs.jl

using Dates
include("../load_funcs.jl")

# Define time period for data loading
time_period = (Date(2000, 3), Date(2022, 4))

# Example 1: Load ERA5 data
println("Loading ERA5 data...")
era5_variables = ["t2m", "msl", "z", "t"]  # Mix of single level and pressure level variables
era5_data, era5_coords = load_era5_data(era5_variables, time_period; data_dir="../../../data/ERA5")

println("Loaded ERA5 variables:")
for (var, data) in pairs(era5_data)
    println("  $var: $(size(data))")
end
println("ERA5 coordinates:")
for (coord_name, coord_data) in pairs(era5_coords)
    if coord_name == "time" || coord_name == "pressure_time"
        println("  $coord_name: $(length(coord_data)) time points")
    else
        println("  $coord_name: $(length(coord_data)) points")
    end
end

# Example 2: Load CERES data
println("\nLoading CERES data...")
ceres_variables = ["gtoa_lw_all_mon", "gtoa_net_all_mon", "global_net_sw"]
ceres_data, ceres_coords = load_ceres_data(ceres_variables, time_period; data_dir="../../../data/CERES")

println("Loaded CERES variables:")
for (var, data) in pairs(ceres_data)
    println("  $var: $(size(data))")
end
println("CERES coordinates:")
for (coord_name, coord_data) in pairs(ceres_coords)
    if coord_name == "time"
        println("  $coord_name: $(length(coord_data)) time points")
    else
        println("  $coord_name: $(length(coord_data)) points")
    end
end

# Example 3: Load ENSO data (single lag - default)
println("\nLoading ENSO data (default - lag 0)...")
enso_data, enso_coords = load_enso_data(time_period; lags=[0], data_dir="../../../data/ENSO")
println("Loaded ENSO variables:")
for (var, data) in pairs(enso_data)
    println("  $var: $(length(data)) time points")
end

# Example 3b: Load ENSO data (multiple specific lags)
println("\nLoading ENSO data (multiple specific lags)...")
enso_multi_data, enso_multi_coords = load_enso_data(time_period; lags=[-2, -1, 0, 1, 2], data_dir="../../../data/ENSO")
println("Loaded ENSO variables with multiple lags:")
for (var, data) in pairs(enso_multi_data)
    println("  $var: $(length(data)) time points")
end

# Example 3c: Load all available ENSO lags
println("\nLoading all available ENSO lags...")
enso_all_data, enso_all_coords = load_enso_data(time_period; lags=nothing, data_dir="../../../data/ENSO")
println("Loaded all ENSO lag variables:")
for (var, data) in pairs(enso_all_data)
    println("  $var: $(length(data)) time points")
end

println("ENSO coordinates:")
for (coord_name, coord_data) in pairs(enso_coords)
    println("  $coord_name: $(length(coord_data)) time points from $(coord_data[1]) to $(coord_data[end])")
end

# Example 4: Access specific variables
println("\nAccessing specific variables...")
if haskey(era5_data, "t2m")
    t2m_data = era5_data["t2m"]
    println("Surface temperature (t2m): $(size(t2m_data)) - dimensions (lon, lat, time)")
end

if haskey(era5_data, "z")
    z_data = era5_data["z"]
    println("Geopotential (z): $(size(z_data)) - dimensions (lon, lat, pressure, time)")
end

if haskey(ceres_data, "gtoa_net_all_mon")
    ceres_net = ceres_data["gtoa_net_all_mon"]
    println("CERES net radiation: $(size(ceres_net))")
end
