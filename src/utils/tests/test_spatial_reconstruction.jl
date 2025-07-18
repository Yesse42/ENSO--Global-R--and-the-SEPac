using Dates
include("../load_funcs.jl")
include("../constants.jl")

println("Testing reconstruct_spatial_arrays function...")

# Load some test data to get the variable indices and shapes
era5_vars = ["t", "z", "t2m", "msl"]
ceres_vars = ["gtoa_lw_all_mon", "gtoa_net_all_mon", "gtoa_sw_all_mon"]

# Load data 
enso_data, enso_coords = load_enso_data(time_period; lags=[-6, -3, 0, 3, 6], data_dir="../../../data/ENSO")
era5_data, era5_coords = load_era5_data(era5_vars, time_period; data_dir="../../../data/ERA5")
ceres_data, ceres_coords = load_ceres_data(ceres_vars, time_period; data_dir="../../../data/CERES")

# Prepare data for concatenation
all_arrays = []
all_names = []
time_indices = []

# Add ERA5 variables
for var in era5_vars
    push!(all_arrays, era5_data[var])
    push!(all_names, var)
    push!(time_indices, ndims(era5_data[var]))
end

# Add CERES variables  
for var in ceres_vars
    push!(all_arrays, ceres_data[var])
    push!(all_names, var)
    push!(time_indices, ndims(ceres_data[var]))
end

# Get the metadata from reshape_and_concatenate
combined_matrix, var_indices, orig_shapes = reshape_and_concatenate(all_arrays, all_names, time_indices)

println("Original combined matrix size: $(size(combined_matrix))")
println("Total spatial features: $(size(combined_matrix, 2))")

# Create a test spatial vector (e.g., random coefficients)
spatial_vector = randn(size(combined_matrix, 2))

println("Testing spatial reconstruction...")
reconstructed = reconstruct_spatial_arrays(spatial_vector, var_indices, orig_shapes)

println("Reconstructed arrays:")
for (var, array) in pairs(reconstructed)
    println("  $var: $(size(array))")
end

# Verify the reconstruction makes sense
println("\nVerifying reconstruction...")
for (i, var) in enumerate(all_names)
    orig_spatial_shape = size(all_arrays[i])[1:end-1]  # Remove time dimension
    reconstructed_shape = size(reconstructed[var])
    
    if orig_spatial_shape == reconstructed_shape
        println("  ✓ $var: spatial shape matches $(orig_spatial_shape)")
    else
        println("  ✗ $var: shape mismatch! Expected $(orig_spatial_shape), got $(reconstructed_shape)")
    end
end

println("\nTest completed successfully!")
