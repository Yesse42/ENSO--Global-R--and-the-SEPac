# Test the reshape and concatenation functions

using Dates
include("../load_funcs.jl")

println("Testing reshape_and_concatenate and deconcatenate_and_reshape functions...")

# Define time period
time_period = (Date(2000, 3), Date(2010, 4))

# Load some sample data
era5_data, era5_coords = load_era5_data(["t2m", "msl"], time_period; data_dir="../../../data/ERA5")

# Create arrays with different shapes but same time dimension
# t2m: (lon, lat, time) - time is dimension 3
# Let's also create a simple 2D array for testing
simple_array = randn(120, length(era5_coords["time"]))  # (spatial, time) - time is dimension 2

arrays = [era5_data["t2m"], era5_data["msl"], simple_array]
array_names = ["t2m", "msl", "simple_var"]
time_indices = [3, 3, 2]  # time dimension for each array

println("\nOriginal array shapes:")
for (name, arr) in zip(array_names, arrays)
    println("  $name: $(size(arr))")
end

# Test reshape and concatenate
println("\nTesting reshape_and_concatenate...")
concat_arr, var_indices, orig_shapes = reshape_and_concatenate(arrays, array_names, time_indices)

println("Concatenated array shape: $(size(concat_arr))")
println("Variable indices:")
for (name, indices) in pairs(var_indices)
    println("  $name: $(indices)")
end
println("Original shapes:")
for (name, shape) in pairs(orig_shapes)
    println("  $name: $shape")
end

# Test deconcatenate and reshape
println("\nTesting deconcatenate_and_reshape...")
recovered_arrays = deconcatenate_and_reshape(concat_arr, var_indices, orig_shapes, time_indices)

println("Recovered array shapes:")
for (name, arr) in pairs(recovered_arrays)
    println("  $name: $(size(arr))")
end

# Verify that recovered arrays match original arrays
println("\nVerifying accuracy of recovery...")
for name in array_names
    if name in keys(era5_data)
        original = era5_data[name]
    else
        original = simple_array
    end
    recovered = recovered_arrays[name]
    
    if size(original) == size(recovered) && isapprox(original, recovered)
        println("  ✓ $name: Successfully recovered")
    else
        println("  ✗ $name: Recovery failed - size mismatch or values differ")
        println("    Original: $(size(original)), Recovered: $(size(recovered))")
    end
end

# Example with different time dimensions
println("\n" * "="^50)
println("Testing with arrays having different time dimensions...")

# Create test arrays with time in different positions
arr_time_last = randn(10, 15, 25)    # time in position 3
arr_time_first = randn(25, 8, 12)    # time in position 1  
arr_time_middle = randn(5, 25, 7)    # time in position 2

test_arrays = [arr_time_last, arr_time_first, arr_time_middle]
test_names = ["time_last", "time_first", "time_middle"]
test_time_indices = [3, 1, 2]

println("Test array shapes:")
for (name, arr) in zip(test_names, test_arrays)
    println("  $name: $(size(arr))")
end

# Concatenate
test_concat, test_var_idx, test_orig_shapes = reshape_and_concatenate(test_arrays, test_names, test_time_indices)
println("Test concatenated shape: $(size(test_concat))")

# Recover
test_recovered = deconcatenate_and_reshape(test_concat, test_var_idx, test_orig_shapes, test_time_indices)

println("Test recovery verification:")
for (i, name) in enumerate(test_names)
    original = test_arrays[i]
    recovered = test_recovered[name]
    
    if size(original) == size(recovered) && isapprox(original, recovered)
        println("  ✓ $name: Successfully recovered")
    else
        println("  ✗ $name: Recovery failed")
    end
end

println("\nAll tests completed!")
