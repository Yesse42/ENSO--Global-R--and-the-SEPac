# Utils Tests

This directory contains test scripts for the utility functions in the parent `utils` directory.

## Test Files

### test_load_funcs.jl
- **Purpose**: Example usage and testing of data loading functions
- **Coverage**: Tests ERA5, CERES, and ENSO data loading functions
- **Usage**: `julia test_load_funcs.jl`
- **Description**: Demonstrates how to load different types of data and provides examples of the output formats

### test_reshape_functions.jl  
- **Purpose**: Tests the reshape and concatenation utility functions
- **Coverage**: Tests `reshape_and_concatenate` and `deconcatenate_and_reshape` functions
- **Usage**: `julia test_reshape_functions.jl`
- **Description**: Validates that data can be properly reshaped for ML models and then reconstructed back to original formats

### test_spatial_reconstruction.jl
- **Purpose**: Tests the spatial array reconstruction functionality
- **Coverage**: Tests `reconstruct_spatial_arrays` function
- **Usage**: `julia test_spatial_reconstruction.jl`
- **Description**: Verifies that spatial vectors (e.g., from PLS regression) can be properly reconstructed into their original spatial dimensions

## Running Tests

From this directory:
```bash
julia test_load_funcs.jl
julia test_reshape_functions.jl  
julia test_spatial_reconstruction.jl
```

Or run all tests:
```bash
for test in test_*.jl; do echo "Running $test..."; julia "$test"; echo ""; done
```

## Dependencies

All test files depend on the utility functions in the parent directory:
- `../load_funcs.jl` - Main data loading and manipulation functions
- `../constants.jl` - Time period definitions (used by some tests)

Make sure the data directories are properly set up relative to the repository root:
- `../../data/ERA5/`
- `../../data/CERES/`
- `../../data/ENSO/`
