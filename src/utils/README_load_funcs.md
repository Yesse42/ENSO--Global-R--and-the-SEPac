# Data Loading Functions for ENSO and Global R Project

This document describes the data loading functions in `load_fu# Load ENSO data (single lag)
enso_data, enso_coords = load_enso_data(time_period; lags=[0])
println("ENSO index: ", size(enso_data["oni_lag_0"]))  # (265,)

# Load ENSO data (multiple lags)
enso_multi_data, enso_multi_coords = load_enso_data(time_period; lags=[-1, 0, 1])
println("Multiple ENSO lags loaded: ", length(enso_multi_data))  # 3

# Load all available ENSO lags
enso_all_data, enso_all_coords = load_enso_data(time_period; lags=nothing)
println("All ENSO lags: ", length(enso_all_data))  # All available lagst provide a unified interface for loading ERA5, CERES, and ENSO data.

## Overview

The loading functions return raw data arrays without any reshaping, making them flexible for different analysis workflows. All functions return:
1. A `Dictionary` mapping variable names to data arrays
2. A `Dictionary` containing coordinate information

## Functions

### `load_era5_data(variables, time_period; kwargs...)`

Loads ERA5 reanalysis data for specified variables and time period.

**Arguments:**
- `variables`: Vector of variable names (e.g., `["t2m", "msl", "z", "t"]`)
- `time_period`: Tuple of `(start_date, end_date)` as Date objects

**Returns:**
- Data dictionary with variable names as keys
- Coordinates dictionary with `"latitude"`, `"longitude"`, `"time"`, and optionally `"pressure_level"` and `"pressure_time"`

**Data dimensions:**
- Single level variables: `(longitude, latitude, time)`
- Pressure level variables: `(longitude, latitude, pressure, time)`

**Common variables:**
- Single level: `"t2m"`, `"msl"`, `"u10"`, `"v10"`, `"sp"`, `"sst"`, etc.
- Pressure level: `"z"`, `"t"`, `"u"`, `"v"`, `"w"`, `"q"`, `"r"`

### `load_ceres_data(variables, time_period; kwargs...)`

Loads CERES radiative flux data.

**Arguments:**
- `variables`: Vector of variable names (e.g., `["gtoa_net_all_mon", "global_net_sw"]`)
- `time_period`: Tuple of `(start_date, end_date)` as Date objects

**Returns:**
- Data dictionary with variable names as keys
- Coordinates dictionary with `"time"` and optionally `"latitude"`, `"longitude"`

**Data dimensions:**
- Global variables: `(time,)` - 1D time series
- Gridded variables: `(longitude, latitude, time)`

**Common variables:**
- Global: `"gtoa_net_all_mon"`, `"gtoa_lw_all_mon"`, `"global_net_sw"`
- Gridded: `"toa_net_all_mon"`, `"toa_lw_all_mon"`, `"gridded_net_sw"`

### `load_enso_data(time_period; kwargs...)`

Loads ENSO index data from CSV file.

**Arguments:**
- `time_period`: Tuple of `(start_date, end_date)` as Date objects
- `lags`: Vector of lag values (e.g., `[-2, -1, 0, 1, 2]`) or `nothing` for all lags

**Returns:**
- Data dictionary with ONI lag column names as keys
- Coordinates dictionary with `"time"`

**Data dimensions:**
- ENSO indices: `(time,)` - 1D time series for each lag

**Examples:**
- Single lag: `load_enso_data(time_period; lags=[0])`
- Multiple lags: `load_enso_data(time_period; lags=[-1, 0, 1])`
- All lags: `load_enso_data(time_period; lags=nothing)`

## Convenience Functions for Array Manipulation

### `reshape_and_concatenate(arrays, array_names, time_indices)`

Reshapes arrays to `(time, spatial_dims)` and concatenates them along the spatial dimension.

**Arguments:**
- `arrays`: Vector of arrays to reshape and concatenate
- `array_names`: Vector of names for each array
- `time_indices`: Vector indicating which dimension is time for each array

**Returns:**
- `concatenated_array`: Matrix of size `(time, total_spatial_dims)`
- `variable_indices`: Dictionary mapping variable names to column indices
- `original_shapes`: Dictionary mapping variable names to original dimensions

### `deconcatenate_and_reshape(concatenated_array, variable_indices, original_shapes, time_indices)`

Reverses the `reshape_and_concatenate` operation to recover original arrays.

**Arguments:**
- `concatenated_array`: Output from `reshape_and_concatenate`
- `variable_indices`: Dictionary from `reshape_and_concatenate`
- `original_shapes`: Dictionary from `reshape_and_concatenate`
- `time_indices`: Vector indicating original time dimensions

**Returns:**
- Dictionary mapping variable names to their original array shapes

## Usage Examples

```julia
using Dates
include("load_funcs.jl")

# Define time period
time_period = (Date(2000, 3), Date(2022, 4))

# Load ERA5 data
era5_data, era5_coords = load_era5_data(["t2m", "msl", "z"], time_period)
println("Surface temperature: ", size(era5_data["t2m"]))  # (1440, 721, 265)

# Load CERES data
ceres_data, ceres_coords = load_ceres_data(["gtoa_net_all_mon"], time_period)
println("CERES net radiation: ", size(ceres_data["gtoa_net_all_mon"]))  # (265,)

# Load ENSO data (single lag)
enso_data, enso_coords = load_enso_data(time_period)
println("ENSO index: ", size(enso_data["oni_lag_0"]))  # (265,)

# Load ENSO data (multiple lags)
enso_multi_data, enso_multi_coords = load_enso_data(time_period; index_columns=["oni_lag_-1", "oni_lag_0", "oni_lag_1"])
println("Multiple ENSO lags loaded: ", length(enso_multi_data))  # 3

# Load all available ENSO lags
enso_all_data, enso_all_coords = load_all_enso_lags(time_period)
println("All ENSO lags: ", length(enso_all_data))  # All available lags

# Access coordinates
lat = era5_coords["latitude"]
lon = era5_coords["longitude"]
time = era5_coords["time"]

# Example of using reshape and concatenate functions
arrays = [era5_data["t2m"], era5_data["msl"]]
names = ["t2m", "msl"]
time_indices = [3, 3]  # time is 3rd dimension for both

concat_data, var_indices, orig_shapes = reshape_and_concatenate(arrays, names, time_indices)
println("Concatenated shape: ", size(concat_data))  # (time, total_spatial_points)

# Recover original arrays with time dimension
recovered = deconcatenate_and_reshape(concat_data, var_indices, orig_shapes, time_indices)
println("Recovered t2m shape: ", size(recovered["t2m"]))  # Original shape

# For spatial-only analysis (e.g., PLS coefficients, patterns without time)
spatial_vector = randn(Float32, size(concat_data, 2))  # Example spatial vector
spatial_arrays = reconstruct_spatial_arrays(spatial_vector, var_indices, orig_shapes)
println("Spatial t2m shape: ", size(spatial_arrays["t2m"]))  # Spatial shape only
```

## File Paths

The functions use relative paths from the `src/utils/` directory:
- ERA5 data: `../../data/ERA5/`
- CERES data: `../../data/CERES/`
- ENSO data: `../../data/ENSO/`

These can be overridden using the `data_dir` parameter in each function.

### reconstruct_spatial_arrays

Reconstructs spatial arrays from a flattened spatial vector using variable indices and original shapes. This function is useful for working with spatial patterns derived from PLS regression or other analyses that produce spatial-only data (without time dimension).

```julia
# Get spatial patterns from PLS or other analysis
spatial_vector = [1.0, 2.0, 3.0, ...]  # Length matches total spatial points

# Reconstruct individual variable arrays
spatial_arrays = reconstruct_spatial_arrays(spatial_vector, var_indices, orig_shapes)

# Access specific variables
temp_pattern = spatial_arrays["t"]      # Shape: (1440, 721, 4) for pressure levels
t2m_pattern = spatial_arrays["t2m"]     # Shape: (1440, 721) for single level
ceres_sw = spatial_arrays["toa_sw_all_mon"]  # Scalar for global CERES data
```

**Parameters:**
- `spatial_vector`: Vector with length equal to total spatial features
- `var_indices`: Dictionary mapping variable names to their indices in the concatenated array
- `orig_shapes`: Dictionary mapping variable names to their original array shapes

**Returns:** Dictionary mapping variable names to their reconstructed spatial arrays (without time dimension).

## Notes

- All data is returned in its original NetCDF/CSV format without reshaping
- Missing values are preserved as they appear in the original files
- Time filtering is applied automatically based on the `time_period` argument
- Single level and pressure level ERA5 data are handled automatically
- The functions automatically determine whether variables are single level or pressure level based on common variable lists
- ENSO data supports loading multiple lag columns simultaneously
- The reshape/concatenate functions are useful for preparing data for machine learning models that expect 2D input matrices
- The `reconstruct_spatial_arrays` function is essential for analyzing spatial patterns from PLS regression or other methods that produce spatial-only vectors
- All functions use `Dictionaries.jl` for efficient key-value mapping
