#Contains functions to load in ERA5 data, CERES data, ENSO data, and any other data that gets loaded in frequently

using NCDatasets, Dates, CSV, DataFrames, Dictionaries

# Helper function to find a variable in multiple datasets
function load_era5_variable(varname, idx_tuple, datasets; throw_error_on_missing = true)
    for ds in datasets
        if haskey(ds, varname)
            outarr =  ds[varname][idx_tuple...]
            if !any(ismissing(el) for el in outarr)
                return Array{Float32}(outarr)
            else
                @warn "Variable $varname contains missing values in dataset"
                return outarr
            end
        end
    end
    throw_error_on_missing && error("Variable $varname not found in any of the datasets")
    return nothing
end

# Helper function to check if a time is within a given period
in_time_period(time, time_period) = time_period[1] <= time < time_period[2]

"""
    load_era5_data(variables, time_period; data_dir="../../data/ERA5", single_level_files=["single_levels_0.nc", "single_levels_1.nc", "single_levels_2.nc"], pressure_level_file="pressure_levels.nc")

Load ERA5 data for specified variables and time period.

# Arguments
- `variables`: Vector of variable names to load (e.g., ["t2m", "msl", "z", "t"])
- `time_period`: Tuple of (start_date, end_date) as Date objects
- `data_dir`: Directory containing ERA5 NetCDF files
- `single_level_files`: Vector of single level NetCDF filenames
- `pressure_level_file`: Pressure level NetCDF filename

# Returns
Dictionary with variable names as keys and raw arrays as values.
Single level variables have dimensions (lon, lat, time).
Pressure level variables have dimensions (lon, lat, pressure, time).
Also returns separate coordinate arrays (lat, lon, pressure if applicable, time).
"""
function load_era5_data(variables, time_period; 
                       data_dir="../../data/ERA5", 
                       single_level_files=["single_levels_0.nc", "single_levels_1.nc", "single_levels_2.nc"], 
                       pressure_level_file="pressure_levels.nc")
    
    # Open datasets
    single_level_paths = [joinpath(data_dir, fname) for fname in single_level_files]
    single_level_dss = [Dataset(path, "r") for path in single_level_paths]
    pressure_level_ds = Dataset(joinpath(data_dir, pressure_level_file), "r")
    
    loaded_data = Dictionary()
    coords = Dictionary()
    pressure_level_loaded = false
    
    try
        # Get time coordinates and filter by time period for single level
        single_level_times = single_level_dss[1]["valid_time"][:]
        isvalid_single_level = in_time_period.(single_level_times, Ref(time_period))
        filtered_single_times = single_level_times[isvalid_single_level]
        single_level_idxs = (:, :, isvalid_single_level)
        
        # Get time coordinates and filter by time period for pressure level
        pressure_level_times = pressure_level_ds["valid_time"][:]
        isvalid_pressure_level = in_time_period.(pressure_level_times, Ref(time_period))
        filtered_pressure_times = pressure_level_times[isvalid_pressure_level]
        pressure_level_idxs = (:, :, :, isvalid_pressure_level)
        
        # Get coordinate information from pressure level dataset (has all coords)
        lat = pressure_level_ds["latitude"][:]
        lon = pressure_level_ds["longitude"][:]
        pressure = pressure_level_ds["pressure_level"][:]
        
        # Store basic coordinates
        set!(coords, "latitude", lat)
        set!(coords, "longitude", lon)
        set!(coords, "time", filtered_single_times) # Use single level time as default
        
        # Get available variables from each dataset type
        single_level_available_vars = Set()
        for ds in single_level_dss
            union!(single_level_available_vars, keys(ds))
        end
        pressure_level_available_vars = Set(keys(pressure_level_ds))
        
        for var in variables
            # Check if variable exists in pressure level dataset
            if var in pressure_level_available_vars
                # Load pressure level variable - no reshaping
                data = load_era5_variable(var, pressure_level_idxs, [pressure_level_ds])
                if data !== nothing
                    set!(loaded_data, var, data)
                    if !pressure_level_loaded
                        set!(coords, "pressure_level", pressure)
                        set!(coords, "pressure_time", filtered_pressure_times)
                        pressure_level_loaded = true
                    end
                end
            # Check if variable exists in single level datasets
            elseif var in single_level_available_vars
                # Load single level variable - no reshaping
                data = load_era5_variable(var, single_level_idxs, single_level_dss)
                if data !== nothing
                    set!(loaded_data, var, data)
                end
            else
                @warn "Variable $var not found in any ERA5 datasets"
            end
        end
        
        return loaded_data, coords
        
    finally
        # Clean up datasets
        close.(single_level_dss)
        close(pressure_level_ds)
    end
end

"""
    load_ceres_data(variables, time_period; data_dir="../../data/CERES", global_file="ceres_global.nc", gridded_file="ceres_gridded.nc")

Load CERES radiative flux data for specified variables and time period.

# Arguments
- `variables`: Vector of variable names to load (e.g., ["gtoa_lw_all_mon", "gtoa_net_all_mon", "global_net_sw"])
- `time_period`: Tuple of (start_date, end_date) as Date objects  
- `data_dir`: Directory containing CERES NetCDF files
- `global_file`: Global mean CERES data filename
- `gridded_file`: Gridded CERES data filename

# Returns
Dictionary with variable names as keys and raw arrays as values.
Global variables are 1D time series.
Gridded variables have dimensions (lon, lat, time).
Also returns coordinate information.
"""
function load_ceres_data(variables, time_period; 
                        data_dir="../../data/CERES", 
                        global_file="ceres_global.nc", 
                        gridded_file="ceres_gridded.nc", lw_sign_flip = false)
    
    global_path = joinpath(data_dir, global_file)
    gridded_path = joinpath(data_dir, gridded_file)
    
    loaded_data = Dictionary()
    coords = Dictionary()
    
    # Get available variables from each file type
    global_available_vars = Set()
    gridded_available_vars = Set()
    
    # Check what variables are available in global file
    if isfile(global_path)
        ds_global = Dataset(global_path, "r")
        try
            union!(global_available_vars, keys(ds_global))
        finally
            close(ds_global)
        end
    end
    
    # Check what variables are available in gridded file
    if isfile(gridded_path)
        ds_gridded = Dataset(gridded_path, "r")
        try
            union!(gridded_available_vars, keys(ds_gridded))
        finally
            close(ds_gridded)
        end
    end
    
    # Now load variables based on availability
    # Try to load from global file first for variables that exist there
    if isfile(global_path) && !isempty(global_available_vars ∩ Set(variables))
        ds_global = Dataset(global_path, "r")
        try
            # Get time and filter by period
            ceres_time = ds_global["time"][:]
            isvalid_time = in_time_period.(ceres_time, Ref(time_period))
            filtered_time = ceres_time[isvalid_time]
            time_idxs = (isvalid_time,)
            
            # Store time coordinate
            set!(coords, "time", filtered_time)
            
            # Load requested variables that exist in global file
            for var in variables
                if var in global_available_vars
                    data = Array{Float32}(ds_global[var][time_idxs...])
                    set!(loaded_data, var, data)
                end
            end
            
        finally
            close(ds_global)
        end
    end
    
    # Try to load from gridded file for any remaining variables
    if isfile(gridded_path)
        ds_gridded = Dataset(gridded_path, "r")
        try
            # Get coordinates if not already set
            if !haskey(coords, "time")
                ceres_time = ds_gridded["time"][:]
                isvalid_time = in_time_period.(ceres_time, Ref(time_period))
                filtered_time = ceres_time[isvalid_time]
                set!(coords, "time", filtered_time)
            else
                # Use existing time filtering
                isvalid_time = in_time_period.(ds_gridded["time"][:], Ref(time_period))
            end
            
            time_idxs = (:, :, isvalid_time)
            
            # Add spatial coordinates from gridded file
            if haskey(ds_gridded, "lat") && haskey(ds_gridded, "lon")
                set!(coords, "latitude", ds_gridded["lat"][:])
                set!(coords, "longitude", ds_gridded["lon"][:])
            end
            
            # Load requested variables that exist in gridded file and haven't been loaded yet
            for var in variables
                if !haskey(loaded_data, var) && var in gridded_available_vars
                    data = Array{Float32}(ds_gridded[var][time_idxs...])
                    set!(loaded_data, var, data)
                end
            end
            
        finally
            close(ds_gridded)
        end
    end
    
    # Warn about any variables that couldn't be loaded
    for var in variables
        if !haskey(loaded_data, var)
            @warn "Variable $var not found in CERES data files"
        end
    end

    #If the variable is lw, multiply it by -1 if lw_sign_flip is true
    if lw_sign_flip
        for (var, data) in loaded_data
            if occursin("lw", var)
                loaded_data[var] .*= -1
            end
        end
    end

    return loaded_data, coords
end

"""
    load_enso_data(time_period; data_dir="../../data/ENSO", filename="enso_data.csv", lags=nothing, date_column="date")

Load ENSO index data for specified time period and lag values.

# Arguments
- `time_period`: Tuple of (start_date, end_date) as Date objects
- `data_dir`: Directory containing ENSO CSV file
- `filename`: ENSO data CSV filename
- `lags`: Vector of lag values (e.g., [-2, -1, 0, 1, 2]) or nothing to load all available lags
- `date_column`: Name of column containing dates

# Returns
Dictionary with ONI lag column names as keys and ENSO index values as arrays.
Also returns coordinates dictionary with time array.

# Examples
```julia
# Load specific lags
enso_data, coords = load_enso_data(time_period; lags=[0, 1, 2])

# Load all available lags
enso_data, coords = load_enso_data(time_period; lags=nothing)

# Load single lag (backward compatibility)
enso_data, coords = load_enso_data(time_period; lags=[0])
```
"""
function load_enso_data(time_period; 
                       data_dir="../../data/ENSO", 
                       filename="enso_data.csv", 
                       lags=nothing, 
                       date_column="date")
    
    enso_path = joinpath(data_dir, filename)
    
    if !isfile(enso_path)
        error("ENSO data file not found: $enso_path")
    end
    
    # Load the CSV data
    enso_df = CSV.read(enso_path, DataFrame)
    
    # Convert dates and add day offset for monthly data
    enso_time = DateTime.(enso_df[!, Symbol(date_column)] .+ Day(14))
    
    # Filter by time period
    valid_times = in_time_period.(enso_time, Ref(time_period))
    filtered_time = enso_time[valid_times]
    
    # Determine which columns to load
    if lags === nothing
        # Load all available lag columns
        all_columns = names(enso_df)
        lag_columns = filter(col -> startswith(string(col), "oni_lag"), all_columns)
        index_columns = string.(lag_columns)
    else
        # Load specific lag columns - handle negative numbers properly
        index_columns = ["oni_lag_$lag" for lag in lags]
    end
    
    # Return as Dictionary and coordinates
    loaded_data = Dictionary()
    coords = Dictionary()
    
    # Load all requested index columns
    for index_col in index_columns
        if index_col in string.(names(enso_df))
            enso_index = enso_df[!, Symbol(index_col)]
            filtered_index = enso_index[valid_times]
            set!(loaded_data, index_col, filtered_index)
        else
            @warn "ENSO column $index_col not found in data file"
        end
    end
    
    set!(coords, "time", filtered_time)
    
    return loaded_data, coords
end

"""
    load_sepac_sst_index(time_period; data_dir="../../data/SEPac_SST", filename="sepac_sst_index.csv", lags=nothing)

Load SEPac SST index data for a specified time period and lag values.

# Arguments
- `time_period`: Tuple of (start_date, end_date) as Date objects
- `data_dir`: Directory containing the SEPac SST index CSV file
- `filename`: SEPac SST index data CSV filename
- `lags`: Vector of lag values (e.g., [-2, -1, 0, 1, 2]) or nothing to load all available lags

# Returns
Dictionary with SEPac SST index and its lags as keys and corresponding values as arrays.
Also returns coordinates dictionary with time array.

# Examples
```julia
# Load specific lags
sepac_sst_data, coords = load_sepac_sst_index(time_period; lags=[0, 1, 2])

# Load all available lags
sepac_sst_data, coords = load_sepac_sst_index(time_period; lags=nothing)

# Load single lag (backward compatibility)
sepac_sst_data, coords = load_sepac_sst_index(time_period; lags=[0])
```
"""
function load_sepac_sst_index(time_period; data_dir="../../data/SEPac_SST", filename="sepac_sst_index.csv", lags=nothing)
    sepac_path = joinpath(data_dir, filename)

    if !isfile(sepac_path)
        error("SEPac SST index file not found: $sepac_path")
    end

    # Load the CSV data
    sepac_df = CSV.read(sepac_path, DataFrame)

    # Convert dates
    sepac_time = Date.(sepac_df[:, :Date])

    # Filter by time period
    valid_times = in_time_period.(sepac_time, Ref(time_period))
    filtered_time = sepac_time[valid_times]

    # Determine which columns to load
    if lags === nothing
        # Load all available lag columns
        all_columns = names(sepac_df)
        lag_columns = filter(col -> startswith(string(col), "SEPac_SST_Index"), all_columns)
        index_columns = string.(lag_columns)
    else
        # Load specific lag columns - handle negative numbers properly
        index_columns = ["SEPac_SST_Index_Lag$lag" for lag in lags]
    end

    # Return as Dictionary and coordinates
    loaded_data = Dictionary()
    coords = Dictionary()

    # Load all requested index columns
    for index_col in index_columns
        if index_col in string.(names(sepac_df))
            sepac_index = sepac_df[!, Symbol(index_col)]
            filtered_index = sepac_index[valid_times]
            set!(loaded_data, index_col, filtered_index)
        else
            @warn "SEPac SST column $index_col not found in data file"
        end
    end

    set!(coords, "time", filtered_time)

    return loaded_data, coords
end

"""
    load_eli_data(time_period; data_dir="../../data/ENSO", filename="eli_data.csv", lags=nothing, date_column="Date")

Load ELI (ENSO Longitude Index) data for specified time period and lag values.

# Arguments
- `time_period`: Tuple of (start_date, end_date) as Date objects
- `data_dir`: Directory containing ELI CSV file
- `filename`: ELI data CSV filename
- `lags`: Vector of lag values (e.g., [-6, -3, 0, 3, 6]) or nothing to load all available lags
- `date_column`: Name of column containing dates

# Returns
Dictionary with ELI lag column names as keys and ELI index values as arrays.
Also returns coordinates dictionary with time array.
"""
function load_eli_data(time_period; 
                      data_dir="../../data/ENSO", 
                      filename="eli_data.csv", 
                      lags=nothing, 
                      date_column="Date")
    
    eli_path = joinpath(data_dir, filename)
    
    if !isfile(eli_path)
        error("ELI data file not found: $eli_path")
    end
    
    # Load the CSV data
    eli_df = CSV.read(eli_path, DataFrame)
    
    # Convert dates and add day offset for monthly data
    eli_time = DateTime.(eli_df[!, Symbol(date_column)] .+ Day(14))
    
    # Filter by time period
    valid_times = in_time_period.(eli_time, Ref(time_period))
    filtered_time = eli_time[valid_times]
    
    # Determine which columns to load
    if lags === nothing
        # Load all available lag columns
        all_columns = names(eli_df)
        lag_columns = filter(col -> startswith(string(col), "ELI_Lag"), all_columns)
        index_columns = string.(lag_columns)
    else
        # Load specific lag columns - handle negative numbers properly
        index_columns = ["ELI_Lag$lag" for lag in lags]
    end
    
    # Return as Dictionary and coordinates
    loaded_data = Dictionary()
    coords = Dictionary()
    
    # Load all requested index columns
    for index_col in index_columns
        if index_col in string.(names(eli_df))
            eli_index = eli_df[!, Symbol(index_col)]
            filtered_index = eli_index[valid_times]
            set!(loaded_data, index_col, filtered_index)
        else
            @warn "ELI column $index_col not found in data file"
        end
    end
    
    set!(coords, "time", filtered_time)
    
    return loaded_data, coords
end

"""
    reshape_and_concatenate(arrays, array_names, time_indices)

Reshape arrays to (time, other_dims) and concatenate along the second dimension.

# Arguments
- `arrays`: Vector of arrays to reshape and concatenate
- `array_names`: Vector of names for each array
- `time_indices`: Vector of integers indicating which dimension is time for each array

# Returns
- `concatenated_array`: Matrix of size (time, total_spatial_dims) 
- `variable_indices`: Dictionary mapping variable names to column indices in concatenated array
- `original_shapes`: Dictionary mapping variable names to original array dimensions

# Example
```julia
# Arrays with different shapes but same time dimension
arr1 = randn(100, 50, 200)  # (lon, lat, time) - time_index = 3
arr2 = randn(100, 50, 10, 200)  # (lon, lat, pressure, time) - time_index = 4
arrays = [arr1, arr2]
array_names = ["surface_temp", "pressure_temp"]
time_indices = [3, 4]

concat_arr, var_indices, orig_shapes = reshape_and_concatenate(arrays, array_names, time_indices)
```
"""
function reshape_and_concatenate(arrays, array_names, time_indices)
    if length(arrays) != length(array_names) || length(arrays) != length(time_indices)
        error("arrays, array_names, and time_indices must have the same length")
    end
    
    # Store original shapes and reshape arrays
    reshaped_arrays = []
    variable_indices = Dictionary()
    original_shapes = Dictionary()
    
    current_col = 1
    
    for (i, (arr, name, time_idx)) in enumerate(zip(arrays, array_names, time_indices))
        # Store original shape
        orig_shape = size(arr)
        set!(original_shapes, name, orig_shape)
        
        # Move time dimension to first position
        perm_dims = [time_idx; setdiff(1:ndims(arr), time_idx)]
        arr_permuted = permutedims(arr, perm_dims)
        
        # Reshape to (time, spatial_dims)
        time_size = size(arr_permuted, 1)
        spatial_size = prod(size(arr_permuted)[2:end])
        arr_reshaped = reshape(arr_permuted, time_size, spatial_size)
        
        # Store column indices for this variable
        end_col = current_col + spatial_size - 1
        set!(variable_indices, name, current_col:end_col)
        current_col = end_col + 1
        
        push!(reshaped_arrays, arr_reshaped)
    end
    
    # Concatenate all arrays along spatial dimension
    concatenated_array = hcat(reshaped_arrays...)
    
    return concatenated_array, variable_indices, original_shapes
end

"""
    deconcatenate_and_reshape(concatenated_array, variable_indices, original_shapes, time_indices)

Reverse the operation of reshape_and_concatenate to recover original arrays.

# Arguments
- `concatenated_array`: Matrix from reshape_and_concatenate
- `variable_indices`: Dictionary mapping variable names to column indices
- `original_shapes`: Dictionary mapping variable names to original dimensions
- `time_indices`: Vector of integers indicating original time dimension for each variable

# Returns
- Dictionary mapping variable names to their original array shapes

# Example
```julia
# Continuing from reshape_and_concatenate example
recovered_arrays = deconcatenate_and_reshape(concat_arr, var_indices, orig_shapes, [3, 4])
surface_temp = recovered_arrays["surface_temp"]  # Back to (100, 50, 200)
pressure_temp = recovered_arrays["pressure_temp"]  # Back to (100, 50, 10, 200)
```
"""
function deconcatenate_and_reshape(concatenated_array, variable_indices, original_shapes, time_indices)
    recovered_arrays = Dictionary()
    
    # Get variable names from either dictionary (they should match)
    var_names = collect(keys(variable_indices))
    
    if length(var_names) != length(time_indices)
        error("Number of variables in indices must match length of time_indices")
    end
    
    for (i, var_name) in enumerate(var_names)
        # Extract columns for this variable
        col_range = variable_indices[var_name]
        var_data = concatenated_array[:, col_range]
        
        # Get original shape and time index
        orig_shape = original_shapes[var_name]
        time_idx = time_indices[i]
        
        # Reshape from (time, spatial) back to (time, spatial_dims...)
        time_size = size(var_data, 1)
        spatial_dims = orig_shape[setdiff(1:length(orig_shape), time_idx)]
        reshaped_data = reshape(var_data, time_size, spatial_dims...)
        
        # Permute dimensions back to original order
        # Create inverse permutation
        perm_dims = [time_idx; setdiff(1:length(orig_shape), time_idx)]
        inv_perm = sortperm(perm_dims)
        final_data = permutedims(reshaped_data, inv_perm)
        
        set!(recovered_arrays, var_name, final_data)
    end
    
    return recovered_arrays
end

"""
    reconstruct_spatial_arrays(spatial_vector, variable_indices, original_shapes)

Reconstruct original array shapes from a flattened spatial vector (no time dimension).

This function is useful for reconstructing spatial patterns from PLS components, 
regression coefficients, or other spatial-only data derived from the concatenated arrays.

# Arguments
- `spatial_vector`: Vector of length equal to total spatial dimensions (no time)
- `variable_indices`: Dictionary mapping variable names to column indices (from reshape_and_concatenate)
- `original_shapes`: Dictionary mapping variable names to original array dimensions (from reshape_and_concatenate)

# Returns
- Dictionary mapping variable names to their reconstructed spatial arrays (without time dimension)

# Example
```julia
# Assume we have PLS regression coefficients for spatial patterns
spatial_coeffs = randn(7267683)  # Length matches total spatial features
reconstructed = reconstruct_spatial_arrays(spatial_coeffs, var_indices, orig_shapes)
# Returns:
# "t": (1440, 721, 4) - temperature spatial pattern
# "z": (1440, 721) - geopotential spatial pattern  
# "t2m": (1440, 721) - surface temp spatial pattern
# etc.
```
"""
function reconstruct_spatial_arrays(spatial_vector, variable_indices, original_shapes)
    reconstructed_arrays = Dictionary()
    
    for (var_name, col_indices) in pairs(variable_indices)
        # Extract the spatial data for this variable
        var_spatial_data = spatial_vector[col_indices]
        
        # Get original shape and remove time dimension
        orig_shape = original_shapes[var_name]
        
        # The spatial shape is all dimensions except the last one (time)
        spatial_shape = orig_shape[1:end-1]
        
        # Reshape the flattened spatial data back to spatial dimensions
        reconstructed_data = reshape(var_spatial_data, spatial_shape)
        
        set!(reconstructed_arrays, var_name, reconstructed_data)
    end
    
    return reconstructed_arrays
end

