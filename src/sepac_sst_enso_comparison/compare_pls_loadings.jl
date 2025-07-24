using Plots, Statistics, StatsBase, Dates, SplitApplyCombine
using PythonCall
@py import matplotlib.pyplot as plt, matplotlib.colors as colors, cartopy.crs as ccrs, cmasher as cmr

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")
include("../utils/plot_global.jl")

"""
This script compares PLS loading components from:
1. SEPac SST Index effects (sepac_sst_effect_on_levels_vars.jl)
2. ENSO effects (enso_effect_on_levels_vars.jl)

It creates vertically stacked plots showing the PLS Y-loadings from both analyses
for direct comparison of spatial patterns.
"""

# Create output directory
visdir = "../../vis/sepac_sst_enso_comparison/"
if !isdir(visdir)
    mkpath(visdir)
end

println("="^60)
println("COMPARING PLS LOADINGS: SEPac SST vs ENSO")
println("="^60)

# Variables to analyze
stacked_var_names = ["temp", "press_geopotential"]
stacked_level_vars = Dictionary(stacked_var_names, [["t2m", "t"], ["msl", "z"]])
level_names = ["sfc", "850hPa", "700hPa", "500hPa", "250hPa"]

single_vars_era = ["u10", "v10"]
single_vars_ceres = ["toa_net_all_mon", "gridded_net_sw", "toa_lw_all_mon"]

# Define lags for both indices
sepac_lags = [-6, -3, 0, 3, 6]
sepac_lag_names = ["SEPac_SST_Index_Lag$lag" for lag in sepac_lags]

oni_lags = [-6, -3, 0, 3, 6]
oni_lag_names = ["oni_lag_$(lag)" for lag in oni_lags]

# Load and prepare SEPac SST Index data
println("Loading SEPac SST Index data...")
sepac_data, sepac_coords = load_sepac_sst_index(time_period; lags=sepac_lags)

# Load and prepare ENSO data
println("Loading ENSO data...")
enso_data, enso_coords = load_enso_data(time_period; lags=oni_lags)

# Prepare time data (assuming same time period for both)
times = sepac_coords["time"]
float_times = @. year(times) + (month(times) - 1) / 12
month_groups = groupfind(month.(times))

# Detrend and deseasonalize SEPac data
for sepac_col in sepac_lag_names
    detrend_and_deseasonalize_precalculated_groups!(sepac_data[sepac_col], float_times, month_groups)
end
sepac_data_block = hcat([sepac_data[col] for col in sepac_lag_names]...)

# Detrend and deseasonalize ENSO data
for oni_col in oni_lag_names
    detrend_and_deseasonalize_precalculated_groups!(enso_data[oni_col], float_times, month_groups)
end
enso_data_block = hcat([enso_data[col] for col in oni_lag_names]...)

"""
Function to compute PLS loadings for a given variable and predictor data
"""
function compute_pls_loadings(var_data, predictor_data, float_times, month_groups)
    # Detrend and deseasonalize variable data
    if ndims(var_data) == 4  # Multi-level data
        for slice in eachslice(var_data, dims = (1,2,3))
            detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
        end
        var_data_dims = size(var_data)
        Y = reshape(var_data, prod(var_data_dims[1:3]), var_data_dims[4])
        Y = permutedims(Y, (2,1))
        
        # Perform PLS regression
        pls_model = make_pls_regressor(predictor_data, Y, 1; print_updates=false)
        pls_y_loadings = pls_model.Y_loadings[:, 1]
        pls_y_loadings = reshape(pls_y_loadings, var_data_dims[1:3]...)
        
        return pls_y_loadings, var_data_dims
        
    elseif ndims(var_data) == 3  # Single level data
        for slice in eachslice(var_data, dims = (1,2))
            detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
        end
        var_data_dims = size(var_data)
        Y = reshape(var_data, prod(var_data_dims[1:2]), var_data_dims[3])
        Y = permutedims(Y, (2,1))
        
        # Perform PLS regression
        pls_model = make_pls_regressor(predictor_data, Y, 1; print_updates=false)
        pls_y_loadings = pls_model.Y_loadings[:, 1]
        pls_y_loadings = reshape(pls_y_loadings, var_data_dims[1:2]...)
        
        return pls_y_loadings, var_data_dims
    end
end

"""
Function to create vertically stacked comparison plots using existing plot utilities
"""
function create_comparison_plot(lat, lon, sepac_loadings, enso_loadings, titles, main_title, filename)
    n_plots = length(titles)
    
    # Prepare all data slices and subtitles for plot_multiple_levels
    # Julia uses column-major order, so we need to interleave SEPac and ENSO for each variable
    all_slices = []
    all_subtitles = []
    
    # Interleave SEPac SST and ENSO loadings for proper column-major layout
    for i in 1:n_plots
        # First add SEPac SST loading (will be in top row due to column-major)
        push!(all_slices, Float64.(sepac_loadings[i]))
        push!(all_subtitles, "SEPac SST - $(titles[i])")
        
        # Then add ENSO loading (will be in bottom row due to column-major)
        push!(all_slices, Float64.(enso_loadings[i]))
        push!(all_subtitles, "ENSO - $(titles[i])")
    end
    
    # Create layout (2 rows × n_plots columns)
    layout = (2, n_plots)
    
    # Use the existing plot_multiple_levels function
    fig = plot_multiple_levels(lat, lon, all_slices, layout;
                              subtitles=all_subtitles,
                              colorbar_label="PLS Loading")
    
    # Add main title
    fig.suptitle(main_title, fontsize=16, y=0.98)
    
    # Save figure
    fig.savefig(joinpath(visdir, filename), dpi=300)
    plt.close(fig)
    
    println("Saved comparison plot: $filename")
end

# Analyze stacked level variables (temp and press_geopotential)
println("\n" * "="^50)
println("ANALYZING STACKED LEVEL VARIABLES")
println("="^50)

for (stacked_var_name, stacked_level_var) in zip(stacked_var_names, stacked_level_vars)
    println("\nAnalyzing $stacked_var_name...")
    
    # Load ERA5 data
    era5_data, era5_coords = load_era5_data(stacked_level_var, time_period)
    
    sfc_name = first(stacked_level_var)
    level_name = last(stacked_level_var)
    
    sfc_dims = size(era5_data[sfc_name])
    level_dims = size(era5_data[level_name])
    
    # Ensure both arrays have the same time dimension
    if length(sfc_dims) == 3 && length(level_dims) == 4
        era5_data[sfc_name] = reshape(era5_data[sfc_name], sfc_dims[1], sfc_dims[2], 1, sfc_dims[3])
    end
    
    vertical_concat_data = cat(era5_data[sfc_name], era5_data[level_name]; dims = 3)
    
    # Compute PLS loadings for SEPac SST
    println("Computing SEPac SST PLS loadings...")
    sepac_loadings, _ = compute_pls_loadings(vertical_concat_data, sepac_data_block, float_times, month_groups)
    
    # Compute PLS loadings for ENSO
    println("Computing ENSO PLS loadings...")
    enso_loadings, _ = compute_pls_loadings(vertical_concat_data, enso_data_block, float_times, month_groups)
    
    # Prepare data for plotting
    lat = Float64.(era5_coords["latitude"])
    lon = Float64.(era5_coords["longitude"])
    
    sepac_slices = [Float64.(sepac_loadings[:, :, j]) for j in 1:length(level_names)]
    enso_slices = [Float64.(enso_loadings[:, :, j]) for j in 1:length(level_names)]
    
    # Create comparison plot
    safe_var_name = replace(stacked_var_name, "/" => "_")
    create_comparison_plot(lat, lon, sepac_slices, enso_slices, level_names,
                          "PLS Loading Comparison: $stacked_var_name",
                          "$(safe_var_name)_pls_comparison.png")
end

# Analyze single level ERA5 variables
println("\n" * "="^50)
println("ANALYZING SINGLE LEVEL ERA5 VARIABLES")
println("="^50)

for var_name in single_vars_era
    println("\nAnalyzing $var_name...")
    
    # Load ERA5 data
    era5_data, era5_coords = load_era5_data([var_name], time_period)
    var_data = era5_data[var_name]
    
    # Compute PLS loadings for SEPac SST
    println("Computing SEPac SST PLS loadings...")
    sepac_loadings, _ = compute_pls_loadings(var_data, sepac_data_block, float_times, month_groups)
    
    # Compute PLS loadings for ENSO
    println("Computing ENSO PLS loadings...")
    enso_loadings, _ = compute_pls_loadings(var_data, enso_data_block, float_times, month_groups)
    
    # Prepare data for plotting
    lat = Float64.(era5_coords["latitude"])
    lon = Float64.(era5_coords["longitude"])
    
    # Create comparison plot (single variable)
    create_comparison_plot(lat, lon, [sepac_loadings], [enso_loadings], [var_name],
                          "PLS Loading Comparison: $var_name",
                          "$(var_name)_pls_comparison.png")
end

# Analyze CERES radiation variables
println("\n" * "="^50)
println("ANALYZING CERES RADIATION VARIABLES")
println("="^50)

for var_name in single_vars_ceres
    println("\nAnalyzing $var_name...")
    
    # Load CERES data
    ceres_data, ceres_coords = load_ceres_data([var_name], time_period)
    var_data = ceres_data[var_name]
    
    # Check if it's gridded data (skip global time series)
    if ndims(var_data) > 1
        # Compute PLS loadings for SEPac SST
        println("Computing SEPac SST PLS loadings...")
        sepac_loadings, _ = compute_pls_loadings(var_data, sepac_data_block, float_times, month_groups)
        
        # Compute PLS loadings for ENSO
        println("Computing ENSO PLS loadings...")
        enso_loadings, _ = compute_pls_loadings(var_data, enso_data_block, float_times, month_groups)
        
        # Prepare data for plotting
        lat = Float64.(ceres_coords["latitude"])
        lon = Float64.(ceres_coords["longitude"])
        
        # Create comparison plot
        create_comparison_plot(lat, lon, [sepac_loadings], [enso_loadings], [var_name],
                              "PLS Loading Comparison: $var_name",
                              "$(var_name)_pls_comparison.png")
    else
        println("$var_name is a global time series - skipping spatial comparison")
    end
end

println("\n" * "="^60)
println("PLS LOADING COMPARISON COMPLETE")
println("Comparison plots saved to: $visdir")
println("="^60)
