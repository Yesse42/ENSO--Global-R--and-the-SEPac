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

# Variables to analyze - start with simple single level variables
single_vars_era = ["u10", "v10"]

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
Function to compute PLS loadings for single level variables
"""
function compute_single_level_pls_loadings(var_data, predictor_data, float_times, month_groups)
    # Detrend and deseasonalize variable data
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
    
    return pls_y_loadings
end

"""
Function to create simple comparison plots for single variables
"""
function create_single_var_comparison(lat, lon, sepac_loading, enso_loading, var_name, filename)
    # Prepare data for plot_multiple_levels with column-major ordering
    # For a 2x1 layout: first element goes to top-left, second to bottom-left
    all_slices = [Float64.(sepac_loading), Float64.(enso_loading)]
    all_subtitles = ["SEPac SST - $var_name", "ENSO - $var_name"]
    
    # Create layout (2 rows × 1 column for vertical stacking)
    layout = (2, 1)
    
    # Use the existing plot_multiple_levels function
    fig = plot_multiple_levels(lat, lon, all_slices, layout;
                              subtitles=all_subtitles,
                              colorbar_label="PLS Loading")
    
    # Add main title
    fig.suptitle("PLS Loading Comparison: $var_name", fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig.savefig(joinpath(visdir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    println("Saved comparison plot: $filename")
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
    sepac_loadings = compute_single_level_pls_loadings(var_data, sepac_data_block, float_times, month_groups)
    
    # Compute PLS loadings for ENSO
    println("Computing ENSO PLS loadings...")
    enso_loadings = compute_single_level_pls_loadings(var_data, enso_data_block, float_times, month_groups)
    
    # Prepare data for plotting
    lat = Float64.(era5_coords["latitude"])
    lon = Float64.(era5_coords["longitude"])
    
    # Create comparison plot
    create_single_var_comparison(lat, lon, sepac_loadings, enso_loadings, var_name,
                                "$(var_name)_pls_comparison.png")
end

println("\n" * "="^60)
println("PLS LOADING COMPARISON COMPLETE")
println("Comparison plots saved to: $visdir")
println("="^60)
