using Plots, Statistics, StatsBase, Dates, SplitApplyCombine, NCDatasets, Dictionaries

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")
include("../utils/plot_global.jl")

"""
This script will:
visualize the effect of SEPac SST Index on various ENSO-deflated atmospheric variables
First it will do so for deflated t2m and t on 4 levels simultaneously
Then it will do so for deflated msl and z
Then it will do so for deflated ceres net, sw, and lw radiation variables (global time series)
It will do so by calculating correlations between SEPac SST Index at lags -6, -3, 0, 3, and 6 and the variable of interest; it will also do 1-component and 3-component PLS regressions
It will save plots to vis/sepac_radiation_effects/enso_deflated_vars/
"""

visdir = "../../vis/sepac_radiation_effects/enso_deflated_vars/"
if !isdir(visdir)
    mkpath(visdir)
end

# Save X weights to text file
x_weights_dir = joinpath(visdir, "x_weights")
if !isdir(x_weights_dir)
    mkpath(x_weights_dir)
end

# Load ENSO deflated data
datapath = "../../data/ENSO_Deflated/era5_ceres_enso_deflated.nc"
dataset = NCDataset(datapath, "r")

# Load coordinate data
lat = dataset["latitude"][:]
lon = dataset["longitude"][:]
p_levels = dataset["pressure_level"][:]
times_days = dataset["time"][:]
# Convert to DateTime
times = times_days

close(dataset)

# Convert times for detrending
float_times = @. year(times) + (month(times) - 1) / 12
month_groups = groupfind(month.(times))

stacked_var_names = ["temp", "press_geopotential"]
stacked_level_vars = Dictionary(stacked_var_names, [["deflated_t2m", "deflated_t"], ["deflated_msl", "deflated_z"]])

level_names = ["sfc", "850hPa", "700hPa", "500hPa", "250hPa"]

single_vars_ceres = ["deflated_gtoa_net_all_mon", "deflated_gtoa_sw_all_mon", "deflated_gtoa_lw_all_mon"]

# Define SEPac SST Index lags to analyze
sepac_lags = [-6, -3, 0, 3, 6]
sepac_lag_names = ["SEPac_SST_Index_Lag$lag" for lag in sepac_lags]

#Load the SEPac SST Index data
println("Loading SEPac SST Index data...")
sepac_data, sepac_coords = load_sepac_sst_index(time_period; lags=sepac_lags)

#Prep it
sepac_times = sepac_coords["time"]
sepac_float_times = @. year(sepac_times) + (month(sepac_times) - 1) / 12
sepac_month_groups = groupfind(month.(sepac_times))
for sepac_col in sepac_lag_names
    detrend_and_deseasonalize_precalculated_groups!(sepac_data[sepac_col], sepac_float_times, sepac_month_groups)
end
sepac_data_block = hcat([sepac_data[col] for col in sepac_lag_names]...) #Block data for PLS

name_of_names_arr = []
arr_of_slices_arr = []

#Now begin looping through ENSO-deflated variables
for (stacked_var_name, stacked_level_var) in zip(stacked_var_names, stacked_level_vars)
    # Load the ENSO deflated data
    dataset = NCDataset(datapath, "r")
    
    sfc_name = first(stacked_level_var)
    level_name = last(stacked_level_var)

    # Load the deflated data
    deflated_sfc_data = dataset[sfc_name][:, :, :]  # (lat, lon, time)
    deflated_level_data = dataset[level_name][:, :, :, :]  # (lat, lon, pressure, time)
    
    close(dataset)
    
    println("Loaded ENSO-deflated data for $stacked_var_name")

    # Transpose to match expected dimensions: (lon, lat, [pressure,] time)
    deflated_sfc_data = permutedims(deflated_sfc_data, (2, 1, 3))  # (lon, lat, time)
    deflated_level_data = permutedims(deflated_level_data, (2, 1, 3, 4))  # (lon, lat, pressure, time)

    sfc_dims = size(deflated_sfc_data)
    level_dims = size(deflated_level_data)

    # Ensure both arrays have the same structure for concatenation
    if length(sfc_dims) == 3 && length(level_dims) == 4
        # Surface data needs an extra dimension for levels
        deflated_sfc_data = reshape(deflated_sfc_data, sfc_dims[1], sfc_dims[2], 1, sfc_dims[3])
    end

    vertical_concat_data = cat(deflated_sfc_data, deflated_level_data; dims = 3)
    
    # The deflated data should already be detrended and deseasonalized, but we might want to 
    # apply additional processing if needed. For now, we'll use it as-is since it's been deflated.
    
    #Now for each SEPac SST Index lag calculate the correlations for each slice of this array
    names_arr = String[]
    slices_arr = []
    for lag_name in sepac_lag_names
        #Calculate the correlations on each level
        sepac_idx_data = sepac_data[lag_name]
        simple_corrs = cor.(eachslice(vertical_concat_data, dims = (1,2,3)), Ref(sepac_idx_data))
        append!(names_arr, ["corr_$(lag_name)_$(level_name)" for level in level_names])
        append!(slices_arr, collect(eachslice(simple_corrs, dims = 3)))
    end

    #Now perform the 1-component PLS regression and grab the y-loadings
    X = sepac_data_block

    vertical_concat_data_dims = size(vertical_concat_data)
    Y = reshape(vertical_concat_data, prod(vertical_concat_data_dims[1:3]), vertical_concat_data_dims[4])
    Y = permutedims(Y, (2,1)) #Now time is first dimension
    
    pls_model = make_pls_regressor(X, Y, 1; print_updates=false)
    
    pls_y_loadings = pls_model.Y_loadings[:, 1]
    pls_y_loadings = reshape(pls_y_loadings, vertical_concat_data_dims[1:3]...)

    X_weights = pls_model.X_weights[:, 1]
    
    # Create text file with X weights and corresponding SEPac SST Index lag information
    safe_var_name = replace(stacked_var_name, "/" => "_")  # Replace / with _ for filename
    x_weights_file = joinpath(x_weights_dir, "$(safe_var_name)_enso_deflated_x_weights.txt")
    open(x_weights_file, "w") do io
        println(io, "PLS X-Weights for ENSO-deflated $(stacked_var_name)")
        println(io, "=" ^ 60)
        println(io, "Component 1 weights showing contribution of each SEPac SST Index lag:")
        println(io, "")
        for (i, lag) in enumerate(sepac_lags)
            println(io, "SEPac SST Index lag $lag months: $(X_weights[i])")
        end
        println(io, "")
        println(io, "Note: Larger absolute values indicate stronger contribution")
        println(io, "Positive values: Variable increases with positive SEPac SST Index")
        println(io, "Negative values: Variable decreases with positive SEPac SST Index")
        println(io, "")
        println(io, "This analysis is performed on ENSO-deflated data where")
        println(io, "the effect of ONI (ENSO) has been regressed out.")
    end
    
    println("Saved X weights to: $x_weights_file")

    # Plot correlations for all SEPac SST Index lags and levels in one large plot
    println("Plotting correlations for ENSO-deflated $stacked_var_name...")
    lat_float = Float64.(lat)  # Convert to Float64
    lon_float = Float64.(lon)  # Convert to Float64
    
    # Prepare data slices and subtitles for the comprehensive plot
    all_corr_slices = []
    all_corr_subtitles = []
    
    for (i, lag_name) in enumerate(sepac_lag_names)
        for (j, level) in enumerate(level_names)
            # Get the correlation slice for this lag and level
            slice_idx = (i-1) * length(level_names) + j
            corr_slice = Float64.(slices_arr[slice_idx]')  # Convert to Float64
            push!(all_corr_slices, corr_slice)
            push!(all_corr_subtitles, "$level - SEPac SST lag $(sepac_lags[i])")
        end
    end
    
    # Create one large plot with all correlations (5 lags × 5 levels = 25 subplots)
    layout = (length(sepac_lags), length(level_names))  # 5 rows (lags) × 5 columns (levels)
    
    # Debug: Check data types
    println("Number of slices: ", length(all_corr_slices))
    println("Sample slice type: ", typeof(all_corr_slices[1]))
    println("Sample slice size: ", size(all_corr_slices[1]))
    
    # Use the fixed plot_multiple_levels function
    corr_fig = plot_multiple_levels(lat_float, lon_float, all_corr_slices, layout;
                                   subtitles=all_corr_subtitles,
                                   colorbar_label="Correlation with SEPac SST Index")
    
    # Save the comprehensive correlation plot
    corr_fig.suptitle("ENSO-deflated $(stacked_var_name) - SEPac SST Index Correlations (Rows: SEPac SST Lags, Columns: Levels)", fontsize=16)
    corr_fig.savefig(joinpath(visdir, "$(safe_var_name)_enso_deflated_all_correlations.png"), dpi=300, bbox_inches="tight")
    plt.close(corr_fig)
    
    # Plot PLS y-loadings for all levels in one plot
    println("Plotting PLS loadings for ENSO-deflated $stacked_var_name...")
    pls_slices = []
    pls_subtitles = []
    
    for (j, level) in enumerate(level_names)
        # Extract the PLS loading slice for this level
        pls_slice = Float64.(pls_y_loadings[:, :, j]')  # Convert to Float64
        push!(pls_slices, pls_slice)
        push!(pls_subtitles, "$level - PLS Loading")
    end
    
    # Create one plot with all PLS loadings (1 row × 5 levels)
    pls_layout = (1, length(level_names))
    pls_fig = plot_multiple_levels(lat_float, lon_float, pls_slices, pls_layout;
                                  subtitles=pls_subtitles,
                                  colorbar_label="PLS Loading")
    
    # Save the PLS plot
    pls_fig.suptitle("ENSO-deflated $(stacked_var_name) - PLS Y-Loadings", fontsize=16)
    pls_fig.savefig(joinpath(visdir, "$(safe_var_name)_enso_deflated_pls_loadings.png"), dpi=300, bbox_inches="tight")
    plt.close(pls_fig)
end

# Now analyze ENSO-deflated CERES radiation variables (global time series)
println("\n" * "="^60)
println("ANALYZING ENSO-DEFLATED CERES RADIATION VARIABLES")
println("="^60)

for var_name in single_vars_ceres
    println("\nLoading ENSO-deflated CERES variable: $var_name")
    
    # Load the deflated data
    dataset = NCDataset(datapath, "r")
    var_data = dataset[var_name][:]  # Global time series
    close(dataset)
    
    println("Data dimensions: ", size(var_data))
    
    # Global time series - perform temporal analysis only
    println("$var_name is a global time series - performing temporal analysis only")
    
    # The deflated data should already be processed, but we could apply additional processing if needed
    
    # Calculate correlations for each SEPac SST Index lag
    lag_correlations = []
    for lag_name in sepac_lag_names
        sepac_idx_data = sepac_data[lag_name]
        corr_val = cor(var_data, sepac_idx_data)
        push!(lag_correlations, corr_val)
    end
    
    # Perform PLS regression
    X = sepac_data_block
    Y = reshape(var_data, length(var_data), 1)
    pls_model = make_pls_regressor(X, Y, 1; print_updates=false)
    X_weights = pls_model.X_weights[:, 1]
    
    # Save results to text file
    safe_var_name = replace(var_name, "/" => "_")
    results_file = joinpath(x_weights_dir, "$(safe_var_name)_enso_deflated_analysis.txt")
    open(results_file, "w") do io
        println(io, "Analysis for ENSO-deflated $(var_name) (Global Time Series)")
        println(io, "=" ^ 70)
        println(io, "")
        println(io, "CORRELATIONS WITH SEPac SST INDEX:")
        for (i, lag) in enumerate(sepac_lags)
            println(io, "SEPac SST Index lag $lag months: $(lag_correlations[i])")
        end
        println(io, "")
        println(io, "PLS X-WEIGHTS:")
        for (i, lag) in enumerate(sepac_lags)
            println(io, "SEPac SST Index lag $lag months: $(X_weights[i])")
        end
        println(io, "")
        println(io, "Note: This analysis is performed on ENSO-deflated data where")
        println(io, "the effect of ONI (ENSO) has been regressed out, allowing us")
        println(io, "to isolate the direct effects of SEPac SST variability.")
    end
    println("Saved analysis to: $results_file")
end

println("\n" * "="^60)
println("ENSO-DEFLATED ANALYSIS COMPLETE")
println("="^60)
println("Results saved to: $visdir")
