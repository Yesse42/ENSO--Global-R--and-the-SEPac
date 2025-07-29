using Plots, Statistics, StatsBase, Dates, SplitApplyCombine

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
visualize the effect of SEPac SST Index on various atmospheric variables
First it will do so for t2m and t on 4 levels simultaneously
Then it will do so for msl and z
Then it will do so for era5 u10, v10, and the ceres net, sw, and lw radiation variables
It will do so by calculating correlations between SEPac SST Index at lags -6, -3, 0, 3, and 6 and the variable of interest; it will also do 1-component and 3-component PLS regressions
It will save plots to vis/sepac_radiation_effects/gridded_vars/
"""

visdir = "../../vis/sepac_radiation_effects/gridded_vars/"
if !isdir(visdir)
    mkpath(visdir)
end

# Save X weights to text file
x_weights_dir = joinpath(visdir, "x_weights")
if !isdir(x_weights_dir)
    mkpath(x_weights_dir)
end

stacked_var_names = ["temp", "press_geopotential"]
stacked_level_vars = Dictionary(stacked_var_names, [["t2m", "t"], ["msl", "z"]])

level_names = ["sfc", "850hPa", "700hPa", "500hPa", "250hPa"]

single_vars_era = ["u10", "v10"]
single_vars_ceres = ["toa_net_all_mon", "gridded_net_sw", "toa_lw_all_mon"]

# Define SEPac SST Index lags to analyze
sepac_lags = [-12, -6, -3, 0, 3, 6, 12]
sepac_lag_names = ["SEPac_SST_Index_Lag$lag" for lag in sepac_lags]

#Load the SEPac SST Index data
println("Loading SEPac SST Index data...")
sepac_data, sepac_coords = load_sepac_sst_index(time_period; lags=sepac_lags)

#Prep it
times = sepac_coords["time"]
float_times = @. year(times) + (month(times) - 1) / 12
month_groups = groupfind(month.(times))
for sepac_col in sepac_lag_names
    detrend_and_deseasonalize_precalculated_groups!(sepac_data[sepac_col], float_times, month_groups)
end
sepac_data_block = hcat([sepac_data[col] for col in sepac_lag_names]...) #Block data for PLS

name_of_names_arr = []
arr_of_slices_arr = []

#Now begin looping through variables
for (stacked_var_name, stacked_level_var) in zip(stacked_var_names, stacked_level_vars)
    era5_data, era5_coords = load_era5_data(stacked_level_var, time_period)
    println("Loaded ERA5 data for $stacked_var_name")

    sfc_name = first(stacked_level_var)
    level_name = last(stacked_level_var)

    sfc_dims = size(era5_data[sfc_name])
    level_dims = size(era5_data[level_name])

    # Ensure both arrays have the same time dimension
    if length(sfc_dims) == 3 && length(level_dims) == 4
        # Surface data needs an extra dimension for levels
        era5_data[sfc_name] = reshape(era5_data[sfc_name], sfc_dims[1], sfc_dims[2], 1, sfc_dims[3])
    end

    vertical_concat_data = cat(era5_data[sfc_name], era5_data[level_name]; dims = 3)
    #Detrend and deseasonalize the vertical_concat_data
    for slice in eachslice(vertical_concat_data, dims = (1,2,3))
        detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
    end

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
    x_weights_file = joinpath(x_weights_dir, "$(safe_var_name)_x_weights.txt")
    open(x_weights_file, "w") do io
        println(io, "PLS X-Weights for $(stacked_var_name)")
        println(io, "=" ^ 50)
        println(io, "Component 1 weights showing contribution of each SEPac SST Index lag:")
        println(io, "")
        for (i, lag) in enumerate(sepac_lags)
            println(io, "SEPac SST Index lag $lag months: $(X_weights[i])")
        end
        println(io, "")
        println(io, "Note: Larger absolute values indicate stronger contribution")
        println(io, "Positive values: Variable increases with positive SEPac SST Index")
        println(io, "Negative values: Variable decreases with positive SEPac SST Index")
    end
    
    println("Saved X weights to: $x_weights_file")

    # Plot correlations for all SEPac SST Index lags and levels in one large plot
    println("Plotting correlations for $stacked_var_name...")
    lat = Float64.(era5_coords["latitude"])  # Convert to Float64
    lon = Float64.(era5_coords["longitude"])  # Convert to Float64
    
    # Prepare data slices and subtitles for the comprehensive plot
    all_corr_slices = []
    all_corr_subtitles = []
    
    for (i, lag_name) in enumerate(sepac_lag_names)
        for (j, level) in enumerate(level_names)
            # Get the correlation slice for this lag and level
            slice_idx = (i-1) * length(level_names) + j
            corr_slice = Float64.(slices_arr[slice_idx])  # Convert to Float64
            push!(all_corr_slices, corr_slice)
            push!(all_corr_subtitles, "$level - SEPac SST lag $(sepac_lags[i])")
        end
    end

    # Create one large plot with all correlations (5 lags × 7 levels = 35 subplots)
    layout = (length(level_names), length(sepac_lags))  # 5 rows (lags) × 7 columns (levels)
    
    # Debug: Check data types
    println("Number of slices: ", length(all_corr_slices))
    println("Sample slice type: ", typeof(all_corr_slices[1]))
    println("Sample slice size: ", size(all_corr_slices[1]))
    
    # Use the fixed plot_multiple_levels function
    corr_fig = plot_multiple_levels(lat, lon, all_corr_slices, layout;
                                   subtitles=all_corr_subtitles,
                                   colorbar_label="Correlation with SEPac SST Index")
    
    # Save the comprehensive correlation plot
    corr_fig.suptitle("$(stacked_var_name) - SEPac SST Index Correlations (Rows: SEPac SST Lags, Columns: Levels)", fontsize=16)
    corr_fig.savefig(joinpath(visdir, "$(safe_var_name)_all_correlations.png"), dpi=300, bbox_inches="tight")
    plt.close(corr_fig)
    
    # Plot PLS y-loadings for all levels in one plot
    println("Plotting PLS loadings for $stacked_var_name...")
    pls_slices = []
    pls_subtitles = []
    
    for (j, level) in enumerate(level_names)
        # Extract the PLS loading slice for this level
        pls_slice = Float64.(pls_y_loadings[:, :, j])  # Convert to Float64
        push!(pls_slices, pls_slice)
        push!(pls_subtitles, "$level - PLS Loading")
    end
    
    # Create one plot with all PLS loadings (1 row × 5 levels)
    pls_layout = (1, length(level_names))
    pls_fig = plot_multiple_levels(lat, lon, pls_slices, pls_layout;
                                  subtitles=pls_subtitles,
                                  colorbar_label="PLS Loading")
    
    # Save the PLS plot
    pls_fig.suptitle("$(stacked_var_name) - PLS Y-Loadings", fontsize=16)
    pls_fig.savefig(joinpath(visdir, "$(safe_var_name)_pls_loadings.png"), dpi=300, bbox_inches="tight")
    plt.close(pls_fig)
end

# Now analyze single level ERA5 variables
println("\n" * "="^60)
println("ANALYZING SINGLE LEVEL ERA5 VARIABLES")
println("="^60)

for var_name in single_vars_era
    println("\nLoading ERA5 variable: $var_name")
    era5_data, era5_coords = load_era5_data([var_name], time_period)
    
    # Get the variable data
    var_data = era5_data[var_name]
    println("Data dimensions: ", size(var_data))
    
    # Detrend and deseasonalize
    for slice in eachslice(var_data, dims = (1,2))
        detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
    end
    
    # Calculate correlations for each SEPac SST Index lag
    names_arr = String[]
    slices_arr = []
    for lag_name in sepac_lag_names
        sepac_idx_data = sepac_data[lag_name]
        simple_corrs = cor.(eachslice(var_data, dims = (1,2)), Ref(sepac_idx_data))
        push!(names_arr, "corr_$(lag_name)")
        push!(slices_arr, simple_corrs)
    end
    
    # Perform PLS regression
    X = sepac_data_block
    var_data_dims = size(var_data)
    Y = reshape(var_data, prod(var_data_dims[1:2]), var_data_dims[3])
    Y = permutedims(Y, (2,1)) # Time first dimension
    pls_model = make_pls_regressor(X, Y, 1; print_updates=false)
    
    pls_y_loadings = pls_model.Y_loadings[:, 1]
    pls_y_loadings = reshape(pls_y_loadings, var_data_dims[1:2]...)
    X_weights = pls_model.X_weights[:, 1]
    
    # Save X weights
    x_weights_file = joinpath(x_weights_dir, "$(var_name)_x_weights.txt")
    open(x_weights_file, "w") do io
        println(io, "PLS X-Weights for $(var_name)")
        println(io, "=" ^ 50)
        println(io, "Component 1 weights showing contribution of each SEPac SST Index lag:")
        println(io, "")
        for (i, lag) in enumerate(sepac_lags)
            println(io, "SEPac SST Index lag $lag months: $(X_weights[i])")
        end
        println(io, "")
    end
    println("Saved X weights to: $x_weights_file")
    
    # Plot correlations
    println("Plotting correlations for $var_name...")
    lat = Float64.(era5_coords["latitude"])
    lon = Float64.(era5_coords["longitude"])
    
    # Prepare correlation slices and subtitles
    corr_slices_float = [Float64.(slice) for slice in slices_arr]
    corr_subtitles = ["SEPac SST lag $(sepac_lags[i])" for i in 1:length(sepac_lags)]
    
    # Plot correlations (1 row × 5 lags)
    layout = (1, length(sepac_lags))
    corr_fig = plot_multiple_levels(lat, lon, corr_slices_float, layout;
                                   subtitles=corr_subtitles,
                                   colorbar_label="Correlation with SEPac SST Index")
    
    corr_fig.suptitle("$(var_name) - SEPac SST Index Correlations", fontsize=16)
    corr_fig.savefig(joinpath(visdir, "$(var_name)_correlations.png"), dpi=300, bbox_inches="tight")
    plt.close(corr_fig)
    
    # Plot PLS loading (single plot)
    println("Plotting PLS loading for $var_name...")
    pls_slice_float = Float64.(pls_y_loadings)
    
    pls_fig = plot_global_heatmap(lat, lon, pls_slice_float;
                                 title="$(var_name) - PLS Y-Loading",
                                 colorbar_label="PLS Loading")
    
    pls_fig.savefig(joinpath(visdir, "$(var_name)_pls_loading.png"), dpi=300, bbox_inches="tight")
    plt.close(pls_fig)
end

# Now analyze CERES radiation variables
println("\n" * "="^60)
println("ANALYZING CERES RADIATION VARIABLES")
println("="^60)

for var_name in single_vars_ceres
    println("\nLoading CERES variable: $var_name")
    ceres_data, ceres_coords = load_ceres_data([var_name], time_period)
    
    # Get the variable data
    var_data = ceres_data[var_name]
    println("Data dimensions: ", size(var_data))
    
    # Check if it's a global time series or gridded data
    if ndims(var_data) == 1
        # Global time series - skip spatial analysis
        println("$var_name is a global time series - performing temporal analysis only")
        
        # Detrend and deseasonalize
        detrend_and_deseasonalize_precalculated_groups!(var_data, float_times, month_groups)
        
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
        results_file = joinpath(x_weights_dir, "$(var_name)_analysis.txt")
        open(results_file, "w") do io
            println(io, "Analysis for $(var_name) (Global Time Series)")
            println(io, "=" ^ 60)
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
        end
        println("Saved analysis to: $results_file")
        
    else
        # Gridded data - perform spatial analysis like ERA5 variables
        # Detrend and deseasonalize
        for slice in eachslice(var_data, dims = (1,2))
            detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
        end
        
        # Calculate correlations for each SEPac SST Index lag
        names_arr = String[]
        slices_arr = []
        for lag_name in sepac_lag_names
            sepac_idx_data = sepac_data[lag_name]
            simple_corrs = cor.(eachslice(var_data, dims = (1,2)), Ref(sepac_idx_data))
            push!(names_arr, "corr_$(lag_name)")
            push!(slices_arr, simple_corrs)
        end
        
        # Perform PLS regression
        X = sepac_data_block
        var_data_dims = size(var_data)
        Y = reshape(var_data, prod(var_data_dims[1:2]), var_data_dims[3])
        Y = permutedims(Y, (2,1)) # Time first dimension
        pls_model = make_pls_regressor(X, Y, 1; print_updates=false)
        
        pls_y_loadings = pls_model.Y_loadings[:, 1]
        pls_y_loadings = reshape(pls_y_loadings, var_data_dims[1:2]...)
        X_weights = pls_model.X_weights[:, 1]
        
        # Save X weights
        x_weights_file = joinpath(x_weights_dir, "$(var_name)_x_weights.txt")
        open(x_weights_file, "w") do io
            println(io, "PLS X-Weights for $(var_name)")
            println(io, "=" ^ 50)
            println(io, "Component 1 weights showing contribution of each SEPac SST Index lag:")
            println(io, "")
            for (i, lag) in enumerate(sepac_lags)
                println(io, "SEPac SST Index lag $lag months: $(X_weights[i])")
            end
            println(io, "")
        end
        println("Saved X weights to: $x_weights_file")
        
        # Plot correlations
        println("Plotting correlations for $var_name...")
        lat = Float64.(ceres_coords["latitude"])
        lon = Float64.(ceres_coords["longitude"])
        
        # Prepare correlation slices and subtitles
        corr_slices_float = [Float64.(slice) for slice in slices_arr]
        corr_subtitles = ["SEPac SST lag $(sepac_lags[i])" for i in 1:length(sepac_lags)]
        
        # Plot correlations (1 row × 5 lags)
        layout = (1, length(sepac_lags))
        corr_fig = plot_multiple_levels(lat, lon, corr_slices_float, layout;
                                       subtitles=corr_subtitles,
                                       colorbar_label="Correlation with SEPac SST Index")
        
        corr_fig.suptitle("$(var_name) - SEPac SST Index Correlations", fontsize=16)
        corr_fig.savefig(joinpath(visdir, "$(var_name)_correlations.png"), dpi=300, bbox_inches="tight")
        plt.close(corr_fig)
        
        # Plot PLS loading (single plot)
        println("Plotting PLS loading for $var_name...")
        pls_slice_float = Float64.(pls_y_loadings)
        
        pls_fig = plot_global_heatmap(lat, lon, pls_slice_float;
                                     title="$(var_name) - PLS Y-Loading",
                                     colorbar_label="PLS Loading")
        
        pls_fig.savefig(joinpath(visdir, "$(var_name)_pls_loading.png"), dpi=300, bbox_inches="tight")
        plt.close(pls_fig)
    end
end

println("\n" * "="^60)
println("ANALYSIS COMPLETE")
println("="^60)
