using Plots, Statistics, StatsBase, Dates, SplitApplyCombine, Printf, Dictionaries, NCDatasets, JLD2

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")
include("../utils/plot_global.jl")
include("../k_fold_validator/k_fold_validator.jl")

"""
This script will:
1. Load t2m, t, z, and msl data from ERA5 
2. Load CERES radiation data (net, sw, lw)
3. Detrend and deseasonalize all data
4. Calculate a 10-component PLS regression model using atmospheric variables to predict CERES radiation
5. Perform 10-fold cross validation of the model and plot the results
6. Visualize PLS components spatially
"""

# Create visualization directory
visdir = "../../vis/basic_atmospheric_pls/"
if !isdir(visdir)
    mkpath(visdir)
end

# Create directory for saving PLS models
plsdir = "../../data/PLSs/"
if !isdir(plsdir)
    mkpath(plsdir)
end

plot_components = 1:5
max_component = maximum(plot_components)

# Use time period from constants.jl
println("Analyzing period: $(time_period[1]) to $(time_period[2])")

# Define variables to load
era5_vars = ["t2m", "t", "z", "msl"]
ceres_vars = ["gtoa_lw_all_mon", "gtoa_net_all_mon", "gtoa_sw_all_mon"]
println("ERA5 predictor variables: $(join(era5_vars, ", "))")
println("CERES predictand variables: $(join(ceres_vars, ", "))")

# Load ERA5 data 
println("Loading ERA5 data...")
era5_data, era5_coords = load_era5_data(era5_vars, time_period)

# Load CERES data
println("Loading CERES data...")
ceres_data, ceres_coords = load_ceres_data(ceres_vars, time_period)

# Get coordinate information
lat = era5_coords["latitude"]
lon = era5_coords["longitude"] 
if haskey(era5_coords, "pressure_level")
    press = era5_coords["pressure_level"]
else
    press = nothing
end

lat_size = length(lat)
lon_size = length(lon)
press_size = press === nothing ? 0 : length(press)

println("ERA5 spatial dimensions: lat=$lat_size, lon=$lon_size")
if press_size > 0
    println("Pressure levels: $press_size")
end

# Use common time from ERA5 dataset
common_time = era5_coords["time"]
println("Using $(length(common_time)) time points")

# Prepare data arrays for detrending and deseasonalization
println("Preparing data for detrending and deseasonalization...")

# Prepare predictor arrays (ERA5 variables) 
predictor_arrays = []
predictor_names = []
predictor_ndims = []

for var in era5_vars
    push!(predictor_arrays, era5_data[var])
    push!(predictor_names, var)
    push!(predictor_ndims, ndims(era5_data[var]))
end

# Prepare predictand arrays (CERES variables)
predictand_arrays = []
predictand_names = []

for var in ceres_vars
    push!(predictand_arrays, ceres_data[var])
    push!(predictand_names, var)
end

# Use the reshape_and_concatenate function for efficient processing of predictors
println("Using reshape_and_concatenate function for predictors...")
X_combined, predictor_indices, predictor_shapes = reshape_and_concatenate(
    predictor_arrays, predictor_names, predictor_ndims)
X_combined = Float32.(X_combined)  # Convert to Float32 for memory efficiency

println("Combined predictor data dimensions: $(size(X_combined))")

# Prepare time information for detrending and deseasonalizing
months = month.(common_time)
float_times = calc_float_time.(common_time)
month_groups = groupfind(months)

println("Detrending and deseasonalizing predictor data...")

# Check for missing data in predictors
has_missing_predictors = any(ismissing, X_combined)
if has_missing_predictors
    println("Warning: Predictor matrix has missing data")
end

# Detrend and deseasonalize predictor data
detrend_and_deseasonalize_precalculated_groups!.(eachcol(X_combined), Ref(float_times), Ref(month_groups))

println("Detrending and deseasonalizing CERES data...")

# Process each CERES variable separately since they're different radiation types
short_rad_names = ["lw", "net", "sw"]

for (short_name, ceres_var, predictand_name) in zip(short_rad_names, ceres_vars, predictand_names)
    println("\n" * "="^60)
    println("Processing $predictand_name (Short name: $short_name)")
    println("="^60)
    
    # Get the CERES data for this radiation type
    Y = copy(ceres_data[ceres_var])
    Y = Float32.(Y)
    
    # Check for missing data in this predictand
    has_missing_predictand = any(ismissing, Y)
    if has_missing_predictand
        println("Warning: Predictand $predictand_name has missing data")
    end
    
    # Detrend and deseasonalize this CERES variable
    if ndims(Y) == 1  # Global mean data
        detrend_and_deseasonalize_precalculated_groups!(Y, float_times, month_groups)
    else
        # For gridded data, process each grid point
        for i in 1:size(Y, 1), j in 1:size(Y, 2)
            detrend_and_deseasonalize_precalculated_groups!(Y[i, j, :], float_times, month_groups)
        end
    end
    
    println("Running PLS regression with $max_component components...")
    pls = make_pls_regressor(X_combined, Y, max_component)
    
    # Save PLS model and metadata
    pls_filename = joinpath(plsdir, "atmospheric_ceres_$(short_name)_pls.jld2")
    println("Saving PLS model to: $pls_filename")
    
    save(pls_filename, Dict(
        "pls_model" => reduce_pls_model(pls; copy=false),
        "predictor_indices" => predictor_indices,
        "predictor_shapes" => predictor_shapes,
        "predictor_names" => predictor_names,
        "predictand_name" => ceres_var,
        "short_name" => short_name,
        "era5_vars" => era5_vars,
        "time_period" => time_period,
        "coordinates" => Dict(
            "latitude" => lat,
            "longitude" => lon,
            "pressure_level" => press
        ),
        "analysis_info" => Dict(
            "n_time_points" => length(common_time),
            "n_features" => size(X_combined, 2),
            "max_components" => max_component,
            "creation_date" => string(now())
        )
    ))

    # Create visualization directory for this radiation type
    plotpath = joinpath(visdir, short_name * "_pls_basic")
    if !isdir(plotpath)
        mkpath(plotpath)
    end

    # Plot PLS components spatially
    for component in plot_components
        println("Plotting component $component for $short_name radiation...")
        
        raw_matrix = make_matrix_to_multiply_by_X_to_get_Y(pls; components=1:component)
        matrices_with_vars = reconstruct_spatial_arrays(raw_matrix, predictor_indices, predictor_shapes)

        for (varname, mat) in pairs(matrices_with_vars)
            println("  - Plotting $varname")
            savedir = joinpath(plotpath, varname)
            !isdir(savedir) && mkpath(savedir)
            
            ndim_mat = ndims(mat)
            if ndim_mat == 3  # Pressure level data
                for (p_idx, p_level) in enumerate(press)
                    fig = plot_global_heatmap(lat, lon, mat[:, :, p_idx]; 
                        title = "PLS #$component for $varname at $p_level hPa predicting $short_name radiation", 
                        colorbar_label = "PLS Weight")
                    fig.savefig(joinpath(savedir, "component$(component)_level$(p_level)hPa.png"))
                    plt.close()
                end
            elseif ndim_mat == 2  # Single level data
                fig = plot_global_heatmap(lat, lon, mat; 
                    title = "PLS #$component for $varname predicting $short_name radiation", 
                    colorbar_label = "PLS Weight")
                fig.savefig(joinpath(savedir, "component$(component).png"))
                plt.close()
            end
        end
    end

    # Perform k-fold validation
    n_folds = 10
    n_components = 1:10
    
    println("Running k-fold validation with $n_folds folds for components $n_components...")
    predictions_by_components, folds = k_fold_validate_pls_regressor_simultaneously(
        X_combined, Y, n_folds=n_folds, n_components=n_components)
    
    # Calculate validation metrics for each component
    validation_results = []
    for component in n_components
        predictions = predictions_by_components[component]
        
        # Calculate R²
        mean_y = mean(Y)
        ss_tot = sum((Y .- mean_y).^2)
        ss_res = sum((Y .- predictions).^2)
        r_squared = 1 - ss_res/ss_tot
        
        # Calculate correlation
        correlation = only(cor(Y, predictions))
        
        push!(validation_results, (component=component, r2=r_squared, cor=correlation))
    end
    
    # Save validation results to text file
    results_file = joinpath(visdir, "$(short_name)_pls_validation_results.txt")
    open(results_file, "w") do io
        println(io, "K-Fold Cross-Validation Results for $predictand_name")
        println(io, "="^60)
        println(io, "Predictors: $(join(era5_vars, ", "))")
        println(io, "Predictand: $predictand_name")
        println(io, "Number of folds: $n_folds")
        println(io, "Data points: $(size(X_combined, 1))")
        println(io, "Features: $(size(X_combined, 2))")
        println(io, "Time period: $(time_period[1]) to $(time_period[2])")
        println(io, "")
        println(io, "Component  R²      Correlation")
        println(io, "-"^30)
        
        for result in validation_results
            @printf(io, "%-9d  %-7.4f %-11.4f\n", 
                    result.component, result.r2, result.cor)
        end
        
        println(io, "")
        println(io, "Best performance by metric:")
        
        # Find best component for each metric
        best_r2_idx = argmax([r.r2 for r in validation_results])
        best_cor_idx = argmax([r.cor for r in validation_results])
        
        println(io, "  Highest R²: Component $(validation_results[best_r2_idx].component) (R² = $(round(validation_results[best_r2_idx].r2, digits=4)))")
        println(io, "  Highest Correlation: Component $(validation_results[best_cor_idx].component) (r = $(round(validation_results[best_cor_idx].cor, digits=4)))")
    end
    
    println("Validation results saved to: $results_file")
    
    # Print summary statistics for this radiation type
    println("\nSummary for $short_name radiation:")
    println("  Best R²: $(round(maximum([r.r2 for r in validation_results]), digits=4))")
    println("  Best Correlation: $(round(maximum([r.cor for r in validation_results]), digits=4))")
end

println("\n" * "="^60)
println("Atmospheric CERES PLS Analysis Complete!")
println("="^60)
println("Results saved to: $visdir")
