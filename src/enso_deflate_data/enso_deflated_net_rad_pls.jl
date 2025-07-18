using Plots, Statistics, StatsBase, Dates, SplitApplyCombine, Printf, Dictionaries, JLD2

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
Calculate a 10-component PLS regression model of z, t, t2m, and msl predicting net rad, sw, and lw from ceres, with the ENSO effect regressed out
It will then perform a 10-fold cross validation of the model and plot the results
"""

visdir = "../../vis/deflated_enso_pls/"
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

datapath = "../../data/ENSO_Deflated/era5_ceres_enso_deflated.nc"

predictor_names = "deflated_" .* ["t2m", "t", "msl", "z"]
ndatadims = [3, 4, 3, 4]
selectors = [ntuple(n ->  Colon(), nd) for nd in ndatadims]
short_rad_names = ["net", "sw", "lw"]
predictand_names = "deflated_gtoa_" .* short_rad_names .* "_all_mon"

dataset = NCDataset(datapath, "r")

predictors = Dictionary(predictor_names, [dataset[var_name][selector...] for (var_name, selector) in zip(predictor_names, selectors)])
predictands = Dictionary(predictand_names, [dataset[var_name][:] for (var_name) in predictand_names])

lat = dataset["latitude"][:]
lon = dataset["longitude"][:]
p_levels = dataset["pressure_level"][:]

close(dataset)

X, predictor_idxs, predictor_shapes = reshape_and_concatenate(predictors, predictor_names, ndatadims)

for (short_name, predictand_name) in zip(short_rad_names, predictand_names)
    println("Predictand: $predictand_name")
    Y = predictands[predictand_name]
    
    println("Running PLS regression with $max_component components...")
    pls = make_pls_regressor(X, Y, max_component)
    
    # Save PLS model and metadata
    pls_filename = joinpath(plsdir, "enso_deflated_$(short_name)_pls.jld2")
    println("Saving PLS model to: $pls_filename")
    
    save(pls_filename, Dict(
        "pls_model" => reduce_pls_model(pls; copy=false),
        "predictor_idxs" => predictor_idxs,
        "predictor_shapes" => predictor_shapes,
        "predictor_names" => predictor_names,
        "predictand_name" => predictand_name,
        "short_name" => short_name,
        "ndatadims" => ndatadims,
        "coordinates" => Dict(
            "latitude" => lat,
            "longitude" => lon,
            "pressure_level" => p_levels
        ),
        "analysis_info" => Dict(
            "datapath" => datapath,
            "max_components" => max_component,
            "creation_date" => string(now())
        )
    ))

    #Now plot the results for 
    plotpath = joinpath(visdir, short_name * "_pls_enso_deflated")

    for component in plot_components
        raw_matrix = make_matrix_to_multiply_by_X_to_get_Y(pls; components=1:component)
        matrices_with_vars = reconstruct_spatial_arrays(raw_matrix, predictor_idxs, predictor_shapes)

        for (varname, mat) in pairs(matrices_with_vars)
            println("Plotting component $component for variable $varname")
            savedir = joinpath(plotpath, varname)
            !isdir(savedir) && mkpath(savedir)
            ndim_mat = ndims(mat)
            if ndim_mat == 3
                for (p_idx, p_level) in enumerate(p_levels)
                    fig = plot_global_heatmap(lat, lon, mat[:, :, p_idx]; title = "PLS #$component for $varname at $p_level hPa predicting $predictand_name", colorbar_label = "PLS Weight")
                    fig.savefig(joinpath(savedir, "component$(component)_level$(p_level)hPa.png"))
                    plt.close()
                end
            elseif ndim_mat == 2
                fig = plot_global_heatmap(lat, lon, mat; title = "PLS #$component for $varname predicting $predictand_name", colorbar_label = "PLS Weight")
                fig.savefig(joinpath(savedir, "component$(component).png"))
                plt.close()
            end
        end
    end

    #Now perform the k-fold validation
    n_folds = 10
    n_components = 1:10
    
    println("Running k-fold validation with $n_folds folds for components $n_components...")
    predictions_by_components, folds = k_fold_validate_pls_regressor_simultaneously(
        X, Y, n_folds=n_folds, n_components=n_components)
    
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
        println(io, "Number of folds: $n_folds")
        println(io, "Data points: $(size(X, 1))")
        println(io, "Features: $(size(X, 2))")
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
end

