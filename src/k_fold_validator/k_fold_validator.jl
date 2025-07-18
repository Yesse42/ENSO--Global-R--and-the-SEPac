dir = pwd()
cd(@__DIR__)
include("../pls_regressor/pls_functions.jl")
cd(dir)

using StatsBase, Random, Dictionaries

function generate_contiguous_folds(n, n_folds)
    fold_size = div(n, n_folds)
    remainder = n % n_folds
    fold_sizes = fill(fold_size, n_folds)
    for i in 1:remainder
        fold_sizes[i] += 1
    end
    fold_ends = cumsum(fold_sizes)
    fold_begins = cumsum([1; fold_sizes[1:end-1]])
    return [fold_begin:fold_end for (fold_begin, fold_end) in zip(fold_begins, fold_ends)]
end

function generate_random_folds(n, n_folds)
    indices = shuffle(1:n)
    fold_size = fld(n, n_folds)
    folds = [indices[(i-1)*fold_size+1:min(i*fold_size, n)] for i in 1:n_folds]
    remaining_indices = indices[(min(n, n_folds * fold_size) + 1):end]
    # Handle the case where n is not perfectly divisible by n_folds
    if !isempty(remaining_indices)
        for i in axes(remaining_indices,1)
            push!(folds[(i-1)%n_folds + 1], remaining_indices[i])
        end
    end
    return folds
end

function make_out_of_sample_data(X, Y, predictorfunc, folds)
    n = size(X, 1)
    out_of_fold = [setdiff(1:n, fold) for fold in folds]
    return [begin
        predictorfunc(X[out_fold, :], Y[out_fold, :], X[in_fold, :])
    end for (in_fold, out_fold) in zip(folds, out_of_fold)]
end

function PLS_Predictor_Func(X_train, Y_train, X_test; n_components=2)
    model = make_pls_regressor(X_train, Y_train, n_components)
    return predict(model, X_test)
end

function k_fold_validate_pls(X, Y; n_folds=size(X, 1), contiguous=true, n_components=2)
    n = size(X, 1)
    folds = if contiguous
        generate_contiguous_folds(n, n_folds)
    else
        generate_random_folds(n, n_folds)
    end

    predictorfunc = (X_train, Y_train, X_test) -> PLS_Predictor_Func(X_train, Y_train, X_test; n_components=n_components)

    out_of_sample_data = make_out_of_sample_data(X, Y, predictorfunc, folds)

    return (folds, out_of_sample_data)
end

function untangle_folds(folds, out_of_sample_data)
    folds = reduce(vcat, folds)
    out_of_sample_data = reduce(vcat, out_of_sample_data)
    folds_perm = sortperm(folds)
    out_of_sample_sorted = out_of_sample_data[folds_perm, :]
    return out_of_sample_sorted
end

function untangle_multi_component_folds(folds, out_of_sample_data_dicts, n_components)
    folds_concatenated = reduce(vcat, folds)
    folds_perm = sortperm(folds_concatenated)
    untangled_by_component = [begin
        out_of_sample_data = reduce(vcat, [out_of_sample_data_dict[component] for out_of_sample_data_dict in out_of_sample_data_dicts])
        out_of_sample_data[folds_perm, :]
    end for component in n_components]
    return Dictionary(n_components, untangled_by_component)
end

"For each individual fold, fit a PLS regressor and return the out-of-sample predictions for all components"
function generate_out_of_sample_predictions_all_components(X, Y, in_fold, out_of_fold, n_components)
    GC.gc()  # Force garbage collection to manage memory usage
    train_X = X[out_of_fold, :]
    train_Y = Y[out_of_fold, :]
    max_components = maximum(n_components)

    regressor = make_pls_regressor(train_X, train_Y, max_components; make_copies = NoCopy(), print_updates = true)
    eval_X = X[in_fold, :]
    predictions = Dictionary(n_components, [predict(regressor, eval_X; components=1:component) for component in n_components])
    return predictions
end


function k_fold_validate_pls_regressor_simultaneously(X, Y; 
    n_folds = size(X, 1), fold_generator = generate_contiguous_folds,
    n_components = 1:8)

    n = size(X, 1)
    folds = fold_generator(size(X, 1), n_folds)
    out_of_folds = [setdiff(1:n, fold) for fold in folds]
    out_of_sample_data = [generate_out_of_sample_predictions_all_components(X, Y, fold, out_fold, n_components) 
                          for (fold, out_fold) in zip(folds, out_of_folds)]
    untangled_data_by_component = untangle_multi_component_folds(folds, out_of_sample_data, n_components)
    return untangled_data_by_component, folds
end

function test_on_perfect_linear_data()
    # Generate perfectly collinear data
    n = 200
    n_features = 10
    n_folds = 20
    
    # Create random coefficients for the linear relationship
    true_coeffs = randn(n_features, n_features)
    
    # Generate random X data
    X = randn(n, n_features)
    
    # Generate Y as a perfect linear combination of X
    Y = X * true_coeffs
    
    # Run k-fold validation
    predictions_by_components, folds = k_fold_validate_pls_regressor_simultaneously(
        X, Y, n_folds=n_folds, n_components=1:n_features)
    
    # Get predictions for component 1 (should be sufficient for perfect linear data)
    predictions = predictions_by_components[n_features]
    
    # Plot results
    
    p = scatter(Y, predictions, 
        xlabel="True Values", 
        ylabel="Out-of-Sample Predictions",
        title="K-Fold Validation Results (n=$n, folds=$n_folds)",
        label="",
        alpha=0.6)
    
    # Add perfect prediction line
    min_val = minimum([minimum(Y), minimum(predictions)])
    max_val = maximum([maximum(Y), maximum(predictions)])
    plot!(p, [min_val, max_val], [min_val, max_val], 
        label="Perfect Prediction", 
        color=:red, 
        linestyle=:dash)
    
    # Calculate R²
    mean_y = mean(Y)
    ss_tot = sum((Y .- mean_y).^2)
    ss_res = sum((Y .- predictions).^2)
    r_squared = 1 - ss_res/ss_tot
    
    # Add R² text to plot
    annotate!(p, [(min_val + 0.1*(max_val-min_val), max_val - 0.1*(max_val-min_val), 
                 text("R² = $(round(r_squared, digits=4))", 10, :left))])
    
    return p, predictions, Y, r_squared
end

# Run the test
if false
    using Plots
    p, predictions, true_values, r_squared = test_on_perfect_linear_data()
    display(p)
    println("R² value: $r_squared")
end
