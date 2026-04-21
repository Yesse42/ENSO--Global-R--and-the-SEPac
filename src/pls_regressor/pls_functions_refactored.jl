using LinearAlgebra, StatsBase, Arpack
olddir = pwd()
cd(@__DIR__)
include("pls_structs.jl")
cd(olddir)

"A special standard deviation fynction which assumes 0 means and uses a fixed dimension for type stability"
my_std_func(X; kwargs...) = [sqrt(slice ⋅ slice / (length(slice) - 1)) for slice in eachcol(X)]'

"Normalizes the input; also returns the data necessary to denormalize the output."
function normalize_input!(X; standardize, already_centered)
    if !already_centered
        means = mean(X; dims=1)
        #Normalize the input
        X .-= means
    else
        means = zeros(eltype(X), 1, size(X, 2))
    end

    if standardize
        stds = my_std_func(X; dims = (1,))
        X ./= stds
    else
        stds = ones(eltype(X), 1, size(X, 2))
    end

    return X, means, stds
end

function normalize_input_precalculated!(X, means, stds; standardize, already_centered)
    #Normalize the input
    if !already_centered 
        X .-= means
    end
    if standardize
        X ./= stds
    end
    return X
end

"Turns the input or output back into the original scale"
function denormalize_output!(X, means, stds; standardize, already_centered)
    if standardize
        X .*= stds
    end
    if !already_centered
        X .+= means
    end
    return X
end

function calculate_single_pls_mode!(X, Y, temp_cov_mat, 
                        X_weights_view, Y_weights_view, 
                        X_scores_view, Y_scores_view,
                        X_loadings_view, Y_loadings_view)
    #Get the weights
    C = mul!(temp_cov_mat, X', Y)
    if size(Y, 2) == 1
        u = C./= norm(C)
        vt = [1.0]'
    else
        svd_result = svds(C; nsv=1)[1]
        u = svd_result.U
        vt = svd_result.Vt
    end

    #Get the weights
    X_weights_view .= u
    Y_weights_view .= vt'

    #Calculate the X and Y scores
    mul!(X_scores_view, X, X_weights_view)
    mul!(Y_scores_view, Y, Y_weights_view)

    #Calculate the X and Y loadings
    inv_norm2_X_scores = 1 / (X_scores_view ⋅ X_scores_view)
    mul!(X_loadings_view, X', X_scores_view, inv_norm2_X_scores, 0)
    mul!(Y_loadings_view, Y', X_scores_view, inv_norm2_X_scores, 0)

    #Deflate X and Y
    mul!(X, X_scores_view, X_loadings_view', -1, 1)
    mul!(Y, X_scores_view, Y_loadings_view', -1, 1)
end

"Calculate all PLS components"
function calculate_pls_model(X::AbstractMatrix{T}, Y, n_components::Int; print_updates) where T
    #Check that the number of components is not larger than the number of features
    if n_components > size(X, 1) || n_components > size(X, 2)
        throw(ArgumentError("Number of components cannot be larger than the number of features in X or the number of samples."))
    end

    Y = @view Y[:,:]  # Ensure Y is a matrix for consistency

    #Initialize the weights and loadings
    X_weights = zeros(T, size(X, 2), n_components)
    Y_weights = zeros(T, size(Y, 2), n_components)
    X_scores = zeros(T, size(X, 1), n_components)
    Y_scores = zeros(T, size(Y, 1), n_components)
    X_loadings = zeros(T, size(X, 2), n_components)
    Y_loadings = zeros(T, size(Y, 2), n_components)

    #Temporary covariance matrix
    temp_cov_mat = zeros(size(X, 2), size(Y, 2))

    for i in 1:n_components
        #Create views for the current component
        X_weights_view = @view X_weights[:, i]
        Y_weights_view = @view Y_weights[:, i]
        X_scores_view = @view X_scores[:, i]
        Y_scores_view = @view Y_scores[:, i]
        X_loadings_view = @view X_loadings[:, i]
        Y_loadings_view = @view Y_loadings[:, i]

        #Calculate the PLS mode
        calculate_single_pls_mode!(X, Y, temp_cov_mat, 
                                   X_weights_view, Y_weights_view, 
                                   X_scores_view, Y_scores_view,
                                   X_loadings_view, Y_loadings_view)

        #Print updates if requested
        print_updates && println("Calculated component $i of $n_components")
    end

    return X_weights, Y_weights, X_scores, Y_scores, X_loadings, Y_loadings
end

"Actually make the struct"
function make_pls_regressor(X, Y, n_components::Int; standardize = true, already_centered = false, make_copies::ToCopy = MakeCopy(), print_updates::Bool = false, trash...)

    if make_copies isa MakeCopy
        print_updates && println("Copying data")
        X = copy(X)
        Y = copy(Y)
    end

    print_updates && println("Normalizing data")

    #Normalize the input
    X, X_means, X_stds = normalize_input!(X; standardize, already_centered)
    Y, Y_means, Y_stds = normalize_input!(Y; standardize, already_centered)


    #Calculate the PLS model
    X_weights, Y_weights, X_scores, Y_scores, X_loadings, Y_loadings = calculate_pls_model(X, Y, n_components; print_updates)

    #Return the PLS regressor struct
    return PLSRegressor(X, Y, n_components,
                        X_weights, Y_weights,
                        X_scores, Y_scores,
                        X_loadings, Y_loadings,
                        X_means, Y_means,
                        X_stds, Y_stds,
                        already_centered, standardize)
end

"This function makes an approximation of Y for a given X"
function predict(pls, X_new; components = Colon())
    X_new = copy(X_new)  # Ensure we work with a copy of the input
    # Normalize the new input
    X_new = normalize_input_precalculated!(X_new, pls.X_means, pls.X_stds; standardize=pls.normalize, already_centered=pls.already_centered)

    # Calculate the predicted Y
    P = @views pls.X_weights[:, components] * pinv(pls.X_loadings[:, components]' * pls.X_weights[:, components])
    coeffs =  pls.Y_stds .* @views(P * pls.Y_loadings[:, components]' ) ./ pls.X_stds'

    # Denormalize the output
    Y_pred = X_new * coeffs .+ pls.Y_means

    return Y_pred
end

function make_matrix_to_multiply_by_X_to_get_Y(pls; components = Colon())
    # Create the matrix to multiply by X to get Y
    P = @views pls.X_weights[:, components] * pinv(pls.X_loadings[:, components]' * pls.X_weights[:, components])
    coeffs = @views P * pls.Y_loadings[:, components]'
    return coeffs
end

function make_P_matrix(pls; components = Colon())
    P = @views pls.X_weights[:, components] * pinv(pls.X_loadings[:, components]' * pls.X_weights[:, components])
    return P
end

function make_matrix_to_get_covarying_pattern_in_Y(pls; components = Colon())
    P = @views pls.Y_weights[:, components] * pinv(pls.Y_loadings[:, components]' * pls.Y_weights[:, components])
    covarying_pattern = @views P * pls.X_loadings[:, components]'
    return covarying_pattern
end

function create_curried_fill(value)
    return function(arr; dims)
        new_shape = ntuple(i -> if i in dims 1 else size(arr, i) end, ndims(arr))
        return fill(value, new_shape)
    end
end

const nostd = create_curried_fill(1.0)
const nomean = create_curried_fill(0.0)

function get_pls_prediction_slice(X, slice; n_component, meanfunc, stdfunc)
    pls_model = make_pls_regressor(X, slice, n_component; print_updates=false, meanfunc, stdfunc)
    pred_slice = vec(predict(pls_model, X))
    return pred_slice
end

function pointwise_pls(X, Y;n_components = 3, slice_dims = (1,2), meanfunc = mean, stdfunc = my_std_func)
    pls_preds = similar(Y)
    for (save_slice, calc_slice) in zip(eachslice(pls_preds, dims=slice_dims), eachslice(Y, dims=slice_dims))
        pred_slice = get_pls_prediction_slice(X, calc_slice; n_component=n_components, meanfunc, stdfunc)
        save_slice .= pred_slice
    end
    return pls_preds
end