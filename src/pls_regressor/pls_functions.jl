using LinearAlgebra, StatsBase, Arpack
olddir = pwd()
cd(@__DIR__)
include("pls_structs.jl")
cd(olddir)

"Normalizes the input; also returns the data necessary to denormalize the output."
function normalize_input!(X, meanfunc = mean, stdfunc = (x...;kwargs...) -> 1)
    means = meanfunc(X; dims=1)
    #Normalize the input
    X .-= means

    stds = stdfunc(X; dims = (1,))
    X ./= stds

    return X, means, stds
end

function normalize_input_precalculated!(X, means, stds)
    #Normalize the input
    X .-= means
    X ./= stds
    return X
end

"Turns the input or output back into the original scale"
function denormalize_output!(X, means, stds)
    X .*= stds
    X .+= means
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

"A special stabdard deviation fynction which assumes 0 means and uses a fixed dimension for type satbility"
my_std_func(X; kwargs...) = [sqrt(slice ⋅ slice / (length(slice) - 1)) for slice in eachcol(X)]'

"Actually make the struct"
function make_pls_regressor(X, Y, n_components::Int; meanfunc = mean, stdfunc = my_std_func, make_copies::ToCopy = MakeCopy(), print_updates::Bool = true)

    if make_copies isa MakeCopy
        print_updates && println("Copying data")
        X = copy(X)
        Y = copy(Y)
    end

    print_updates && println("Normalizing data")

    #Normalize the input
    X, X_means, X_stds = normalize_input!(X, meanfunc, stdfunc)
    Y, Y_means, Y_stds = normalize_input!(Y, meanfunc, stdfunc)


    #Calculate the PLS model
    X_weights, Y_weights, X_scores, Y_scores, X_loadings, Y_loadings = calculate_pls_model(X, Y, n_components; print_updates)

    #Return the PLS regressor struct
    return PLSRegressor(X, Y, n_components,
                        X_weights, Y_weights,
                        X_scores, Y_scores,
                        X_loadings, Y_loadings,
                        X_means, Y_means,
                        X_stds, Y_stds)
end

"This function makes an approximation of Y for a given X"
function predict(pls, X_new; components = Colon())
    X_new = copy(X_new)  # Ensure we work with a copy of the input
    # Normalize the new input
    X_new = normalize_input_precalculated!(X_new, pls.X_means, pls.X_stds)

    # Calculate the predicted Y
    P = @views pls.X_weights[:, components] * pinv(pls.X_loadings[:, components]' * pls.X_weights[:, components])
    coeffs = @views P * pls.Y_loadings[:, components]'

    # Denormalize the output
    Y_pred = denormalize_output!(X_new * coeffs, pls.Y_means, pls.Y_stds)

    return Y_pred
end

function make_matrix_to_multiply_by_X_to_get_Y(pls; components = Colon())
    # Create the matrix to multiply by X to get Y
    P = @views pls.X_weights[:, components] * pinv(pls.X_loadings[:, components]' * pls.X_weights[:, components])
    coeffs = @views P * pls.Y_loadings[:, components]'
    return coeffs
end