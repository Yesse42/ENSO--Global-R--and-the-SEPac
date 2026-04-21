"Precompute the matching indices between data times and ENSO times, and the ENSO lag matrix.
Returns (enso_matrix, data_idx, enso_idx) where data_idx and enso_idx index into data_times and enso_times respectively."
function precompute_enso_match(data_times, enso_times, enso_lag_matrix)
    enso_time_set = Dict(t => i for (i, t) in enumerate(enso_times))
    data_idx = Int[]
    enso_idx = Int[]
    for (i, t) in enumerate(data_times)
        j = get(enso_time_set, t, nothing)
        if j !== nothing
            push!(data_idx, i)
            push!(enso_idx, j)
        end
    end
    return enso_lag_matrix[enso_idx, :], data_idx
end

"Remove ENSO signal from y using a precomputed ENSO lag matrix via 1-component PLS. Returns the residual."
function remove_enso_via_pls(enso_matrix::Matrix, y::AbstractVector; verbose = false, label = "")
    pls_model = make_pls_regressor(enso_matrix, y, 1; print_updates=false)
    predicted_y = vec(predict(pls_model, enso_matrix))
    pls_corr = cor(predicted_y, y)
    verbose && println("PLS correlation for $label: $pls_corr")
    return y .- predicted_y
end
