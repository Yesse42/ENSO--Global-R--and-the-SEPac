# Note: in_time_period is also defined in load_funcs.jl — keep in sync if changed.
in_time_period(time, time_period) = time_period[1] <= time < time_period[2]

function filter_time_period(data, times, time_period, t_idx)
    valid_idxs = findall(t -> in_time_period(t, time_period), times)
    filter_tuple = ntuple(i -> i == t_idx ? valid_idxs : Colon(), length(size(data)))
    return data[filter_tuple...], times[valid_idxs]
end

calc_float_time(date_time) = year(date_time) + (month(date_time) - 1) / 12

function round_dates_down_to_nearest_month(dates)
    return Date.(year.(dates), month.(dates), 1)
end

function time_lag(data, lag)
    lag = -lag
    T = eltype(data)
    lagged_data = convert(Vector{Union{T, Missing}}, copy(data))
    if lag == 0
        lagged_data
    elseif lag > 0
        @views lagged_data[lag+1:end] .= data[1:end-lag]
        lagged_data[1:lag] .= missing
        lagged_data
    elseif lag < 0
        @views lagged_data[1:end+lag] .= data[-lag+1:end]
        lagged_data[end+lag+1:end] .= missing
        lagged_data
    end
end

"""
    calculate_lag_correlations(reference_data, lagged_data_dict; lags=-24:24)

Calculate correlation as a function of lag between reference data and lagged variables.

# Arguments
- `reference_data`: Vector of reference values (e.g., radiation data)
- `lagged_data_dict`: Dictionary mapping lag values to corresponding lagged data vectors
- `lags`: Range of lags to calculate correlations for (default: -24:24)

# Returns
- `Dict{Int, Float64}`: Sorted dictionary mapping lags to correlation values
"""

using StatsBase

function calculate_lag_func(reference_data, lagged_data_dict; lags, func)
    correlations = Dict{Int, Float64}()
    for lag in lags
        if haskey(lagged_data_dict, lag)
            lagged_data = lagged_data_dict[lag]
            valid_indices = .!(ismissing.(reference_data) .| ismissing.(lagged_data))
            if sum(valid_indices) > 0
                correlations[lag] = func(reference_data[valid_indices], lagged_data[valid_indices])
            else
                correlations[lag] = NaN
            end
        else
            correlations[lag] = NaN
        end
    end
    return Dictionary(lags, [correlations[lag] for lag in lags])
end

function calculate_lag_correlations(reference_data, lagged_data_dict; lags)
    return calculate_lag_func(reference_data, lagged_data_dict; lags, func = cor)
end

# Apply time_lag to a single vector y then evaluate func at each lag
function calculate_lag_func_lag_in_place(x, y, lags; func)
    lagged_data_dict = Dict(lag => time_lag(y, lag) for lag in lags)
    return calculate_lag_func(x, lagged_data_dict; lags, func)
end

function calculate_lag_corrs_lag_in_place(unlagged_data, lagged_data, lags)
    return calculate_lag_func_lag_in_place(unlagged_data, lagged_data, lags; func = cor)
end
