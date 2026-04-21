weighted_mean(values, weights) = (values ⋅ weights) / sum(weights)

function circular_rollmean(data::Vector{T}, weights, window::Int) where T <: Real
    prepend_len = floor(Int, (window - 1) / 2)
    append_len = ceil(Int, (window - 1) / 2)
    padded_data = vcat(data[end-prepend_len+1:end], data, data[1:append_len])
    padded_weights = vcat(weights[end-prepend_len+1:end], weights, weights[1:append_len])
    result = rolling(weighted_mean, padded_data, padded_weights, window)
    return result
end

precalculate_daily_data_groups(times) = sortkeys!(groupfind(tuple.(month.(times), day.(times))))

function deseasonalize_daily_data!(slice, times; aggfunc = mean, windowsize = 29)
    month_day_groups = sortkeys!(groupfind(tuple.(month.(times), day.(times))))
    if length(month_day_groups) < 365
        error("Not all month-day combinations are present in the data.")
    end
    month_day_aggs = map(month_day_groups) do idxs
        aggfunc(slice[idxs])
    end
    n_obs_per_day = length.(month_day_groups)
    smoothed_month_day_aggs = circular_rollmean(collect(month_day_aggs), collect(n_obs_per_day), windowsize)
    for (idxs, smoothed_val) in zip(month_day_groups, smoothed_month_day_aggs)
        @. slice[idxs] -= smoothed_val
    end
    return slice
end

function deseasonalize_daily_data!(slice, month_day_groups, smoothed_month_day_aggs)
    for (idxs, smoothed_val) in zip(month_day_groups, smoothed_month_day_aggs)
        @. slice[idxs] -= smoothed_val
    end
    return slice
end

function detrend_and_deseasonalize_daily_data!(slice, float_times, times; aggfunc = mean, windowsize = 29, trendfunc = least_squares_fit)
    fit = trendfunc(float_times, slice)
    detrend!(slice, float_times, fit.slope, fit.intercept)
    deseasonalize_daily_data!(slice, times; aggfunc, windowsize)
    return nothing
end

function detrend_and_deseasonalize_daily_data_precalculated_groups!(slice, float_times, month_day_groups; aggfunc = mean, windowsize = 29, trendfunc = least_squares_fit)
    fit = trendfunc(float_times, slice)
    detrend!(slice, float_times, fit.slope, fit.intercept)
    month_day_aggs = map(month_day_groups) do idxs
        aggfunc(slice[idxs])
    end
    n_obs_per_day = length.(month_day_groups)
    smoothed_month_day_aggs = circular_rollmean(collect(month_day_aggs), collect(n_obs_per_day), windowsize)
    deseasonalize_daily_data!(slice, month_day_groups, smoothed_month_day_aggs)
    return nothing
end

function detrend_and_deseasonalize_daily_data_precalculated_groups_twice!(slice, float_times, month_day_groups; aggfunc = mean, windowsize = 29, trendfunc = least_squares_fit)
    for _ in 1:2
        detrend_and_deseasonalize_daily_data_precalculated_groups!(slice, float_times, month_day_groups; aggfunc, windowsize, trendfunc)
    end
    return nothing
end

function deseasonalize_and_detrend_daily_data!(slice, float_times, times; aggfunc = mean, windowsize = 29, trendfunc = least_squares_fit)
    deseasonalize_daily_data!(slice, times; aggfunc, windowsize)
    fit = trendfunc(float_times, slice)
    detrend!(slice, float_times, fit.slope, fit.intercept)
    return nothing
end

function deseasonalize_and_detrend_daily_data_precalculated_groups!(slice, float_times, month_day_groups; aggfunc = mean, windowsize = 29, trendfunc = least_squares_fit)
    month_day_aggs = map(month_day_groups) do idxs
        aggfunc(slice[idxs])
    end
    n_obs_per_day = length.(month_day_groups)
    smoothed_month_day_aggs = circular_rollmean(collect(month_day_aggs), collect(n_obs_per_day), windowsize)
    deseasonalize_daily_data!(slice, month_day_groups, smoothed_month_day_aggs)
    fit = trendfunc(float_times, slice)
    detrend!(slice, float_times, fit.slope, fit.intercept)
    return nothing
end

function deseasonalize_and_detrend_daily_data_precalculated_groups_twice!(slice, float_times, month_day_groups; aggfunc = mean, windowsize = 29, trendfunc = least_squares_fit)
    for _ in 1:2
        deseasonalize_and_detrend_daily_data_precalculated_groups!(slice, float_times, month_day_groups; aggfunc, windowsize, trendfunc)
    end
    return nothing
end
