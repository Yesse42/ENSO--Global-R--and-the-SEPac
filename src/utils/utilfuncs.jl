using SplitApplyCombine, Dates, Statistics, StatsBase, NCDatasets, Dictionaries

function theil_sen_fit(x, y)
    n = length(x)
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in 1:n for j in 1:i-1 if x[j] != x[i]]
    slope = median!(slopes)
    if ismissing(slope)
        slope = NaN
    end
    return (;slope, intercept = median!(y .- slope .* x))
end

function least_squares_fit(x, y)
    n = length(x)
    A = hcat(x, ones(n))
    sol =  A \ y
    return (;slope = sol[1], intercept = sol[2])
end

get_lsq_slope(x, y) = least_squares_fit(x, y).slope

function detrend!(slice, times, slope, intercept)
    @. slice -= (slope * times + intercept)
    return slice
end

function retrend!(slice, times, slope, intercept)
    @. slice += (slope * times + intercept)
    return slice
end

"Get the indices associated with each month, sorted"
function get_seasonal_cycle(months)
    season_idxs = sortkeys!(groupfind(months))
end

function aggregate_by_month(data, months)
    # Get the indices associated with each month
    season_idxs = get_seasonal_cycle(months)
    # Calculate the mean for each month
    means = map(season_idxs) do idxs
        mean(data[idxs])
    end
    return means
end

function calc_seasonal_cycle(data, months; aggfunc = mean)
    # Get the indices associated with each month
    season_idxs = get_seasonal_cycle(months)
    # Calculate the mean for each month
    season_stats = map(season_idxs) do idxs
        aggfunc(data[idxs])
    end
    return season_stats
end

const monthlengths = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
weighted_average_monthly_stats(month_stats) = mean(month_stats, weights(month_lengths))

function deseasonalize!(slice, idx_groups, idx_means)
    for (idxs, mean) in zip(idx_groups, idx_means)
        @. slice[idxs] -= mean
    end
    return slice
end

function reseasonalize(slice, idx_groups, idx_means)
    for (idxs, mean) in zip(idx_groups, idx_means)
        @. slice[idxs] += mean
    end
    return slice
end

function deseasonalize_strongly_trended_data!(slice, times; monthfunc = month, aggfunc = mean, trendfunc = least_squares_fit)
    # Copy the original slice
    slice_copy = copy(slice)
    # Fit the trend
    float_times = @. year(times) + (month(times)-1) / 12
    fit = trendfunc(float_times, slice_copy)
    # Detrend the copy
    detrend!(slice_copy, float_times, fit.slope, fit.intercept)
    #Add the mean back to the copy
    slice_copy .+= aggfunc(slice)
    # Get month indices
    months = monthfunc.(times)
    idx_groups = get_seasonal_cycle(months)
    idx_means = map(idx_groups) do idxs
        aggfunc(slice_copy[idxs])
    end
    # Deseasonalize the original slice
    deseasonalize!(slice, idx_groups, idx_means)
    return slice
end

function detrend_and_deseasonalize_precalculated_groups!(slice, float_times, idx_groups; aggfunc = mean, trendfunc = least_squares_fit)
    # Fit the trend
    fit = trendfunc(float_times, slice)
    # Detrend the copy
    detrend!(slice, float_times, fit.slope, fit.intercept)
    # Calculate the means for each month
    for idx_group in idx_groups
        mean_val = aggfunc(slice[idx] for idx in idx_group)
        @. slice[idx_group] -= mean_val
    end
end

function detrend_and_deseasonalize!(slice, float_times, months; aggfunc = mean, trendfunc = least_squares_fit)
    # Fit the trend
    fit = trendfunc(float_times, slice)
    # Detrend the copy
    detrend!(slice, float_times, fit.slope, fit.intercept)
    # Get month indices
    idx_groups = get_seasonal_cycle(months)
    idx_means = map(idx_groups) do idxs
        aggfunc(slice[idxs])
    end
    # Deseasonalize the original slice
    deseasonalize!(slice, idx_groups, idx_means)
    return (slice, fit)
end

in_time_period(time, time_period) = time_period[1] <= time < time_period[2]

function filter_time_period(data, times, time_period, t_idx)
    valid_idxs = findall(t -> in_time_period(t, time_period), times)
    filter_tuple = ntuple(i -> i == t_idx ? valid_idxs : Colon(), length(size(data)))
    return data[filter_tuple...], times[valid_idxs]
end

calc_float_time(date_time) = year(date_time) + (month(date_time) - 1) / 12

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