"Get the indices associated with each month, sorted"
function get_seasonal_cycle(months)
    sortkeys!(groupfind(months))
end

function aggregate_by_month(data, months)
    season_idxs = get_seasonal_cycle(months)
    means = map(season_idxs) do idxs
        mean(data[idxs])
    end
    return means
end

function calc_seasonal_cycle(data, months; aggfunc = mean)
    season_idxs = get_seasonal_cycle(months)
    season_stats = map(season_idxs) do idxs
        aggfunc(data[idxs])
    end
    return season_stats
end

const monthlengths = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
weighted_average_monthly_stats(month_stats) = mean(month_stats, weights(monthlengths))

function deseasonalize!(slice, idx_groups, idx_means)
    for (idxs, mean) in zip(idx_groups, idx_means)
        @. slice[idxs] -= mean
    end
    return slice
end

function deseasonalize!(slice, months)
    idx_groups = get_seasonal_cycle(months)
    for idxs in idx_groups
        mean_val = mean(slice[idxs])
        @. slice[idxs] -= mean_val
    end
    return slice
end

function reseasonalize(slice, idx_groups, idx_means)
    for (idxs, mean) in zip(idx_groups, idx_means)
        @. slice[idxs] += mean
    end
    return slice
end

function deseasonalize_precalculated_groups!(slice, idx_groups; aggfunc = mean)
    for idx_group in idx_groups
        mean_val = aggfunc(slice[idx_group])
        @. slice[idx_group] -= mean_val
    end
    return slice
end

function deseasonalize_strongly_trended_data!(slice, times; monthfunc = month, aggfunc = mean, trendfunc = least_squares_fit)
    slice_copy = copy(slice)
    float_times = @. year(times) + (month(times)-1) / 12
    fit = trendfunc(float_times, slice_copy)
    detrend!(slice_copy, float_times, fit.slope, fit.intercept)
    slice_copy .+= aggfunc(slice)
    months = monthfunc.(times)
    idx_groups = get_seasonal_cycle(months)
    idx_means = map(idx_groups) do idxs
        aggfunc(slice_copy[idxs])
    end
    deseasonalize!(slice, idx_groups, idx_means)
    return nothing
end
