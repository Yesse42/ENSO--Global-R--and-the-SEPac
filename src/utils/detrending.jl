function detrend!(slice, times; trendfunc = least_squares_fit)
    fit = trendfunc(times, slice)
    @. slice -= (fit.slope * times + fit.intercept)
    return slice
end

function detrend!(slice, times, slope, intercept)
    @. slice -= (slope * times + intercept)
    return slice
end

function retrend!(slice, times, slope, intercept)
    @. slice += (slope * times + intercept)
    return slice
end

function detrend_each_season_individually!(slice, float_times, month_groups; trendfunc = least_squares_fit)
    for idx_group in month_groups
        times = float_times[idx_group]
        slice_view = @view slice[idx_group]
        fit = trendfunc(times, slice_view)
        detrend!(slice_view, times, fit.slope, fit.intercept)
    end
    return nothing
end

function detrend_and_deseasonalize_precalculated_groups!(slice, float_times, idx_groups; aggfunc = mean, trendfunc = least_squares_fit)
    fit = trendfunc(float_times, slice)
    detrend!(slice, float_times, fit.slope, fit.intercept)
    for idx_group in idx_groups
        mean_val = aggfunc(slice[idx] for idx in idx_group)
        @. slice[idx_group] -= mean_val
    end
    return nothing
end

function deseasonalize_and_detrend_precalculated_groups!(slice, float_times, idx_groups; aggfunc = mean, trendfunc = least_squares_fit)
    for idx_group in idx_groups
        mean_val = aggfunc(slice[idx] for idx in idx_group)
        @. slice[idx_group] -= mean_val
    end
    fit = trendfunc(float_times, slice)
    detrend!(slice, float_times, fit.slope, fit.intercept)
    return nothing
end

deseasonalize_and_detrend_precalculated_groups_twice!(slice, float_times, idx_groups; aggfunc = mean, trendfunc = least_squares_fit) = for _ in 1:2
    deseasonalize_and_detrend_precalculated_groups!(slice, float_times, idx_groups; aggfunc, trendfunc)
    nothing
end

function detrend_and_deseasonalize!(slice, float_times, months; aggfunc = mean, trendfunc = least_squares_fit)
    fit = trendfunc(float_times, slice)
    detrend!(slice, float_times, fit.slope, fit.intercept)
    idx_groups = get_seasonal_cycle(months)
    idx_means = map(idx_groups) do idxs
        aggfunc(slice[idxs])
    end
    for (idxs, mean) in zip(idx_groups, idx_means)
        @. slice[idxs] -= mean
    end
    return (slice, fit)
end

function deseasonalize_and_detrend!(slice, float_times, months; aggfunc = mean, trendfunc = least_squares_fit)
    idx_groups = get_seasonal_cycle(months)
    idx_means = map(idx_groups) do idxs
        aggfunc(slice[idxs])
    end
    for (idxs, mean) in zip(idx_groups, idx_means)
        @. slice[idxs] -= mean
    end
    fit = trendfunc(float_times, slice)
    detrend!(slice, float_times, fit.slope, fit.intercept)
    return (slice, fit)
end

deseasonalize_and_detrend_twice!(slice, float_times, months; aggfunc = mean, trendfunc = least_squares_fit) = for _ in 1:2
    deseasonalize_and_detrend!(slice, float_times, months; aggfunc, trendfunc)
    nothing
end
