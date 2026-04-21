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

    # Use analytical formulas for better numerical stability and efficiency
    x_mean = sum(x) / n
    y_mean = sum(y) / n

    # Calculate sums of squares and cross products in a single pass
    sxx = zero(eltype(x))
    sxy = zero(promote_type(eltype(x), eltype(y)))

    @inbounds for i in 1:n
        x_dev = x[i] - x_mean
        sxx += x_dev * x_dev
        sxy += x_dev * (y[i] - y_mean)
    end

    # Handle degenerate case where all x values are the same
    if sxx == 0
        return (;slope = 0.0, intercept = y_mean)
    end

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    return (;slope, intercept)
end

get_lsq_slope(x, y) = least_squares_fit(x, y).slope
