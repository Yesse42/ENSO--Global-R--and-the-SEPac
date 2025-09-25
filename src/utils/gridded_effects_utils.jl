using StatsBase, Statistics

function calculate_corrfunc_grid(grid, time_series; corrfunc = cor)
    return corrfunc.(eachslice(grid; dims = (1,2)), Ref(time_series))
end

"Calculate the total correlation of two variables, represented as a sum of other time series"
function partitioned_correlations(var1sum, var2sum)
    @assert allequal(length, var1sum)
    @assert allequal(length, var2sum)
    @assert length(var1sum[1]) == length(var2sum[1])

    var1total = reduce(+, var1sum)
    var2total = reduce(+, var2sum)
    total_corr = cor(var1total, var2total)
    var1std = std(var1total)
    var2std = std(var2total)

    out_corrs = Array{typeof(total_corr), 2}(undef, length(var1sum[1]), length(var2sum[1]))
    out_weighted_corrs = similar(out_corrs)
    for (i, var1) in enumerate(var1sum)
        for (j, var2) in enumerate(var2sum)
            out_corrs[i, j] = cor(var1, var2)
            out_weighted_corrs[i, j] = out_corrs[i, j] * (std(var1) * std(var2)) / (var1std * var2std)
        end
    end

    @assert isapprox(sum(out_weighted_corrs), total_corr; rtol = 1e-6)
    return total_corr, out_corrs, out_weighted_corrs
end