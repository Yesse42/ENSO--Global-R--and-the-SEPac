"""
Stepwise regression implementation in Julia

Needs a whole set of predictors, a predictand, a function to evaluate how close the predictor and predictand are, and a threshold for that function.
"""

using StatsBase

"Note that current indices is an array of Bools"
function one_step_regress!(current_indices, predictors, predictand, eval_func, all_indices; min_gain = 0.0)

    indices_to_try = setdiff(all_indices, current_indices)
    current_eval = eval_func(current_indices, predictors, predictand)

    function eval_with_extra_point(new_index)
        # Temporarily add the new index to the current indices
        push!(current_indices, new_index)
        returnval = eval_func(current_indices, predictors, predictand)
        pop!(current_indices) # revert back
        return returnval
    end

    added_point_flag = false

    val, best_new_point = findmax(eval_with_extra_point, indices_to_try)

    if val - current_eval > min_gain
        push!(current_indices, best_new_point)
        added_point_flag = true
    end

    return current_indices, added_point_flag, val
end

function stepwise_regress(predictors, predictand, eval_func, all_indices; max_num = Inf, min_gain = 0.0)
    current_indices = Int[]
    vals = Float64[]

    added_point_flag = true
    while added_point_flag && (length(current_indices) < min(max_num, length(all_indices)))
        current_indices, added_point_flag, val = one_step_regress!(current_indices, predictors, predictand, eval_func, all_indices; min_gain = min_gain)
        push!(vals, val)
    end

    return current_indices, vals
end