function generate_spatial_mean(data, latitudes, mask = ones(size(data, 1), size(data, 2)))
    replace!(data, missing => NaN)
    return vec(sum(data .* mask .* cosd.(latitudes'); dims=(1, 2)) ./ sum(mask .* cosd.(latitudes')))
end

function spatial_mean_kernel(data_slice, latitudes, mask = nothing)
    if isnothing(mask)
        latitudes_size_ratio = (size(data_slice, 1) * size(data_slice, 2)) / length(latitudes)
        return sum(data_slice .* cosd.(latitudes')) ./ (sum(cosd.(latitudes)) * latitudes_size_ratio)
    else
        return sum(data_slice .* mask .* cosd.(latitudes')) ./ sum(mask .* cosd.(latitudes'))
    end
end

function generate_spatial_mean_netcdf_compatible(data, latitudes, mask = nothing)
    out = Vector{Float64}(undef, size(data, 3))
    for i in axes(data, 3)
        out[i] = spatial_mean_kernel(data[:, :, i], latitudes, mask)
    end
    return out
end

function generate_spatial_mean_netcdf_and_p_level_compatible(data, latitudes, mask = nothing, p_level_idx = 1)
    out = Vector{Float64}(undef, size(data, 4))
    for i in axes(data, 4)
        out[i] = spatial_mean_kernel(data[:, :, p_level_idx, i], latitudes, mask)
    end
    return out
end

"Mask is a 2d lon x lat array of falses and trues / 0s and 1s, and lats is an untransposed vector of latitudes corresponding to the lat dimension of the mask"
function calculate_mask_fractional_area(mask, lats)
    total_area = sum(cosd.(lats)) * size(mask, 1)
    masked_area = sum(cosd.(lats) .* sum(mask; dims=1)')
    return masked_area / total_area
end
