#In this script, I calculate various measures of convective aggregation
#The standard deviation of tropical SSTs and the gini coefficient of tropical SSTs

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using NCDatasets, StatsBase, Plots, Trapz

include("../pls_regressor/pls_functions.jl")

function gini_coeff_weighted_inefficient(data, w)
    data = vec(data)
    w = vec(w)
    n = length(data)
    
    # Calculate weighted mean once
    weighted_mean = sum(data .* w) / sum(w)
    
    # Calculate MAD without creating the full matrix
    mad_sum = 0.0
    w_sum = 0.0
    
    for i in 1:n
        for j in (i+1):n
            weight_ij = w[i] * w[j] * 2
            mad_sum += abs(data[i] - data[j]) * weight_ij
            w_sum += weight_ij
        end
    end
    
    MAD = mad_sum / w_sum
    return MAD / (2 * weighted_mean)
end

function gini_coeff_weighted(data, w)
    data = vec(data)
    w = vec(w)
    n = length(data)
    
    # Sort data and weights by data values
    sorted_indices = sortperm(data)
    sorted_data = data[sorted_indices]
    sorted_w = w[sorted_indices]

    # Cumulative weights and normalized cumulative weights
    cum_w = cumsum(sorted_w)
    total_w = sum(sorted_w)
    norm_cum_w = cum_w / total_w
    total_data = cumsum(sorted_data .* sorted_w)
    norm_total_data = total_data ./ total_data[end]  # Normalize to 1

    #Add (0,0) to the beginning of the curves
    pushfirst!(norm_cum_w, 0.0)
    pushfirst!(norm_total_data, 0.0)

    one_to_one_line = norm_cum_w

    #Now calculate the area between the curves (via the trapezoidal rule) divided by the total area of the 1-1 curve (which is just a triangle of area 0.5)

    area_between_curves = trapz(norm_cum_w, abs.(norm_total_data .- one_to_one_line))
    gini_coefficient = area_between_curves / 0.5  # Normalize by the area of the triangle under the 1-1 line
    return gini_coefficient
end

function std_weighted(data, w)
    weighted_mean = sum(data .* w) / sum(w)
    variance = sum(w .* (data .- weighted_mean).^2) / sum(w)
    return sqrt(variance)
end

#Now load in the era5 sst data
period = (Date(1980), Date(2024, 12, 31))

era5_data, era5_coords = load_era5_data(["sst"], period)

sst_data = era5_data["sst"]  #Assuming dimensions are (time, lat, lon)
lats = era5_coords["latitude"]
full_lats = repeat(lats', size(sst_data, 1))

lsm = map(slice -> all(val ->!ismissing(val) && !isnan(val), slice), eachslice(sst_data, dims = (1,2)))
lat_mask = abs.(full_lats) .<= 5

full_mask = vec(lsm .& lat_mask)

sst_data_linear = reshape(sst_data, size(sst_data, 1) * size(sst_data, 2), size(sst_data, 3))
full_lats_linear = reshape(full_lats, :)

sst_data_valid = sst_data_linear[full_mask, :]
lats_valid = full_lats_linear[full_mask]

function calculate_aggregation_measures(sst_data_valid, lats_valid)
    gini_coeff_timeseries = vec(map(slice -> gini_coeff_weighted(slice, cosd.(lats_valid)), eachcol(sst_data_valid)))
    sst_std_timeseries = vec(map(slice -> std_weighted(slice, cosd.(lats_valid)), eachcol(sst_data_valid)))
    return gini_coeff_timeseries, sst_std_timeseries
end

gini_coeff_timeseries, sst_std_timeseries = calculate_aggregation_measures(sst_data_valid, lats_valid)

#compare a standardized version of these two time series
standardize_ts(ts) = (ts .- mean(ts)) ./ std(ts)
gini_coeff_std = standardize_ts(gini_coeff_timeseries)
sst_std_std = standardize_ts(sst_std_timeseries)
using Plots

# Create time axis
time_axis = collect(period[1]:Month(1):period[2])[1:length(gini_coeff_std)]

# Plot both standardized time series
plot(time_axis, gini_coeff_std, label="Standardized Gini Coefficient", linewidth=2)
plot!(time_axis, sst_std_std, label="Standardized SST Std Dev", linewidth=2)
xlabel!("Time")
ylabel!("Standardized Values")
title!("Convective Aggregation Measures")

datasavedir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/Convective_Aggregation"
using CSV, DataFrames

# Create DataFrame with the measures
df = DataFrame(
    date = era5_coords["time"][:],
    gini_coefficient = gini_coeff_timeseries,
    sst_std_dev = sst_std_timeseries,
    gini_coefficient_standardized = gini_coeff_std,
    sst_std_dev_standardized = sst_std_std
)

# Save to CSV
CSV.write(joinpath(datasavedir, "convective_aggregation_measures.csv"), df)

# --- Tropical Pacific only (lon 120-280, lat ±5) ---
lons = era5_coords["longitude"]
full_lons = repeat(ones(length(lats))', length(lons)) .* lons
full_lons_linear = reshape(full_lons, :)

lon_mask = (full_lons_linear .>= 120) .& (full_lons_linear .<= 280)
tpac_mask = vec(lsm .& lat_mask) .& lon_mask

sst_data_tpac = sst_data_linear[tpac_mask, :]
lats_tpac = full_lats_linear[tpac_mask]

gini_coeff_tpac, sst_std_tpac = calculate_aggregation_measures(sst_data_tpac, lats_tpac)

gini_coeff_tpac_std = standardize_ts(gini_coeff_tpac)
sst_std_tpac_std = standardize_ts(sst_std_tpac)

# Plot tropical Pacific timeseries
plot(time_axis, gini_coeff_tpac_std, label="Standardized Gini Coefficient (Trop Pac)", linewidth=2)
plot!(time_axis, sst_std_tpac_std, label="Standardized SST Std Dev (Trop Pac)", linewidth=2)
xlabel!("Time")
ylabel!("Standardized Values")
title!("Convective Aggregation Measures - Tropical Pacific")

# Save tropical Pacific measures
df_tpac = DataFrame(
    date = era5_coords["time"][:],
    gini_coefficient = gini_coeff_tpac,
    sst_std_dev = sst_std_tpac,
    gini_coefficient_standardized = gini_coeff_tpac_std,
    sst_std_dev_standardized = sst_std_tpac_std
)

CSV.write(joinpath(datasavedir, "convective_aggregation_measures_tropical_pacific.csv"), df_tpac)

# --- Correlation between tropical mean T_500 and convective aggregation measures ---
# Load ERA5 pressure-level temperature data
era5_plev_data, era5_plev_coords = load_era5_data(["t"], period)

t_data = era5_plev_data["t"]  # Dimensions: (lon, lat, pressure, time)
plev_lats = era5_plev_coords["latitude"]
pressure_levels = era5_plev_coords["pressure_level"]

# Find the 500 hPa pressure level index
p500_idx = findfirst(==(500), pressure_levels)
if p500_idx === nothing
    error("500 hPa pressure level not found! Available levels: $pressure_levels")
end

# Extract T at 500 hPa: (lon, lat, time)
t500_data = t_data[:, :, p500_idx, :]

# Tropical mean T_500 (|lat| <= 30) weighted by cos(lat)
trop_lat_mask = abs.(plev_lats) .<= 30
cos_weights = cosd.(plev_lats[trop_lat_mask])
t500_trop = t500_data[:, trop_lat_mask, :]

# Compute area-weighted tropical mean T_500 timeseries
n_times_plev = size(t500_trop, 3)
t500_tropical_mean = vec(sum(t500_trop .* reshape(cos_weights, 1, :, 1); dims=(1,2)) ./ sum(cos_weights * size(t500_trop, 1)))

# Match pressure-level times to the aggregation measure times
plev_times = era5_plev_coords["pressure_time"]
agg_times = era5_coords["time"][:]

plev_date_idx = indexin(agg_times, plev_times)
if any(isnothing, plev_date_idx)
    # Use only overlapping dates
    common_dates = intersect(agg_times, plev_times)
    agg_idx = indexin(common_dates, agg_times)
    plev_idx = indexin(common_dates, plev_times)
else
    agg_idx = 1:length(agg_times)
    plev_idx = plev_date_idx
end

t500_matched = t500_tropical_mean[plev_idx]
gini_matched = gini_coeff_timeseries[agg_idx]
sst_std_matched = sst_std_timeseries[agg_idx]
gini_tpac_matched = gini_coeff_tpac[agg_idx]
sst_std_tpac_matched_local = sst_std_tpac[agg_idx]
matched_times = agg_times[agg_idx]

# Deseasonalize and detrend all series
float_time_matched = calc_float_time.(matched_times)
month_groups_matched = groupfind(month, matched_times)

t500_detrended = copy(t500_matched)
deseasonalize_and_detrend_precalculated_groups_twice!(t500_detrended, float_time_matched, month_groups_matched; aggfunc = mean, trendfunc = least_squares_fit)

gini_detrended = copy(gini_matched)
deseasonalize_and_detrend_precalculated_groups_twice!(gini_detrended, float_time_matched, month_groups_matched; aggfunc = mean, trendfunc = least_squares_fit)

sst_std_detrended = copy(sst_std_matched)
deseasonalize_and_detrend_precalculated_groups_twice!(sst_std_detrended, float_time_matched, month_groups_matched; aggfunc = mean, trendfunc = least_squares_fit)

gini_tpac_detrended = copy(gini_tpac_matched)
deseasonalize_and_detrend_precalculated_groups_twice!(gini_tpac_detrended, float_time_matched, month_groups_matched; aggfunc = mean, trendfunc = least_squares_fit)

sst_std_tpac_detrended = copy(sst_std_tpac_matched_local)
deseasonalize_and_detrend_precalculated_groups_twice!(sst_std_tpac_detrended, float_time_matched, month_groups_matched; aggfunc = mean, trendfunc = least_squares_fit)

# Calculate correlations
cor_t500_gini = cor(t500_detrended, gini_detrended)
cor_t500_sst_std = cor(t500_detrended, sst_std_detrended)
cor_t500_gini_tpac = cor(t500_detrended, gini_tpac_detrended)
cor_t500_sst_std_tpac = cor(t500_detrended, sst_std_tpac_detrended)

println("="^60)
println("CORRELATIONS: Tropical Mean T_500 vs Aggregation Measures")
println("(detrended & deseasonalized)")
println("="^60)
println()
println("Global tropics (|lat| ≤ 5):")
println("  Cor(T_500, Gini coeff):      $(round(cor_t500_gini, digits=4))")
println("  Cor(T_500, SST std dev):      $(round(cor_t500_sst_std, digits=4))")
println()
println("Tropical Pacific (lon 120-280, |lat| ≤ 5):")
println("  Cor(T_500, Gini coeff):      $(round(cor_t500_gini_tpac, digits=4))")
println("  Cor(T_500, SST std dev):      $(round(cor_t500_sst_std_tpac, digits=4))")
println()
println("="^60)

df_t500_ts = DataFrame(
    date = matched_times,
    T500_tropical_mean = t500_matched,
    T500_detrended = t500_detrended
)
CSV.write(joinpath(datasavedir, "T500_tropical_mean_timeseries.csv"), df_t500_ts)

# --- Remove ENSO and recalculate T500 correlations ---
enso_data, enso_dates_raw = load_enso_data(period)
enso_times = DateTime.(round_dates_down_to_nearest_month(enso_dates_raw["time"]))

lags = -24:24
lag_columns = ["oni_lag_$lag" for lag in lags]

n_enso = length(enso_times)
enso_lag_matrix_full = Matrix{Union{Float64, Missing}}(missing, n_enso, length(lag_columns))
for (j, col) in enumerate(lag_columns)
    if haskey(enso_data, col)
        enso_lag_matrix_full[:, j] .= enso_data[col]
    end
end
valid_enso_rows = [all(!ismissing, enso_lag_matrix_full[i, :]) for i in 1:n_enso]
enso_times_valid = enso_times[valid_enso_rows]
enso_lag_matrix = Float64.(enso_lag_matrix_full[valid_enso_rows, :])

enso_matrix_matched, data_valid_idx = precompute_enso_match(matched_times, enso_times_valid, enso_lag_matrix)

# Remove ENSO from all detrended series
t500_no_enso = remove_enso_via_pls(enso_matrix_matched, t500_detrended[data_valid_idx]; verbose=true, label="t500_detrended")
gini_no_enso = remove_enso_via_pls(enso_matrix_matched, gini_detrended[data_valid_idx]; verbose=true, label="gini_detrended")
sst_std_no_enso = remove_enso_via_pls(enso_matrix_matched, sst_std_detrended[data_valid_idx]; verbose=true, label="sst_std_detrended")
gini_tpac_no_enso = remove_enso_via_pls(enso_matrix_matched, gini_tpac_detrended[data_valid_idx]; verbose=true, label="gini_tpac_detrended")
sst_std_tpac_no_enso = remove_enso_via_pls(enso_matrix_matched, sst_std_tpac_detrended[data_valid_idx]; verbose=true, label="sst_std_tpac_detrended")

# Correlations after ENSO removal
cor_t500_gini_no_enso = cor(t500_no_enso, gini_no_enso)
cor_t500_sst_std_no_enso = cor(t500_no_enso, sst_std_no_enso)
cor_t500_gini_tpac_no_enso = cor(t500_no_enso, gini_tpac_no_enso)
cor_t500_sst_std_tpac_no_enso = cor(t500_no_enso, sst_std_tpac_no_enso)

println("="^60)
println("CORRELATIONS AFTER ENSO REMOVAL: Tropical Mean T_500 vs Aggregation Measures")
println("(detrended, deseasonalized, ENSO removed via PLS)")
println("="^60)
println()
println("Global tropics (|lat| ≤ 5):")
println("  Cor(T_500, Gini coeff):      $(round(cor_t500_gini_no_enso, digits=4))")
println("  Cor(T_500, SST std dev):      $(round(cor_t500_sst_std_no_enso, digits=4))")
println()
println("Tropical Pacific (lon 120-280, |lat| ≤ 5):")
println("  Cor(T_500, Gini coeff):      $(round(cor_t500_gini_tpac_no_enso, digits=4))")
println("  Cor(T_500, SST std dev):      $(round(cor_t500_sst_std_tpac_no_enso, digits=4))")
println()
println("="^60)

# Save both before and after ENSO removal correlations
df_t500_corr_both = DataFrame(
    measure = ["Gini (global)", "SST σ (global)", "Gini (trop Pac)", "SST σ (trop Pac)"],
    correlation_with_T500 = [cor_t500_gini, cor_t500_sst_std, cor_t500_gini_tpac, cor_t500_sst_std_tpac],
    correlation_with_T500_no_ENSO = [cor_t500_gini_no_enso, cor_t500_sst_std_no_enso, cor_t500_gini_tpac_no_enso, cor_t500_sst_std_tpac_no_enso]
)
CSV.write(joinpath(datasavedir, "T500_aggregation_correlations.csv"), df_t500_corr_both)
