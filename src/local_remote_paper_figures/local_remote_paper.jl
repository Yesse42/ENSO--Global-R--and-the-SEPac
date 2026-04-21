# This script calculate various plots showing the local vs remote impacts of the southeast pacific, as well as other regions.

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")

using JLD2, Dates, StatsBase, DataFrames, Printf, CSV

visdir_base = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/local_nonlocal_paper_figs"

"""
    plot_table_heatmap(table_data, row_labels, col_labels; title="", colorbar_label="", digits=2, separator_rows=[0.5, 2.5], figsize=(6, 4), savepath=nothing, clim=nothing)

Plot a heatmap table with annotated cell values, diverging colormap centered at zero, and optional separator lines.

# Arguments
- `table_data`: Matrix of values to display
- `row_labels`: Vector of row label strings
- `col_labels`: Vector of column label strings
- `title`: Title for the table plot
- `colorbar_label`: Label for the colorbar
- `digits`: Number of decimal places to show in cell annotations (default: 2)
- `separator_rows`: y-positions for thick horizontal separator lines (default: [0.5, 2.5])
- `figsize`: Figure size tuple (default: (6, 4))
- `savepath`: If provided, save the figure to this path
- `clim`: If provided, use this value as the absolute max for the color scale (default: nothing, auto-computed from data)
"""
function plot_table_heatmap(table_data, row_labels, col_labels;
        title="", colorbar_label="", digits=2, separator_rows=[0.5, 2.5],
        figsize=(6, 4), savepath=nothing, clim=nothing)
    absmax = isnothing(clim) ? maximum(abs.(table_data)) : clim
    norm = colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    cmap = cmr.prinsenvlag.reversed()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(table_data, cmap=cmap, norm=norm, aspect="auto")

    for i in 0:(size(table_data, 1)-1), j in 0:(size(table_data, 2)-1)
        val = table_data[i+1, j+1]
        text_color = abs(val) > 0.6 * absmax ? "white" : "black"
        ax.text(j, i, string(round(val; digits=digits)), ha="center", va="center", fontsize=10, color=text_color, fontweight="bold")
    end

    for y in separator_rows
        ax.axhline(y, color="black", linewidth=2.5)
    end

    ax.set_xticks(pylist(collect(0:length(col_labels)-1)))
    ax.set_xticklabels(pylist(col_labels), fontsize=11)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(pylist(collect(0:length(row_labels)-1)))
    ax.set_yticklabels(pylist(row_labels), fontsize=11)
    ax.tick_params(length=0)

    ax.set_title(title, pad=25, fontsize=13)
    plt.colorbar(im, ax=ax, orientation="vertical", label=colorbar_label, shrink=0.8, pad=0.04)

    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    end

    return fig, ax
end

date_range = (Date(2000, 3), Date(2024, 2, 28))

# First load in all the necessary data

# Load in the ERA5 t2m dataset

era5_var = ["t2m"]
era5_data, era5_coords = load_era5_data(era5_var, date_range)

# Load in the CEREs radiation dataset

ceres_net_rad_var = ["toa_net_all_mon"]
ceres_sw_lw_vars = ["toa_net_sw_mon", "toa_net_lw_mon"]
ceres_clr_sky_vars = ["toa_sw_clr_t_mon", "toa_lw_clr_t_mon"]
ceres_vars_needed_to_construct_cloudy_sky = ["toa_sw_all_mon", "toa_lw_all_mon", "solar_mon"]
ceres_vars_to_load = vcat(ceres_net_rad_var, ceres_sw_lw_vars, ceres_clr_sky_vars, ceres_vars_needed_to_construct_cloudy_sky)

ceres_data, ceres_coords = load_new_ceres_data(ceres_vars_to_load, date_range)
# Reformat the CEREs data to match the ERA5 time coordinates

ceres_coords["time"] = round_dates_down_to_nearest_month(ceres_coords["time"])

common_time = ceres_coords["time"]
if !all(common_time .== round_dates_down_to_nearest_month(era5_coords["time"]))
    error("ERA5 and CEREs time coordinates do not match even after rounding!")
end

# Construct the cloudy sky radiation components and add them to the ceres_data dictionary

# Longwave needs the sign flip because these are upwards fluxes, and sw needs the sign flip and sunlight so that everything adds up to the sw flux


set!(ceres_data, "toa_sw_clr_t_mon", ceres_data["solar_mon"] .- ceres_data["toa_sw_clr_t_mon"])
set!(ceres_data, "toa_sw_cld_t_mon", ceres_data["toa_net_sw_mon"] .- ceres_data["toa_sw_clr_t_mon"])
set!(ceres_data, "toa_lw_cld_t_mon", -1 .* (ceres_data["toa_lw_all_mon"] .- ceres_data["toa_lw_clr_t_mon"]))
set!(ceres_data, "toa_lw_clr_t_mon", ceres_data["toa_lw_clr_t_mon"] .* -1)


# Deseasonalize and detrend both datasets twice

for datadict in [era5_data, ceres_data]
    time = common_time
    float_time = calc_float_time.(time)
    month_groups = groupfind(month, time)
    for var_data in datadict
        for slice in eachslice(var_data; dims = (1,2))
            deseasonalize_and_detrend_precalculated_groups_twice!(slice, float_time, month_groups; aggfunc = mean, trendfunc = least_squares_fit)
        end
    end
end

# Generate/load the sepac and nepac region masks
regions_to_inspect = ["SEPac_feedback_definition", "NEPac", "SEAtl"]
jldpath = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/"
era5_region_mask_dict = jldopen(joinpath(jldpath, "stratocumulus_region_masks.jld2"))["regional_masks_era5"]
rad_region_mask_dict = jldopen(joinpath(jldpath, "stratocumulus_exclusion_masks.jld2"))["stratocum_masks"]

region_data_dict = Dictionary(regions_to_inspect, [Dictionary(["temp_mask", "rad_mask", "fractional_area_of_region", "fractional_area_of_temp_region"],
    [era5_region_mask_dict[region_name], rad_region_mask_dict[region_name],
    calculate_mask_fractional_area(rad_region_mask_dict[region_name], ceres_coords["latitude"]),
    calculate_mask_fractional_area(era5_region_mask_dict[region_name], era5_coords["latitude"])] ) for region_name in regions_to_inspect])

# Use these masks to calculate the regional temperature means, the global mean temp, global average rad, and regional and nonregional average rad. Dicts are region_name => var_name => local, global, or nonlocal (if applicable) => data vector
ceres_spatial_means_dict = Dict{String, Dict{String, Dict{String, Vector{Float64}}}}()
era5_spatial_means_dict = Dict{String, Dict{String, Dict{String, Vector{Float64}}}}()

# Global means (no regional mask)
ceres_spatial_means_dict["global"] = Dict{String, Dict{String, Vector{Float64}}}()
for var_name in keys(ceres_data)
    ceres_spatial_means_dict["global"][var_name] = Dict("global" => generate_spatial_mean(ceres_data[var_name], ceres_coords["latitude"]))
end

era5_spatial_means_dict["global"] = Dict{String, Dict{String, Vector{Float64}}}()
for var_name in keys(era5_data)
    era5_spatial_means_dict["global"][var_name] = Dict("global" => generate_spatial_mean(era5_data[var_name], era5_coords["latitude"]))
end

# Regional means: local (within mask), nonlocal (outside mask)

for region_name in regions_to_inspect
    rad_mask = region_data_dict[region_name]["rad_mask"]
    temp_mask = region_data_dict[region_name]["temp_mask"]

    ceres_spatial_means_dict[region_name] = Dict{String, Dict{String, Vector{Float64}}}()
    for var_name in keys(ceres_data)
        ceres_spatial_means_dict[region_name][var_name] = Dict(
            "local" => generate_spatial_mean(ceres_data[var_name], ceres_coords["latitude"], rad_mask),
            "nonlocal" => generate_spatial_mean(ceres_data[var_name], ceres_coords["latitude"], .!rad_mask)
        )
    end

    era5_spatial_means_dict[region_name] = Dict{String, Dict{String, Vector{Float64}}}()
    for var_name in keys(era5_data)
        era5_spatial_means_dict[region_name][var_name] = Dict(
            "local" => generate_spatial_mean(era5_data[var_name], era5_coords["latitude"], temp_mask),
            "nonlocal" => generate_spatial_mean(era5_data[var_name], era5_coords["latitude"], .!temp_mask)
        )
    end
end

# Load tropical SST variability (std dev) data for correlation tables
sst_std_csv = CSV.read("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/Convective_Aggregation/convective_aggregation_measures.csv", DataFrame)
sst_std_dates = Date.(sst_std_csv.date)
sst_std_vals = sst_std_csv.sst_std_dev

# Match SST std dev dates to common_time
sst_std_date_idx = indexin(common_time, sst_std_dates)
if any(isnothing, sst_std_date_idx)
    error("Some common_time dates are not found in the SST std dev data!")
end
sst_std_matched = sst_std_vals[sst_std_date_idx]

# Deseasonalize and detrend the SST std dev series to match other data processing
sst_std_matched_copy = copy(sst_std_matched)
float_time_sst = calc_float_time.(common_time)
month_groups_sst = groupfind(month, common_time)
deseasonalize_and_detrend_precalculated_groups_twice!(sst_std_matched_copy, float_time_sst, month_groups_sst; aggfunc = mean, trendfunc = least_squares_fit)
sst_std_detrended = sst_std_matched_copy

# Load tropical Pacific SST variability data for correlation table
sst_std_tpac_csv = CSV.read("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/Convective_Aggregation/convective_aggregation_measures_tropical_pacific.csv", DataFrame)
sst_std_tpac_dates = Date.(sst_std_tpac_csv.date)
sst_std_tpac_vals = sst_std_tpac_csv.sst_std_dev

# Match tropical Pacific SST std dev dates to common_time
sst_std_tpac_date_idx = indexin(common_time, sst_std_tpac_dates)
if any(isnothing, sst_std_tpac_date_idx)
    error("Some common_time dates are not found in the tropical Pacific SST std dev data!")
end
sst_std_tpac_matched = sst_std_tpac_vals[sst_std_tpac_date_idx]

# Deseasonalize and detrend the tropical Pacific SST std dev series
sst_std_tpac_matched_copy = copy(sst_std_tpac_matched)
deseasonalize_and_detrend_precalculated_groups_twice!(sst_std_tpac_matched_copy, float_time_sst, month_groups_sst; aggfunc = mean, trendfunc = least_squares_fit)
sst_std_tpac_detrended = sst_std_tpac_matched_copy

# Load ENSO data and precompute the lag matrix and matching indices
enso_data, enso_dates_raw = load_enso_data(date_range)
enso_times = round_dates_down_to_nearest_month(enso_dates_raw["time"])

# Build the ENSO lag matrix (n_times x n_lags), dropping rows with any missing values
lags = -24:24
lag_columns = ["oni_lag_$lag" for lag in lags]

# Handle missings: build full matrix then remove rows with any NaN/missing
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

# Precompute matching indices between data times and valid ENSO times
enso_matrix_matched, data_valid_idx = precompute_enso_match(common_time, enso_times_valid, enso_lag_matrix)

# Dictionary to store all plot data for deferred plotting with shared color limits
plot_data = Dict{String, Dict{String, Any}}()

# Save original data for restoration before ENSO removal passes
original_ceres_spatial_means = deepcopy(ceres_spatial_means_dict)
original_era5_spatial_means = deepcopy(era5_spatial_means_dict)
original_sst_std_detrended = copy(sst_std_detrended)
original_sst_std_tpac_detrended = copy(sst_std_tpac_detrended)
original_ceres_data = deepcopy(ceres_data)
original_era5_data = deepcopy(era5_data)
original_common_time = copy(common_time)

# --- Helper functions for lagged SST regression ---

_isvalid(x) = !ismissing(x) && !isnan(x)

"""Convert array with potential missings to Float64, replacing missing → NaN."""
function missings_to_nan(data::AbstractArray)
    out = Array{Float64}(undef, size(data))
    @inbounds for i in eachindex(data)
        out[i] = ismissing(data[i]) ? NaN : Float64(data[i])
    end
    return out
end
missings_to_nan(data::AbstractArray{<:AbstractFloat}) = Float64.(data)

"""Set entire timeseries to NaN at any (lon, lat) gridpoint where any time step is NaN."""
function mask_incomplete_gridpoints!(data::AbstractArray{<:AbstractFloat, 3})
    @inbounds for j in axes(data, 2), i in axes(data, 1)
        ts = @view data[i, j, :]
        if any(isnan, ts)
            ts .= NaN
        end
    end
end

"""
    nanmissing_regcoef(x, y)

Regression coefficient β = cov(x,y)/var(x), skipping indices where either
value is missing or NaN. Returns NaN if fewer than 3 valid pairs.
"""
function nanmissing_regcoef(x::AbstractVector, y::AbstractVector)
    n = 0; sx = 0.0; sy = 0.0
    @inbounds for i in eachindex(x, y)
        (_isvalid(x[i]) && _isvalid(y[i])) || continue
        n += 1; sx += Float64(x[i]); sy += Float64(y[i])
    end
    n < 3 && return NaN
    xm, ym = sx / n, sy / n
    sxx = 0.0; sxy = 0.0
    @inbounds for i in eachindex(x, y)
        (_isvalid(x[i]) && _isvalid(y[i])) || continue
        dx = Float64(x[i]) - xm
        sxx += dx * dx; sxy += dx * (Float64(y[i]) - ym)
    end
    sxx == 0.0 && return NaN
    return sxy / sxx
end

"""Regress each gridpoint of a 3D array (lon × lat × time) onto an index vector.
Returns a 2D map of regression coefficients."""
function regress_grid_on_index(gridded::AbstractArray{<:AbstractFloat, 3}, index::AbstractVector)
    coefs = Matrix{Float64}(undef, size(gridded, 1), size(gridded, 2))
    @inbounds for j in axes(gridded, 2), i in axes(gridded, 1)
        coefs[i, j] = nanmissing_regcoef(index, @view gridded[i, j, :])
    end
    return coefs
end

"""Compute lagged regression maps of a gridded field onto an index at each lag.
Returns a Dict(lag => 2D coefficient map)."""
function compute_lagged_regression_maps(gridded::AbstractArray{<:AbstractFloat, 3}, index::AbstractVector, lags)
    return Dict(lag => regress_grid_on_index(gridded, time_lag(index, lag)) for lag in lags)
end

# --- Load and prepare ERA5 SST data ---

println("Loading ERA5 SST data...")
sst_dict, sst_coords = load_era5_data(["sst"], date_range)
sst = missings_to_nan(sst_dict["sst"])
mask_incomplete_gridpoints!(sst)

# Verify time alignment
sst_time = round_dates_down_to_nearest_month(sst_coords["time"])
@assert all(sst_time .== common_time) "SST time does not match common_time!"

# Deseasonalize and detrend (skip all-NaN land gridpoints)
sst_float_time = calc_float_time.(sst_time)
sst_month_groups = groupfind(month, sst_time)
for slice in eachslice(sst; dims=(1, 2))
    any(isnan, slice) && continue
    deseasonalize_and_detrend_precalculated_groups_twice!(slice, sst_float_time, sst_month_groups;
        aggfunc=mean, trendfunc=least_squares_fit)
end

sst_lat = Float64.(sst_coords["latitude"])
sst_lon = Float64.(sst_coords["longitude"])

regression_lags = [-12, -6, -3, -1, 0, 1, 3, 6, 12]
original_sst = deepcopy(sst)

# Run analysis: E (with ENSO), NE (ENSO removed via PLS), NEA (ENSO removed additively via global R)
for enso_mode in [:E, :NE, :NEA]

# Restore original data for ENSO removal passes
if enso_mode != :E
    for (rk, vd) in original_ceres_spatial_means
        for (vn, ld) in vd
            for (lk, ts) in ld
                ceres_spatial_means_dict[rk][vn][lk] = copy(ts)
            end
        end
    end
    for (rk, vd) in original_era5_spatial_means
        for (vn, ld) in vd
            for (lk, ts) in ld
                era5_spatial_means_dict[rk][vn][lk] = copy(ts)
            end
        end
    end
    for (key, var_data) in pairs(original_ceres_data)
        set!(ceres_data, key, copy(var_data))
    end
    for (key, var_data) in pairs(original_era5_data)
        set!(era5_data, key, copy(var_data))
    end
    global sst_std_detrended = copy(original_sst_std_detrended)
    global sst_std_tpac_detrended = copy(original_sst_std_tpac_detrended)
    global common_time = copy(original_common_time)
end

if enso_mode == :E
    visdir = visdir_base

elseif enso_mode == :NE
    visdir = visdir_base * "_enso_removed"

    # Remove ENSO from all spatial mean time series (trim to matched dates)

    for spatial_dict in [ceres_spatial_means_dict, era5_spatial_means_dict]
        for (region_key, var_dict) in spatial_dict
            for (var_name, locality_dict) in var_dict
                for (locality_key, ts) in locality_dict
                    y = ts[data_valid_idx]
                    locality_dict[locality_key] = remove_enso_via_pls(enso_matrix_matched, y; verbose=true, label="$(region_key)_$(var_name)_$(locality_key)")
                end
            end
        end
    end

    # Remove ENSO from SST std dev (convective aggregation) time series
    global sst_std_detrended = remove_enso_via_pls(enso_matrix_matched, sst_std_detrended[data_valid_idx]; verbose=true, label="sst_std_detrended")
    global sst_std_tpac_detrended = remove_enso_via_pls(enso_matrix_matched, sst_std_tpac_detrended[data_valid_idx]; verbose=true, label="sst_std_tpac_detrended")

    # Remove ENSO from gridded radiation and temperature data (slice-wise per grid point), pad with NaN, then trim

    for datadict in [ceres_data, era5_data]
        for var_data in datadict
            for slice in eachslice(var_data; dims = (1,2))
                y = vec(slice)[data_valid_idx]
                residual = remove_enso_via_pls(enso_matrix_matched, y)
                slice .= NaN
                slice[data_valid_idx] .= residual
            end
        end
    end

    # Trim gridded data to only the valid time indices

    for datadict in [ceres_data, era5_data]
        for (key, var_data) in pairs(datadict)
            set!(datadict, key, var_data[:, :, data_valid_idx])
        end
    end
    global common_time = common_time[data_valid_idx]

elseif enso_mode == :NEA
    visdir = visdir_base * "_enso_removed_additivity_preserved"

    # Step 1: Fit PLS of ENSO lag matrix on global mean net radiation to get the ENSO component
    global_R_for_pls = ceres_spatial_means_dict["global"]["toa_net_all_mon"]["global"][data_valid_idx]
    pls_model_R = make_pls_regressor(enso_matrix_matched, global_R_for_pls, 1; print_updates=false)
    enso_R_hat = vec(predict(pls_model_R, enso_matrix_matched))
    enso_R_hat_var = dot(enso_R_hat, enso_R_hat)
    println("NEA: PLS correlation for global mean net radiation: $(cor(enso_R_hat, global_R_for_pls))")

    # Step 2: For global mean net R, use PLS residual directly
    # For all other variables, linearly remove enso_R_hat to preserve additivity

    for spatial_dict in [ceres_spatial_means_dict, era5_spatial_means_dict]
        for (region_key, var_dict) in spatial_dict
            for (var_name, locality_dict) in var_dict
                for (locality_key, ts) in locality_dict
                    y = ts[data_valid_idx]
                    if region_key == "global" && var_name == "toa_net_all_mon" && locality_key == "global"
                        locality_dict[locality_key] = y .- enso_R_hat
                    else
                        β = dot(y, enso_R_hat) / enso_R_hat_var
                        locality_dict[locality_key] = y .- β .* enso_R_hat
                    end
                end
            end
        end
    end

    # Remove ENSO from SST std dev time series via linear removal of enso_R_hat
    y_sst = sst_std_detrended[data_valid_idx]
    global sst_std_detrended = y_sst .- (dot(y_sst, enso_R_hat) / enso_R_hat_var) .* enso_R_hat
    y_sst_tpac = sst_std_tpac_detrended[data_valid_idx]
    global sst_std_tpac_detrended = y_sst_tpac .- (dot(y_sst_tpac, enso_R_hat) / enso_R_hat_var) .* enso_R_hat

    # Remove ENSO from gridded data via linear removal of enso_R_hat

    for datadict in [ceres_data, era5_data]
        for var_data in datadict
            for slice in eachslice(var_data; dims = (1,2))
                y = vec(slice)[data_valid_idx]
                β = dot(y, enso_R_hat) / enso_R_hat_var
                residual = y .- β .* enso_R_hat
                slice .= NaN
                slice[data_valid_idx] .= residual
            end
        end
    end

    # Trim gridded data to only the valid time indices

    for datadict in [ceres_data, era5_data]
        for (key, var_data) in pairs(datadict)
            set!(datadict, key, var_data[:, :, data_valid_idx])
        end
    end
    global common_time = common_time[data_valid_idx]
end

enso_label = enso_mode == :E ? "enso" : enso_mode == :NE ? "enso_removed" : "enso_removed_additivity_preserved"
enso_abbrev = enso_mode == :E ? "E" : enso_mode == :NE ? "NE" : "NEA"
pd = Dict{String, Any}()
pd["visdir"] = visdir
pd["regions"] = Dict{String, Dict{String, Any}}()

# First calculate L, which is the covariance of global R and local T_i normalized by the global temperature variance
global_mean_R = ceres_spatial_means_dict["global"]["toa_net_all_mon"]["global"]
local_T = era5_data["t2m"]
global_mean_T = era5_spatial_means_dict["global"]["t2m"]["global"]
global_temp_variance = var(global_mean_T; mean = mean(global_mean_T))

L = cov.(Ref(global_mean_R), vec.(eachslice(local_T; dims = (1,2)))) ./ global_temp_variance

L_global_mean = generate_spatial_mean(L, era5_coords["latitude"])

println("The global mean of L is (with $enso_label) $(round(mean(L_global_mean), digits=4)) W/m²/K")

# Save L for deferred plotting
pd["L"] = copy(L)

# Then decompose L_SEPac (do other regions via for loop) into local and remote contributions. Make this a table

for region_name in regions_to_inspect

    region_visdir = joinpath(visdir, region_name)
    mkpath(region_visdir)

    rd = Dict{String, Any}()
    rd["region_visdir"] = region_visdir

    R_sep = ceres_spatial_means_dict[region_name]["toa_net_all_mon"]["local"]
    R_nonsep = ceres_spatial_means_dict[region_name]["toa_net_all_mon"]["nonlocal"]

    sep_area_fraction = region_data_dict[region_name]["fractional_area_of_region"]
    nonsep_area_fraction = 1.0 - sep_area_fraction

    T_SEP = era5_spatial_means_dict[region_name]["t2m"]["local"]
    L_SEP = cov(global_mean_R, T_SEP) / global_temp_variance
    L_SEP_local = sep_area_fraction * cov(R_sep, T_SEP) / global_temp_variance
    L_SEP_remote = nonsep_area_fraction * cov(R_nonsep, T_SEP) / global_temp_variance
    if abs(L_SEP - (L_SEP_local + L_SEP_remote)) > 1e-5
        @warn "L_SEP does not equal the sum of its local and remote contributions!"
    end

    # Pretty print the results

    println("="^60)
    println("FEEDBACK PARAMETER DECOMPOSITION: $region_name")
    println("="^60)
    println()

    println("Feedback Parameter (L_$region_name):")
    println("  Total:   $(round(L_SEP, digits=4)) W/m²/K")
    println("  Local:   $(round(L_SEP_local, digits=4)) W/m²/K ($(round(100*L_SEP_local/L_SEP, digits=1))%)")
    println("  Remote:  $(round(L_SEP_remote, digits=4)) W/m²/K ($(round(100*L_SEP_remote/L_SEP, digits=1))%)")
    println()

    println("Area fractions:")
    println("  Region:      $(round(100*sep_area_fraction, digits=1))%")
    println("  Non-region:  $(round(100*nonsep_area_fraction, digits=1))%")
    println()

    println("Summary statistics:")
    println("  Global mean R variance:  $(round(var(global_mean_R), digits=4)) (W/m²)²")
    println("  Global mean T variance:  $(round(global_temp_variance, digits=4)) K²")
    println("  $region_name T - Global R corr: $(round(cor(global_mean_R, T_SEP), digits=4))")
    println()
    println("="^60)

    # Decompose these local and nonlocal contributions into sw and lw components, and then after that analysis decompose those into cloudy and clear sky components.
    sw_lw_var_names = Dict("sw" => "toa_net_sw_mon", "lw" => "toa_net_lw_mon")
    sw_lw_results = Dict{String, NamedTuple}()
    for (rad_type, var_name) in sw_lw_var_names
        R_local = ceres_spatial_means_dict[region_name][var_name]["local"]
        R_nonlocal = ceres_spatial_means_dict[region_name][var_name]["nonlocal"]
        L_comp = sep_area_fraction * cov(R_local, T_SEP) / global_temp_variance
        L_comp_remote = nonsep_area_fraction * cov(R_nonlocal, T_SEP) / global_temp_variance
        total = L_comp + L_comp_remote
        sw_lw_results[rad_type] = (; total, loc = L_comp, remote = L_comp_remote)
    end

    # Check that SW + LW adds up to total
    sw_lw_sum_local = sw_lw_results["sw"].loc + sw_lw_results["lw"].loc
    sw_lw_sum_remote = sw_lw_results["sw"].remote + sw_lw_results["lw"].remote
    if abs(L_SEP_local - sw_lw_sum_local) > 1e-5
        @warn "$region_name: SW + LW local ($(sw_lw_sum_local)) does not sum to total local ($(L_SEP_local))!"
    end
    if abs(L_SEP_remote - sw_lw_sum_remote) > 1e-5
        @warn "$region_name: SW + LW remote ($(sw_lw_sum_remote)) does not sum to total remote ($(L_SEP_remote))!"
    end

    println()
    println("SW / LW Decomposition:")
    println("-"^60)
    for rad_type in ["sw", "lw"]
        r = sw_lw_results[rad_type]
        println("  $(uppercase(rad_type)):")
        println("    Total:   $(round(r.total, digits=4)) W/m²/K")
        println("    Local:   $(round(r.loc, digits=4)) W/m²/K")
        println("    Remote:  $(round(r.remote, digits=4)) W/m²/K")
    end
    println()

    # Decompose SW and LW into clear sky and cloudy sky components
    clr_cld_var_names = Dict(
        "sw_clr" => "toa_sw_clr_t_mon", "sw_cld" => "toa_sw_cld_t_mon",
        "lw_clr" => "toa_lw_clr_t_mon", "lw_cld" => "toa_lw_cld_t_mon"
    )
    clr_cld_results = Dict{String, NamedTuple}()
    for (rad_type, var_name) in clr_cld_var_names
        R_local = ceres_spatial_means_dict[region_name][var_name]["local"]
        R_nonlocal = ceres_spatial_means_dict[region_name][var_name]["nonlocal"]
        L_comp = sep_area_fraction * cov(R_local, T_SEP) / global_temp_variance
        L_comp_remote = nonsep_area_fraction * cov(R_nonlocal, T_SEP) / global_temp_variance
        total = L_comp + L_comp_remote
        clr_cld_results[rad_type] = (; total, loc = L_comp, remote = L_comp_remote)
    end

    # Check that clear + cloudy adds up to the SW and LW totals
    
    for parent in ["sw", "lw"]
        for (component_label, field) in [("local", :loc), ("remote", :remote)]
            parent_val = getfield(sw_lw_results[parent], field)
            child_sum = getfield(clr_cld_results["$(parent)_clr"], field) + getfield(clr_cld_results["$(parent)_cld"], field)
            if abs(parent_val - child_sum) > 1e-5
                @warn "$region_name: $(uppercase(parent)) clr + cld $component_label ($(child_sum)) does not sum to $(uppercase(parent)) $component_label ($(parent_val))!"
            end
        end
    end

    println("Clear Sky / Cloudy Sky Decomposition:")
    println("-"^60)
    for rad_type in ["sw_clr", "sw_cld", "lw_clr", "lw_cld"]
        r = clr_cld_results[rad_type]
        label = replace(uppercase(rad_type), "_" => " ")
        println("  $label:")
        println("    Total:   $(round(r.total, digits=4)) W/m²/K")
        println("    Local:   $(round(r.loc, digits=4)) W/m²/K")
        println("    Remote:  $(round(r.remote, digits=4)) W/m²/K")
    end
    println()
    println("="^60)

    # Save table data for deferred plotting
    row_labels = ["Net", "SW", "LW", "SW Clear", "SW Cloud", "LW Clear", "LW Cloud"]
    col_labels = ["Total", "Local", "Nonlocal"]

    table_data = [
        L_SEP                        L_SEP_local                        L_SEP_remote;
        sw_lw_results["sw"].total    sw_lw_results["sw"].loc            sw_lw_results["sw"].remote;
        sw_lw_results["lw"].total    sw_lw_results["lw"].loc            sw_lw_results["lw"].remote;
        clr_cld_results["sw_clr"].total  clr_cld_results["sw_clr"].loc  clr_cld_results["sw_clr"].remote;
        clr_cld_results["sw_cld"].total  clr_cld_results["sw_cld"].loc  clr_cld_results["sw_cld"].remote;
        clr_cld_results["lw_clr"].total  clr_cld_results["lw_clr"].loc  clr_cld_results["lw_clr"].remote;
        clr_cld_results["lw_cld"].total  clr_cld_results["lw_cld"].loc  clr_cld_results["lw_cld"].remote
    ]

    temp_area_fraction = region_data_dict[region_name]["fractional_area_of_temp_region"]
    table_data_frac = table_data .* temp_area_fraction

    rd["table_data"] = table_data
    rd["table_data_frac"] = table_data_frac
    rd["row_labels"] = row_labels
    rd["col_labels"] = col_labels

    # Compute and save map data for L_region
    local_R = ceres_data["toa_net_all_mon"]
    L_region = cov.(vec.(eachslice(local_R; dims = (1,2))), Ref(T_SEP)) ./ global_temp_variance

    rad_mask = region_data_dict[region_name]["rad_mask"]
    temp_mask = region_data_dict[region_name]["temp_mask"]
    L_region_masked = copy(L_region)
    L_region_masked[rad_mask] .= NaN

    ceres_lat = ceres_coords["latitude"]
    ceres_lon = ceres_coords["longitude"]

    # Compute zonal means of L_region (full and remote-only)
    zonal_mean_full = vec(mean(L_region; dims=1))

    mask_count = vec(sum(.!rad_mask; dims=1))
    zonal_mean_remote = [mask_count[j] > 0 ? sum(L_region_masked[:, j][.!rad_mask[:, j]]) / mask_count[j] : NaN for j in axes(L_region, 2)]

    rd["L_region"] = L_region
    rd["L_region_masked"] = L_region_masked
    rd["zonal_mean_full"] = zonal_mean_full
    rd["zonal_mean_remote"] = zonal_mean_remote

    # Compute and save 9-panel map data (Net, SW, LW) × (All, Clear, Cloud)
    L_region_sw = cov.(vec.(eachslice(ceres_data["toa_net_sw_mon"]; dims=(1,2))), Ref(T_SEP)) ./ global_temp_variance
    L_region_lw = cov.(vec.(eachslice(ceres_data["toa_net_lw_mon"]; dims=(1,2))), Ref(T_SEP)) ./ global_temp_variance

    L_region_sw_clr = cov.(vec.(eachslice(ceres_data["toa_sw_clr_t_mon"]; dims=(1,2))), Ref(T_SEP)) ./ global_temp_variance
    L_region_lw_clr = cov.(vec.(eachslice(ceres_data["toa_lw_clr_t_mon"]; dims=(1,2))), Ref(T_SEP)) ./ global_temp_variance
    L_region_sw_cld = cov.(vec.(eachslice(ceres_data["toa_sw_cld_t_mon"]; dims=(1,2))), Ref(T_SEP)) ./ global_temp_variance
    L_region_lw_cld = cov.(vec.(eachslice(ceres_data["toa_lw_cld_t_mon"]; dims=(1,2))), Ref(T_SEP)) ./ global_temp_variance
    L_region_net_clr = L_region_sw_clr .+ L_region_lw_clr
    L_region_net_cld = L_region_sw_cld .+ L_region_lw_cld

    rd["L_panels"] = [L_region, L_region_sw, L_region_lw,
                      L_region_net_clr, L_region_sw_clr, L_region_lw_clr,
                      L_region_net_cld, L_region_sw_cld, L_region_lw_cld]
    rd["panel_titles"] = ["[$enso_abbrev] Net", "[$enso_abbrev] SW", "[$enso_abbrev] LW",
                          "[$enso_abbrev] Net Clear", "[$enso_abbrev] SW Clear", "[$enso_abbrev] LW Clear",
                          "[$enso_abbrev] Net Cloud", "[$enso_abbrev] SW Cloud", "[$enso_abbrev] LW Cloud"]

    # Compute and save correlation table data
    rad_var_map = Dict(
        "Net"      => "toa_net_all_mon",
        "SW"       => "toa_net_sw_mon",
        "LW"       => "toa_net_lw_mon",
        "SW Clear" => "toa_sw_clr_t_mon",
        "SW Cloud" => "toa_sw_cld_t_mon",
        "LW Clear" => "toa_lw_clr_t_mon",
        "LW Cloud" => "toa_lw_cld_t_mon"
    )
    corr_row_labels = ["Net", "SW", "LW", "SW Clear", "SW Cloud", "LW Clear", "LW Cloud"]
    corr_col_labels = ["Total", "Local", "Nonlocal"]

    corr_table_data = zeros(length(corr_row_labels), 3)
    for (i, rad_label) in enumerate(corr_row_labels)
        var_name = rad_var_map[rad_label]
        R_local = ceres_spatial_means_dict[region_name][var_name]["local"]
        R_nonlocal = ceres_spatial_means_dict[region_name][var_name]["nonlocal"]
        R_global_var = ceres_spatial_means_dict["global"][var_name]["global"]
        corr_table_data[i, 1] = cor(R_global_var, sst_std_detrended)
        corr_table_data[i, 2] = cor(R_local, sst_std_detrended)
        corr_table_data[i, 3] = cor(R_nonlocal, sst_std_detrended)
    end

    rd["corr_table_data"] = corr_table_data
    rd["corr_row_labels"] = corr_row_labels
    rd["corr_col_labels"] = corr_col_labels

    pd["regions"][region_name] = rd

end

# Compute and save SEPac tropical Pacific correlation table data

begin
    sepac_region = "SEPac_feedback_definition"
    sepac_visdir = joinpath(visdir, sepac_region)
    mkpath(sepac_visdir)

    rad_var_map_tpac = Dict(
        "Net"      => "toa_net_all_mon",
        "SW"       => "toa_net_sw_mon",
        "LW"       => "toa_net_lw_mon",
        "SW Clear" => "toa_sw_clr_t_mon",
        "SW Cloud" => "toa_sw_cld_t_mon",
        "LW Clear" => "toa_lw_clr_t_mon",
        "LW Cloud" => "toa_lw_cld_t_mon"
    )
    corr_row_labels_tpac = ["Net", "SW", "LW", "SW Clear", "SW Cloud", "LW Clear", "LW Cloud"]
    corr_col_labels_tpac = ["Total", "Local", "Nonlocal"]

    corr_table_tpac = zeros(length(corr_row_labels_tpac), 3)
    for (i, rad_label) in enumerate(corr_row_labels_tpac)
        var_name = rad_var_map_tpac[rad_label]
        R_local = ceres_spatial_means_dict[sepac_region][var_name]["local"]
        R_nonlocal = ceres_spatial_means_dict[sepac_region][var_name]["nonlocal"]
        R_global_var = ceres_spatial_means_dict["global"][var_name]["global"]
        corr_table_tpac[i, 1] = cor(R_global_var, sst_std_tpac_detrended)
        corr_table_tpac[i, 2] = cor(R_local, sst_std_tpac_detrended)
        corr_table_tpac[i, 3] = cor(R_nonlocal, sst_std_tpac_detrended)
    end

    pd["corr_table_tpac"] = corr_table_tpac
    pd["corr_row_labels_tpac"] = corr_row_labels_tpac
    pd["corr_col_labels_tpac"] = corr_col_labels_tpac
    pd["sepac_visdir"] = sepac_visdir
end

# --- Lagged SST regression for this ENSO mode ---

sst_current = deepcopy(original_sst)

if enso_mode == :NE
    for slice in eachslice(sst_current; dims=(1, 2))
        any(isnan, slice) && continue
        y = vec(slice)[data_valid_idx]
        residual = remove_enso_via_pls(enso_matrix_matched, y)
        slice .= NaN
        slice[data_valid_idx] .= residual
    end
    sst_current = sst_current[:, :, data_valid_idx]
elseif enso_mode == :NEA
    for slice in eachslice(sst_current; dims=(1, 2))
        any(isnan, slice) && continue
        y = vec(slice)[data_valid_idx]
        β = dot(y, enso_R_hat) / enso_R_hat_var
        residual = y .- β .* enso_R_hat
        slice .= NaN
        slice[data_valid_idx] .= residual
    end
    sst_current = sst_current[:, :, data_valid_idx]
end

pd["lag_reg"] = Dict{String, Dict{String, Any}}()
for region_name in regions_to_inspect
    T_region = era5_spatial_means_dict[region_name]["t2m"]["local"]

    println("[$enso_abbrev] Computing lagged SST regressions for $region_name...")
    reg_maps = compute_lagged_regression_maps(sst_current, T_region, regression_lags)

    lag_slices = [reg_maps[lag] for lag in regression_lags]
    lag_subtitles = [lag == 0 ? "Lag = 0 mo" :
                     lag > 0 ? "Lag = $lag mo (SST leads)" :
                     "Lag = $lag mo (T_region leads)" for lag in regression_lags]

    pd["lag_reg"][region_name] = Dict{String, Any}(
        "lag_slices" => lag_slices,
        "lag_subtitles" => lag_subtitles
    )
end

plot_data[enso_label] = pd

end # for enso_mode

# ============================================================
# Deferred plotting with shared color limits across ENSO conditions
# ============================================================

enso_labels = ["enso", "enso_removed", "enso_removed_additivity_preserved"]
enso_abbrev_map = Dict("enso" => "E", "enso_removed" => "NE", "enso_removed_additivity_preserved" => "NEA")

# Helper to compute the shared absolute maximum across multiple arrays (ignoring NaNs)
_shared_absmax(arrays) = maximum(maximum(abs.(filter(!isnan, vec(a)))) for a in arrays)

# --- Plot L_global_map with shared clims ---
L_shared_absmax = _shared_absmax([plot_data[l]["L"] for l in enso_labels])
L_shared_colornorm = colors.Normalize(vmin=-L_shared_absmax, vmax=L_shared_absmax)

for enso_label in enso_labels
    pd = plot_data[enso_label]
    L_fig = plot_global_heatmap(era5_coords["latitude"], era5_coords["longitude"], pd["L"];
        title = "[$(enso_abbrev_map[enso_label])] L: Cov(R̄, Tᵢ) / Var(T̄)  [W/m²/K]",
        colorbar_label = "W/m²/K",
        colornorm = L_shared_colornorm)
    ax_L = L_fig.get_axes()[0]
    for rname in regions_to_inspect
        add_region_contours!(ax_L,
            ceres_coords["latitude"], ceres_coords["longitude"], region_data_dict[rname]["rad_mask"],
            era5_coords["latitude"], era5_coords["longitude"], region_data_dict[rname]["temp_mask"])
    end
    L_fig.savefig(joinpath(pd["visdir"], "L_global_map.png"), dpi=300, bbox_inches="tight")
    plt.close(L_fig)
end

# --- Plot per-region figures with shared clims ---
for region_name in regions_to_inspect
    rd_pair = [plot_data[l]["regions"][region_name] for l in enso_labels]

    # Compute shared clims for each plot type across ENSO conditions
    shared_table_clim = _shared_absmax([rd["table_data"] for rd in rd_pair])
    shared_table_frac_clim = _shared_absmax([rd["table_data_frac"] for rd in rd_pair])
    shared_Lregion_absmax = _shared_absmax([rd["L_region"] for rd in rd_pair])
    shared_Lregion_colornorm = colors.Normalize(vmin=-shared_Lregion_absmax, vmax=shared_Lregion_absmax)
    shared_3panel_absmax = _shared_absmax(vcat([rd["L_panels"] for rd in rd_pair]...))
    shared_3panel_colornorm = colors.Normalize(vmin=-shared_3panel_absmax, vmax=shared_3panel_absmax)
    shared_corr_clim = _shared_absmax([rd["corr_table_data"] for rd in rd_pair])

    ceres_lat = ceres_coords["latitude"]
    ceres_lon = ceres_coords["longitude"]
    rad_mask = region_data_dict[region_name]["rad_mask"]
    temp_mask = region_data_dict[region_name]["temp_mask"]
    central_lon = region_name == "SEAtl" ? 0 : 180

    for enso_label in enso_labels
        rd = plot_data[enso_label]["regions"][region_name]
        region_visdir = rd["region_visdir"]

        # L decomposition table
        plot_table_heatmap(rd["table_data"], rd["row_labels"], rd["col_labels"];
            title="[$(enso_abbrev_map[enso_label])] L Decomposition: $region_name  [W/m²/K]",
            colorbar_label="W/m²/K", digits=1,
            clim=shared_table_clim,
            savepath=joinpath(region_visdir, "L_decomposition_table.png"))

        # L decomposition table scaled by area fraction
        plot_table_heatmap(rd["table_data_frac"], rd["row_labels"], rd["col_labels"];
            title="[$(enso_abbrev_map[enso_label])] L × Area Fraction: $region_name  [W/m²/K]",
            colorbar_label="W/m²/K", digits=3,
            clim=shared_table_frac_clim,
            savepath=joinpath(region_visdir, "L_decomposition_table_area_fraction.png"))

        # L_region map with zonal mean side panel
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

        ax_map = fig.add_subplot(gs[0], projection=ccrs.Robinson(central_longitude=central_lon))
        c = plot_global_heatmap_on_ax!(ax_map, ceres_lat, ceres_lon, rd["L_region"];
            colornorm=shared_Lregion_colornorm, title="[$(enso_abbrev_map[enso_label])] L_$(region_name): Cov(Rᵢ, T_region) / Var(T̄)")
        add_region_contours!(ax_map, ceres_lat, ceres_lon, rad_mask, era5_coords["latitude"], era5_coords["longitude"], temp_mask)

        ax_zonal = fig.add_subplot(gs[1])
        ax_zonal.plot(rd["zonal_mean_full"], ceres_lat, "k-", linewidth=1.5, label="Full")
        ax_zonal.plot(rd["zonal_mean_remote"], ceres_lat, "k--", linewidth=1.5, label="Remote only")
        ax_zonal.set_ylim(-90, 90)
        ax_zonal.set_ylabel("Latitude")
        ax_zonal.set_xlabel("W/m²/K")
        ax_zonal.legend(fontsize=8)
        ax_zonal.axvline(0, color="gray", linewidth=0.5, linestyle=":")
        ax_zonal.set_title("Zonal Mean")

        plt.colorbar(cm.ScalarMappable(norm=shared_Lregion_colornorm, cmap=cmr.prinsenvlag.reversed()), ax=[ax_map, ax_zonal], orientation="horizontal", label="W/m²/K", pad=0.08, shrink=0.8)

        fig.savefig(joinpath(region_visdir, "L_map.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 9-panel Net/SW/LW × All/Clear/Cloud maps
        fig_3 = plot_multiple_levels_rowmajor(ceres_lat, ceres_lon, rd["L_panels"], (3, 3);
            subtitles=rd["panel_titles"],
            colorbar_label="W/m²/K",
            colornorm=shared_3panel_colornorm,
            proj=ccrs.Robinson(central_longitude=central_lon))
        for ax_panel in fig_3.get_axes()
            try add_region_contours!(ax_panel, ceres_lat, ceres_lon, rad_mask, era5_coords["latitude"], era5_coords["longitude"], temp_mask) catch end
        end

        fig_3.suptitle("[$(enso_abbrev_map[enso_label])] L Decomposition: $region_name", fontsize=14, y=1.02)
        fig_3.savefig(joinpath(region_visdir, "L_net_sw_lw_maps.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_3)

        # Correlation table
        plot_table_heatmap(rd["corr_table_data"], rd["corr_row_labels"], rd["corr_col_labels"];
            title="[$(enso_abbrev_map[enso_label])] Cor(Radiation, Tropical SST σ): $region_name",
            colorbar_label="Correlation", digits=2,
            clim=shared_corr_clim,
            savepath=joinpath(region_visdir, "radiation_sst_std_correlation_table.png"))
    end
end

# --- SEPac tropical Pacific correlation table with shared clims ---
begin
    sepac_region = "SEPac_feedback_definition"
    shared_corr_tpac_clim = _shared_absmax([plot_data[l]["corr_table_tpac"] for l in enso_labels])

    for enso_label in enso_labels
        pd = plot_data[enso_label]
        plot_table_heatmap(pd["corr_table_tpac"], pd["corr_row_labels_tpac"], pd["corr_col_labels_tpac"];
            title="[$(enso_abbrev_map[enso_label])] Cor(SEPac Radiation, Trop Pac SST σ)",
            colorbar_label="Correlation", digits=2,
            clim=shared_corr_tpac_clim,
            savepath=joinpath(pd["sepac_visdir"], "radiation_tpac_sst_std_correlation_table.png"))
    end
end

# --- Plot lagged SST regressions with shared color limits ---
for region_name in regions_to_inspect
    rd_lag_pair = [plot_data[l]["lag_reg"][region_name] for l in enso_labels]
    shared_lag_absmax = _shared_absmax(vcat([rd["lag_slices"] for rd in rd_lag_pair]...))
    shared_lag_colornorm = colors.Normalize(vmin=-shared_lag_absmax, vmax=shared_lag_absmax)

    for enso_label in enso_labels
        rd_lag = plot_data[enso_label]["lag_reg"][region_name]
        regdir = joinpath(plot_data[enso_label]["visdir"], region_name)
        mkpath(regdir)

        fig = plot_multiple_levels_rowmajor(sst_lat, sst_lon, rd_lag["lag_slices"], (3, 3);
            subtitles=rd_lag["lag_subtitles"],
            colorbar_label="SST regression coef (K / K)",
            colornorm=shared_lag_colornorm)
        fig.suptitle("[$(enso_abbrev_map[enso_label])] ERA5 SST regressed on $region_name T index", fontsize=14, y=1.02)
        fig.savefig(joinpath(regdir, "sst_lagged_regression_on_T_index.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        println("  Saved: $regdir/sst_lagged_regression_on_T_index.png")
    end
end

