# ENSO Decomposition of T2M and Net Radiation into Lag Components
#
# Decomposes t2m and net radiation into ENSO and non-ENSO components via PLS,
# then further decomposes each ENSO component into 49 individual lag contributions
# (one per column of the ONI lag matrix, lags -24:24).
#
# For each variable (t2m, net radiation):
#   - Gridwise:    pointwise PLS → ENSO field, residual field, 49 coefficient maps
#   - Global mean: scalar PLS    → ENSO time series, residual, 49 lag time series
#
# The 49 lag contributions sum to the total ENSO component (up to a negligible
# additive mean ≈ 0 from deseasonalized/detrended data).

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")

using JLD2, Dates, Statistics

# ============================================================
# Helper Functions: PLS Lag Decomposition
# ============================================================

"""
    extract_pls_coefficients(pls_model)

Extract the effective regression coefficient vector from a 1-component PLS model.
Maps normalized X columns to normalized Y.  Returns a length-`n_lags` vector.
"""
function extract_pls_coefficients(pls_model)
    P = pls_model.X_weights * pinv(pls_model.X_loadings' * pls_model.X_weights)
    return vec(P * pls_model.Y_loadings')
end

"""
    enso_decompose_scalar_full(enso_matrix, y)

Decompose scalar time series `y` into:
  - `enso`: total ENSO component (length `n_time`)
  - `resid`: non-ENSO residual (length `n_time`)
  - `lag_contributions`: `(n_time × 49)` matrix — column `j` is lag `j`'s contribution

The columns of `lag_contributions` sum to `enso` (plus a negligible mean offset).
"""
function enso_decompose_scalar_full(enso_matrix::Matrix, y::AbstractVector)
    pls_model = make_pls_regressor(enso_matrix, y, 1; print_updates=false)
    enso_comp = vec(predict(pls_model, enso_matrix))
    resid = y .- enso_comp

    # Normalized ENSO matrix and PLS coefficients
    X_norm = copy(enso_matrix)
    normalize_input_precalculated!(X_norm, pls_model.X_means, pls_model.X_stds)
    coeffs = extract_pls_coefficients(pls_model)
    Y_std = pls_model.Y_stds[1]

    # contribution_j(t) = X_norm[t, j] * coeffs[j] * Y_std
    lag_contributions = X_norm .* coeffs' .* Y_std

    return (; enso=enso_comp, resid=resid, lag_contributions=lag_contributions)
end

"""
    pointwise_pls_with_lag_decomposition(enso_matrix, Y_3d; n_components=1)

Pointwise PLS on a `(lon × lat × time)` field.  Returns a NamedTuple:
  - `enso`:           ENSO component field `(lon × lat × time)`
  - `resid`:          residual field `(lon × lat × time)`
  - `lag_coeff_maps`: PLS coefficient at each grid point per lag `(lon × lat × 49)`
  - `y_stds_map`:     Y standard deviation at each grid point `(lon × lat)`
  - `y_means_map`:    Y mean at each grid point `(lon × lat)`
  - `X_norm`:         normalized ENSO matrix `(n_time × 49)`, shared across all grid points

To reconstruct lag `j`'s contribution at point `(i, k)`:
    `X_norm[:, j] .* lag_coeff_maps[i, k, j] .* y_stds_map[i, k]`
"""
function pointwise_pls_with_lag_decomposition(enso_matrix::Matrix, Y_3d::AbstractArray{<:Real, 3};
        n_components::Int=1, meanfunc=mean, stdfunc=my_std_func)

    n_lon, n_lat, n_time = size(Y_3d)
    n_lags = size(enso_matrix, 2)

    enso_3d        = similar(Y_3d)
    lag_coeff_maps = zeros(n_lon, n_lat, n_lags)
    y_stds_map     = zeros(n_lon, n_lat)
    y_means_map    = zeros(n_lon, n_lat)
    X_norm         = nothing  # computed once from first PLS model

    total_points = n_lon * n_lat
    for (count, I) in enumerate(CartesianIndices((n_lon, n_lat)))
        i, j = Tuple(I)
        y_ts = Y_3d[i, j, :]

        pls_model = make_pls_regressor(enso_matrix, y_ts, n_components;
                                       print_updates=false, meanfunc=meanfunc, stdfunc=stdfunc)
        enso_3d[i, j, :] .= vec(predict(pls_model, enso_matrix))

        lag_coeff_maps[i, j, :] .= extract_pls_coefficients(pls_model)
        y_stds_map[i, j]  = pls_model.Y_stds[1]
        y_means_map[i, j] = pls_model.Y_means[1]

        if X_norm === nothing
            X_norm = copy(enso_matrix)
            normalize_input_precalculated!(X_norm, pls_model.X_means, pls_model.X_stds)
        end

        if count % 10000 == 0
            println("  Processed $count / $total_points grid points")
        end
    end

    resid_3d = Y_3d .- enso_3d

    return (; enso=enso_3d, resid=resid_3d,
              lag_coeff_maps=lag_coeff_maps, y_stds_map=y_stds_map,
              y_means_map=y_means_map, X_norm=X_norm)
end

"""
    reconstruct_gridded_lag_contribution(decomp, lag_idx)

Reconstruct the full `(lon × lat × time)` contribution of a single lag
from the output of `pointwise_pls_with_lag_decomposition`.
"""
function reconstruct_gridded_lag_contribution(decomp::NamedTuple, lag_idx::Int)
    (; lag_coeff_maps, y_stds_map, X_norm) = decomp
    n_lon, n_lat = size(y_stds_map)
    n_time = size(X_norm, 1)
    x_col = X_norm[:, lag_idx]

    contrib = zeros(n_lon, n_lat, n_time)
    for j in 1:n_lat, i in 1:n_lon
        contrib[i, j, :] .= x_col .* lag_coeff_maps[i, j, lag_idx] .* y_stds_map[i, j]
    end
    return contrib
end

"""
    reconstruct_gridded_lag_global_mean(decomp, latitudes, lag_idx)

Reconstruct the cos-weighted global-mean time series of a single lag's
contribution to the ENSO component.  Returns a length-`n_time` vector.
"""
function reconstruct_gridded_lag_global_mean(decomp::NamedTuple, latitudes, lag_idx::Int)
    (; lag_coeff_maps, y_stds_map, X_norm) = decomp
    eff_coeff = lag_coeff_maps[:, :, lag_idx] .* y_stds_map  # (lon × lat)
    cos_w = cosd.(latitudes')
    mean_coeff = sum(eff_coeff .* cos_w) / sum(ones(size(eff_coeff, 1)) .* cos_w)
    return X_norm[:, lag_idx] .* mean_coeff
end

"""
    reconstruct_point_lag_contributions(decomp, lon_idx, lat_idx)

Reconstruct all 49 lag contribution time series at a single grid point.
Returns an `(n_time × 49)` matrix (same layout as the scalar version).
"""
function reconstruct_point_lag_contributions(decomp::NamedTuple, lon_idx::Int, lat_idx::Int)
    (; lag_coeff_maps, y_stds_map, X_norm) = decomp
    coeffs = lag_coeff_maps[lon_idx, lat_idx, :]
    y_std  = y_stds_map[lon_idx, lat_idx]
    return X_norm .* coeffs' .* y_std
end

"""
    compute_global_mean_lag_contributions(decomp, latitudes)

From a gridded decomposition, compute the cos-weighted global-mean
contribution of each of the 49 lags.  Returns an `(n_time × 49)` matrix
whose columns sum (across lags) to the global-mean ENSO component.
"""
function compute_global_mean_lag_contributions(decomp::NamedTuple, latitudes)
    (; lag_coeff_maps, y_stds_map, X_norm) = decomp
    n_lags = size(X_norm, 2)
    n_lon, n_lat = size(y_stds_map)

    # Effective coefficient map in original Y units per unit normalized X
    eff_coeff = lag_coeff_maps .* y_stds_map  # (lon × lat × 49)

    # Cos-weighted spatial mean of effective coefficients for each lag
    cos_w = cosd.(latitudes')  # (1 × n_lat) for broadcasting
    denom = sum(ones(n_lon) .* cos_w)
    mean_coeffs = zeros(n_lags)
    for k in 1:n_lags
        mean_coeffs[k] = sum(eff_coeff[:, :, k] .* cos_w) / denom
    end

    # Global-mean lag contributions = X_norm * diag(mean_coeffs)
    return X_norm .* mean_coeffs'
end

# ============================================================
# Load Data
# ============================================================

println("Loading ERA5 t2m...")
date_range = (Date(2000, 3), Date(2024, 2, 28))
era5_data, era5_coords = load_era5_data(["t2m"], date_range)

println("Loading CERES net radiation...")
ceres_vars_to_load = ["toa_net_all_mon"]
ceres_data, ceres_coords = load_new_ceres_data(ceres_vars_to_load, date_range)

ceres_coords["time"] = round_dates_down_to_nearest_month(ceres_coords["time"])
common_time = ceres_coords["time"]
@assert all(common_time .== round_dates_down_to_nearest_month(era5_coords["time"])) "ERA5 / CERES time mismatch!"

# Deseasonalize and detrend twice
println("Deseasonalizing and detrending...")
float_time = calc_float_time.(common_time)
month_groups = groupfind(month, common_time)
for datadict in [era5_data, ceres_data]
    for var_data in datadict
        for slice in eachslice(var_data; dims=(1, 2))
            deseasonalize_and_detrend_precalculated_groups_twice!(slice, float_time, month_groups;
                aggfunc=mean, trendfunc=least_squares_fit)
        end
    end
end

# Load and prepare ENSO lag matrix
println("Loading ENSO data...")
enso_data, enso_dates_raw = load_enso_data(date_range)
enso_times = round_dates_down_to_nearest_month(enso_dates_raw["time"])
lags = -24:24
lag_columns = ["oni_lag_$lag" for lag in lags]
n_enso = length(enso_times)
enso_lag_matrix_full = Matrix{Union{Float64, Missing}}(missing, n_enso, length(lag_columns))
for (j, col) in enumerate(lag_columns)
    haskey(enso_data, col) && (enso_lag_matrix_full[:, j] .= enso_data[col])
end
valid_enso_rows = [all(!ismissing, enso_lag_matrix_full[i, :]) for i in 1:n_enso]
enso_times_valid = enso_times[valid_enso_rows]
enso_lag_matrix = Float64.(enso_lag_matrix_full[valid_enso_rows, :])
enso_matrix_matched, data_valid_idx = precompute_enso_match(common_time, enso_times_valid, enso_lag_matrix)

println("ENSO matrix: $(size(enso_matrix_matched)) ($(length(lags)) lags)")

# ============================================================
# Gridwise ENSO Decomposition — T2M
# ============================================================

println("\n=== Gridwise PLS decomposition of T2M ===")
T_valid = era5_data["t2m"][:, :, data_valid_idx]
t2m_gridded_decomp = pointwise_pls_with_lag_decomposition(enso_matrix_matched, T_valid)
println("  T2M gridwise decomposition complete.")
println("  ENSO field size:   $(size(t2m_gridded_decomp.enso))")
println("  Coeff maps size:   $(size(t2m_gridded_decomp.lag_coeff_maps))")

# ============================================================
# Gridwise ENSO Decomposition — Net Radiation
# ============================================================

println("\n=== Gridwise PLS decomposition of Net Radiation ===")
R_valid = ceres_data["toa_net_all_mon"][:, :, data_valid_idx]
netrad_gridded_decomp = pointwise_pls_with_lag_decomposition(enso_matrix_matched, R_valid)
println("  Net Radiation gridwise decomposition complete.")
println("  ENSO field size:   $(size(netrad_gridded_decomp.enso))")
println("  Coeff maps size:   $(size(netrad_gridded_decomp.lag_coeff_maps))")

# ============================================================
# Global Mean ENSO Decomposition — T2M
# ============================================================

println("\n=== Global mean PLS decomposition of T2M ===")
t2m_global_mean = generate_spatial_mean(era5_data["t2m"], era5_coords["latitude"])
t2m_global_valid = t2m_global_mean[data_valid_idx]
t2m_scalar_decomp = enso_decompose_scalar_full(enso_matrix_matched, t2m_global_valid)
println("  T2M global mean decomposition complete.")
println("  Lag contributions size: $(size(t2m_scalar_decomp.lag_contributions))")
println("  Reconstruction check (max abs error): ",
    round(maximum(abs.(sum(t2m_scalar_decomp.lag_contributions; dims=2) .- t2m_scalar_decomp.enso)); sigdigits=3))

# ============================================================
# Global Mean ENSO Decomposition — Net Radiation
# ============================================================

println("\n=== Global mean PLS decomposition of Net Radiation ===")
netrad_global_mean = generate_spatial_mean(ceres_data["toa_net_all_mon"], ceres_coords["latitude"])
netrad_global_valid = netrad_global_mean[data_valid_idx]
netrad_scalar_decomp = enso_decompose_scalar_full(enso_matrix_matched, netrad_global_valid)
println("  Net Radiation global mean decomposition complete.")
println("  Lag contributions size: $(size(netrad_scalar_decomp.lag_contributions))")
println("  Reconstruction check (max abs error): ",
    round(maximum(abs.(sum(netrad_scalar_decomp.lag_contributions; dims=2) .- netrad_scalar_decomp.enso)); sigdigits=3))

# ============================================================
# Global Mean of Gridded Lag Contributions
# ============================================================

println("\n=== Computing global-mean lag contributions from gridded decomposition ===")
t2m_gridded_global_lag_contribs = compute_global_mean_lag_contributions(
    t2m_gridded_decomp, era5_coords["latitude"])
netrad_gridded_global_lag_contribs = compute_global_mean_lag_contributions(
    netrad_gridded_decomp, ceres_coords["latitude"])

println("  T2M gridded → global mean lag contributions: $(size(t2m_gridded_global_lag_contribs))")
println("  Net Rad gridded → global mean lag contributions: $(size(netrad_gridded_global_lag_contribs))")

# ============================================================
# Summary
# ============================================================

println("\n" * "="^60)
println("DECOMPOSITION SUMMARY")
println("="^60)
println("Variables: t2m (ERA5), toa_net_all_mon (CERES)")
println("Time period: $(date_range[1]) to $(date_range[2])")
println("ENSO lags: $(first(lags)):$(last(lags)) ($(length(lags)) columns)")
println()
println("Gridded decompositions:")
println("  t2m_gridded_decomp      — .enso, .resid, .lag_coeff_maps, .y_stds_map, .y_means_map, .X_norm")
println("  netrad_gridded_decomp   — same fields")
println()
println("Scalar (global mean) decompositions:")
println("  t2m_scalar_decomp       — .enso, .resid, .lag_contributions (n_time × 49)")
println("  netrad_scalar_decomp    — same fields")
println()
println("Gridded → global-mean lag contributions:")
println("  t2m_gridded_global_lag_contribs    — (n_time × 49)")
println("  netrad_gridded_global_lag_contribs — (n_time × 49)")
println()
println("Helper functions:")
println("  reconstruct_gridded_lag_contribution(decomp, lag_idx) → (lon × lat × time)")
println("  reconstruct_point_lag_contributions(decomp, lon_idx, lat_idx) → (n_time × 49)")
println("  compute_global_mean_lag_contributions(decomp, latitudes) → (n_time × 49)")
println("="^60)

# ============================================================
# λ = Cov(R̄, T̄) / Var(T̄)  —  ENSO / non-ENSO decomposition
# ============================================================

@py import matplotlib.pyplot as plt

println("\n=== Plotting λ decomposition (5-panel) ===")

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/investigate_pls_lingering_enso"
mkpath(visdir)

# Extract the four components
T_E  = t2m_scalar_decomp.enso
T_NE = t2m_scalar_decomp.resid
R_E  = netrad_scalar_decomp.enso
R_NE = netrad_scalar_decomp.resid

std_T = std(t2m_global_valid)
var_T = std_T^2

# Define the 5 panels: (R_component, T_component, title_suffix)
panel_specs = [
    (netrad_global_valid, t2m_global_valid, "Total"),
    (R_E,                 T_E,              "R̄_E · T̄_E"),
    (R_E,                 T_NE,             "R̄_E · T̄_NE"),
    (R_NE,                T_E,              "R̄_NE · T̄_E"),
    (R_NE,                T_NE,             "R̄_NE · T̄_NE"),
]

# Time axis for plotting
valid_times = common_time[data_valid_idx]
time_floats = calc_float_time.(valid_times)

fig, axs = plt.subplots(1, 5, figsize=(22, 3.5), sharey=true)

for (idx, (R_comp, T_comp, label)) in enumerate(panel_specs)
    ax = axs[idx - 1]

    # Fully normalize each component to unit variance
    R_normed = R_comp ./ std(R_comp)
    T_normed = T_comp ./ std(T_comp)

    # Actual contribution and magnitude scale
    λ_val = cov(R_comp, T_comp) / var_T
    mag = std(R_comp) * std(T_comp) / var_T

    ax.plot(time_floats, T_normed, linewidth=0.8, alpha=0.85, label="T̄ component")
    ax.plot(time_floats, R_normed, linewidth=0.8, alpha=0.85, label="R̄ component")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Year", fontsize=8)

    # Annotate with λ value and magnitude scale
    ax.text(0.03, 0.97, "λ = $(round(λ_val; digits=3))\nσ_R σ_T / σ²_T̄ = $(round(mag; digits=3))",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"wheat", "alpha"=>0.8))

    if idx == 1
        ax.set_ylabel("Normalized anomaly", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    end
end

fig.suptitle("λ = Cov(R̄, T̄) / Var(T̄)  —  ENSO Decomposition", fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig(joinpath(visdir, "lambda_enso_decomposition_5panel.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

println("  Saved: $(joinpath(visdir, "lambda_enso_decomposition_5panel.png"))")

# Print numerical summary
println("\n  λ decomposition values:")
for (_, (R_comp, T_comp, label)) in enumerate(panel_specs)
    λ_val = cov(R_comp, T_comp) / var_T
    mag   = std(R_comp) * std(T_comp) / var_T
    println("    $label:  λ = $(round(λ_val; digits=4)),  σ_R·σ_T/σ²_T̄ = $(round(mag; digits=4)) W/m²/K")
end
λ_sum = sum(cov(R, T) / var_T for (R, T, _) in panel_specs[2:end])
println("    Sum of 4 terms: $(round(λ_sum; digits=4)) W/m²/K")

# ============================================================
# 7×7 table heatmaps of lag contributions to cross-terms
# ============================================================

@py import matplotlib.colors as colors

"""
    plot_7x7_lag_table(lag_ts_matrix, fixed_ts, var_T, lags; suptitle, savepath)

Plot a 7×7 heatmap table where each cell corresponds to one ONI lag and
shows Cov(lag_j, fixed) / Var(T̄)  [W/m²/K].
"""
function plot_7x7_lag_table(lag_ts_matrix, fixed_ts, var_T, lags; suptitle, savepath)
    n_lags = length(lags)
    flat_values = [cov(lag_ts_matrix[:, j], fixed_ts) / var_T for j in 1:n_lags]
    total_cov = round(sum(flat_values); digits=3)
    values = reshape(flat_values, 7, 7)'

    absmax = maximum(abs.(values))
    norm = colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(values, cmap=DAVES_CMAP, norm=norm, aspect="auto")

    lag_strs = permutedims(reshape(["$(l)" for l in lags], 7, 7))
    for i in 0:6, j in 0:6
        val = values[i+1, j+1]
        text_color = abs(val) > 0.6 * absmax ? "white" : "black"
        ax.text(j, i, "lag $(lag_strs[i+1, j+1])\n$(round(val; digits=3))",
                ha="center", va="center", fontsize=7, color=text_color, fontweight="bold")
    end

    ax.set_xticks(pylist(Int[]))
    ax.set_yticks(pylist(Int[]))
    ax.set_title("$suptitle\nTotal = $total_cov W/m²/K", fontsize=11, pad=12)
    plt.colorbar(im, ax=ax, orientation="horizontal", label="Cov / Var(T̄)  [W/m²/K]",
                 shrink=0.8, pad=0.06)

    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
end

# Table 1: T̄_E (49 lag components) × R̄_NE
println("\n=== 7×7 table: T̄_E lag components × R̄_NE ===")
plot_7x7_lag_table(
    t2m_scalar_decomp.lag_contributions, R_NE,
    var_T, collect(lags);
    suptitle = "Cov(T̄_E_lag_j, R̄_NE) / Var(T̄)  [W/m²/K]",
    savepath = joinpath(visdir, "lambda_T_E_lags_x_R_NE_7x7_table.png"))
println("  Saved: $(joinpath(visdir, "lambda_T_E_lags_x_R_NE_7x7_table.png"))")

# Table 2: R̄_E (49 lag components) × T̄_NE
println("\n=== 7×7 table: R̄_E lag components × T̄_NE ===")
plot_7x7_lag_table(
    netrad_scalar_decomp.lag_contributions, T_NE,
    var_T, collect(lags);
    suptitle = "Cov(R̄_E_lag_j, T̄_NE) / Var(T̄)  [W/m²/K]",
    savepath = joinpath(visdir, "lambda_R_E_lags_x_T_NE_7x7_table.png"))
println("  Saved: $(joinpath(visdir, "lambda_R_E_lags_x_T_NE_7x7_table.png"))")

# ============================================================
# Same 3 plots but using cos-weighted averages of pointwise
# decomposed fields (gridded decomp → global mean)
# ============================================================

println("\n=== Gridded-then-averaged λ decomposition ===")

# Cos-weighted global means of pointwise total, ENSO, and residual fields
T_gw_total = generate_spatial_mean(T_valid, era5_coords["latitude"])
T_gw_E     = generate_spatial_mean(t2m_gridded_decomp.enso, era5_coords["latitude"])
T_gw_NE    = generate_spatial_mean(t2m_gridded_decomp.resid, era5_coords["latitude"])

R_gw_total = generate_spatial_mean(R_valid, ceres_coords["latitude"])
R_gw_E     = generate_spatial_mean(netrad_gridded_decomp.enso, ceres_coords["latitude"])
R_gw_NE    = generate_spatial_mean(netrad_gridded_decomp.resid, ceres_coords["latitude"])

std_T_gw = std(T_gw_total)
var_T_gw = std_T_gw^2

# --- 5-panel plot ---
gw_panel_specs = [
    (R_gw_total, T_gw_total, "Total"),
    (R_gw_E,     T_gw_E,     "R̄_E · T̄_E"),
    (R_gw_E,     T_gw_NE,    "R̄_E · T̄_NE"),
    (R_gw_NE,    T_gw_E,     "R̄_NE · T̄_E"),
    (R_gw_NE,    T_gw_NE,    "R̄_NE · T̄_NE"),
]

fig, axs = plt.subplots(1, 5, figsize=(22, 3.5), sharey=true)

for (idx, (R_comp, T_comp, label)) in enumerate(gw_panel_specs)
    ax = axs[idx - 1]

    R_normed = R_comp ./ std(R_comp)
    T_normed = T_comp ./ std(T_comp)

    λ_val = cov(R_comp, T_comp) / var_T_gw
    mag = std(R_comp) * std(T_comp) / var_T_gw

    ax.plot(time_floats, T_normed, linewidth=0.8, alpha=0.85, label="T̄ component")
    ax.plot(time_floats, R_normed, linewidth=0.8, alpha=0.85, label="R̄ component")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Year", fontsize=8)

    ax.text(0.03, 0.97, "λ = $(round(λ_val; digits=3))\nσ_R σ_T / σ²_T̄ = $(round(mag; digits=3))",
            transform=ax.transAxes, fontsize=8, verticalalignment="top",
            bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"wheat", "alpha"=>0.8))

    if idx == 1
        ax.set_ylabel("Normalized anomaly", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    end
end

fig.suptitle("λ = Cov(R̄, T̄) / Var(T̄)  —  ENSO Decomposition (gridded → global mean)", fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig(joinpath(visdir, "lambda_enso_decomposition_5panel_gridded.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
println("  Saved 5-panel (gridded)")

println("\n  λ decomposition values (gridded → global mean):")
for (_, (R_comp, T_comp, label)) in enumerate(gw_panel_specs)
    λ_val = cov(R_comp, T_comp) / var_T_gw
    mag   = std(R_comp) * std(T_comp) / var_T_gw
    println("    $label:  λ = $(round(λ_val; digits=4)),  σ_R·σ_T/σ²_T̄ = $(round(mag; digits=4)) W/m²/K")
end
λ_sum_gw = sum(cov(R, T) / var_T_gw for (R, T, _) in gw_panel_specs[2:end])
println("    Sum of 4 terms: $(round(λ_sum_gw; digits=4)) W/m²/K")

# --- 7×7 tables using gridded lag contributions ---
# T̄_E lag components (from gridded) × R̄_NE (from gridded)
println("\n=== 7×7 table (gridded): T̄_E lag components × R̄_NE ===")
plot_7x7_lag_table(
    t2m_gridded_global_lag_contribs, R_gw_NE,
    var_T_gw, collect(lags);
    suptitle = "Cov(⟨T_E_lag_j⟩, ⟨R_NE⟩) / Var(⟨T⟩)  [W/m²/K]  (gridded)",
    savepath = joinpath(visdir, "lambda_T_E_lags_x_R_NE_7x7_table_gridded.png"))
println("  Saved 7×7 table (gridded T_E lags × R_NE)")

# R̄_E lag components (from gridded) × T̄_NE (from gridded)
println("\n=== 7×7 table (gridded): R̄_E lag components × T̄_NE ===")
plot_7x7_lag_table(
    netrad_gridded_global_lag_contribs, T_gw_NE,
    var_T_gw, collect(lags);
    suptitle = "Cov(⟨R_E_lag_j⟩, ⟨T_NE⟩) / Var(⟨T⟩)  [W/m²/K]  (gridded)",
    savepath = joinpath(visdir, "lambda_R_E_lags_x_T_NE_7x7_table_gridded.png"))
println("  Saved 7×7 table (gridded R_E lags × T_NE)")

# ============================================================
# Power Spectra: T and R, Full Variable and Components
# ============================================================

using FFTW

"""
    compute_power_spectrum(signal; dt=1.0)

Compute the one-sided power spectral density of a signal.
Returns (frequencies, power) where frequencies are in cycles per unit time.
"""
function compute_power_spectrum(signal; dt=1.0)
    n = length(signal)
    
    # Remove mean
    signal_centered = signal .- mean(signal)
    
    # Compute FFT
    fft_result = fft(signal_centered)
    
    # Compute power (one-sided)
    power = abs2.(fft_result[1:div(n, 2)+1])
    power[2:end-1] .*= 2  # Double power for non-DC, non-Nyquist frequencies
    power ./= n^2  # Normalize
    
    # Frequency array
    freqs = fftfreq(n, 1/dt)[1:div(n, 2)+1]
    
    return freqs, power
end

println("\n=== Computing power spectra ===")

# Time step in years (monthly data = 1/12 year)
dt_years = 1.0 / 12.0

# Compute power spectra for all components
println("  Computing T power spectra...")
freqs_T, ps_T_total = compute_power_spectrum(t2m_global_valid; dt=dt_years)
_, ps_T_E_scalar = compute_power_spectrum(T_E; dt=dt_years)
_, ps_T_NE_scalar = compute_power_spectrum(T_NE; dt=dt_years)
_, ps_T_E_gridded = compute_power_spectrum(T_gw_E; dt=dt_years)
_, ps_T_NE_gridded = compute_power_spectrum(T_gw_NE; dt=dt_years)

println("  Computing R power spectra...")
freqs_R, ps_R_total = compute_power_spectrum(netrad_global_valid; dt=dt_years)
_, ps_R_E_scalar = compute_power_spectrum(R_E; dt=dt_years)
_, ps_R_NE_scalar = compute_power_spectrum(R_NE; dt=dt_years)
_, ps_R_E_gridded = compute_power_spectrum(R_gw_E; dt=dt_years)
_, ps_R_NE_gridded = compute_power_spectrum(R_gw_NE; dt=dt_years)

println("\n=== Plotting 10-panel power spectra ===")

fig, axs = plt.subplots(2, 5, figsize=(24, 8))

# Convert frequencies to periods (in years) for plotting
periods_T = 1.0 ./ freqs_T[2:end]  # Skip DC component
periods_R = 1.0 ./ freqs_R[2:end]

# Temperature row (top)
temp_specs = [
    (ps_T_total, "Total T̄"),
    (ps_T_E_scalar, "T̄_ENSO (scalar)"),
    (ps_T_NE_scalar, "T̄_resid (scalar)"),
    (ps_T_E_gridded, "T̄_ENSO (gridded)"),
    (ps_T_NE_gridded, "T̄_resid (gridded)")
]

for (idx, (ps, label)) in enumerate(temp_specs)
    ax = axs[0, idx - 1]
    ax.loglog(periods_T, ps[2:end], linewidth=1.2, color="C0")
    ax.set_xlabel("Period (years)", fontsize=9)
    ax.set_ylabel("Power Spectral Density (K²)", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.grid(true, which="both", alpha=0.3)
    ax.set_xlim(periods_T[end], periods_T[1])  # Reverse x-axis for period
end

# Radiation row (bottom)
rad_specs = [
    (ps_R_total, "Total R̄"),
    (ps_R_E_scalar, "R̄_ENSO (scalar)"),
    (ps_R_NE_scalar, "R̄_resid (scalar)"),
    (ps_R_E_gridded, "R̄_ENSO (gridded)"),
    (ps_R_NE_gridded, "R̄_resid (gridded)")
]

for (idx, (ps, label)) in enumerate(rad_specs)
    ax = axs[1, idx - 1]
    ax.loglog(periods_R, ps[2:end], linewidth=1.2, color="C1")
    ax.set_xlabel("Period (years)", fontsize=9)
    ax.set_ylabel("Power Spectral Density (W²/m⁴)", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.grid(true, which="both", alpha=0.3)
    ax.set_xlim(periods_R[end], periods_R[1])  # Reverse x-axis for period
end

fig.suptitle("Power Spectra: Global Mean T and R — Full Variable and Decomposition Components", 
             fontsize=14, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(joinpath(visdir, "power_spectra_T_R_decomposition_10panel.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

println("  Saved: $(joinpath(visdir, "power_spectra_T_R_decomposition_10panel.png"))")

println("\n=== Power spectrum analysis complete ===")
