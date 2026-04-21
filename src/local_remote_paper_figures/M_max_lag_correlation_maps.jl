# M_max_lag_correlation_maps.jl
#
# For each region (SEPac, NEPac, SEAtl) and for ENSO (ONI index as the "region"),
# produces one multi-pane global map showing, at each gridpoint, the lag
# (in months) of T_region that maximises the Pearson correlation between
# that gridpoint's anomaly time series and the lagged regional temperature.
#
# Panels (3 × 4 = 12):
#   Row 1: ERA5 T @ 1000 / 850 / 700 / 500 hPa
#   Row 2: ERA5 T @ 250 hPa | CERES Net | CERES SW | CERES LW
#   Row 3: CERES SW_clr | CERES LW_clr | CERES SW_cld (CRE) | CERES LW_cld (CRE)
#
# Colorbar label: "var leads T_<Region> by _ months  (+ve = var leads)"
# Data loading is identical to L_gridpoint_temp_decomp.jl.

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")

using JLD2, Dates, Statistics
@py import matplotlib.pyplot as plt, matplotlib.colors as colors, matplotlib.cm as cm, cartopy.crs as ccrs
const py_gc = pyimport("gc")

const visdir_base = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/max_lag_correlation_maps"
mkpath(visdir_base)

const LAG_RANGE = -12:12          # lags to search (months)

# ============================================================
# Helpers
# ============================================================

"""
    cor_vec_with_grid(t_vec, grid_3d)

Vectorised Pearson correlation of a 1-D vector `t_vec` (length n) with every
spatial column of `grid_3d` (lon × lat × n).  Returns a (lon × lat) matrix.
"""
function cor_vec_with_grid(t_vec::AbstractVector, grid_3d::AbstractArray{T,3}) where T<:Real
    n    = length(t_vec)
    # Reshape to (1, 1, n) so broadcasting against (lon, lat, n) is unambiguous
    tc   = reshape(t_vec .- mean(t_vec), 1, 1, :)
    t_σ  = std(t_vec)
    # Pre-materialise centred grid to avoid fused-broadcast shape inference issues
    G_c  = grid_3d .- mean(grid_3d; dims=3)          # (lon, lat, n)
    G_σ  = dropdims(std(grid_3d; dims=3); dims=3)    # (lon, lat)
    cov_map = dropdims(sum(tc .* G_c; dims=3); dims=3) ./ (n - 1)
    return cov_map ./ (t_σ .* G_σ)
end

"""
    max_lag_map(T_region, grid_3d, lags) → best_lag, best_cor

For each spatial point in `grid_3d` (lon × lat × time) return the lag in `lags`
at which  cor( grid_3d(x,·), time_lag(T_region, lag) )  explains the most
variance (max R²), together with the correlation at that lag.

Convention: `time_lag(T_region, L)[t] = T_region[t+L]` for L > 0,
so a positive returned lag means the gridpoint variable leads T_region.
"""
function max_lag_map(T_region::AbstractVector, grid_3d::AbstractArray, lags)
    sz       = size(grid_3d)[1:2]
    best_lag = zeros(Float32, sz)
    best_cor = zeros(Float32, sz)
    best_r2  = fill(-Inf32, sz)

    for lag in lags
        T_lag  = time_lag(T_region, lag)
        valid  = .!ismissing.(T_lag)
        T_v    = Float64.(collect(T_lag[valid]))
        G_v    = Float64.(grid_3d[:, :, valid])
        c_map  = cor_vec_with_grid(T_v, G_v)
        r2_map = c_map .^ 2
        mask   = isfinite.(c_map) .& (r2_map .> best_r2)
        best_r2[mask]  .= Float32.(r2_map[mask])
        best_cor[mask] .= Float32.(c_map[mask])
        best_lag[mask] .= Float32(lag)
    end
    return best_lag, best_cor
end

"""
    r2_alpha(best_cor) → Float32 matrix

Tanh crossfade from transparent to opaque over R² ∈ [0.10, 0.20].
  R² ≤ 0.10 → α ≈ 0.01  (clear)
  R² = 0.15 → α = 0.50  (midpoint)
  R² ≥ 0.20 → α ≈ 0.99  (opaque)
"""
function r2_alpha(best_cor::AbstractMatrix{Float32})
    r2 = best_cor .^ 2
    clamp.(0.5f0 .* (1.0f0 .+ tanh.(52.0f0 .* (r2 .- 0.10f0))), 0.0f0, 1.0f0)
end

function _flat_axes(axs, nrows, ncols)
    (nrows == 1 && ncols == 1) ? [axs] :
    (nrows == 1 || ncols == 1) ? collect(axs) :
    reduce(vcat, [pyconvert(Array, r) for r in axs])
end

# ============================================================
# Data Loading  (identical approach to L_gridpoint_temp_decomp.jl)
# ============================================================

println("Loading ERA5 (t2m + pressure-level T from new_pressure_levels.nc)…")
date_range = (Date(2000, 3), Date(2024, 2, 28))
era5_data, era5_coords = load_era5_data(["t2m", "t"], date_range;
    pressure_level_file = "new_pressure_levels.nc")
# era5_data["t2m"] : (lon, lat, time)
# era5_data["t"]   : (lon, lat, plev, time)   plev = [1000,850,700,500,250] hPa

println("Loading CERES…")
ceres_varnames = ["toa_net_all_mon", "toa_sw_all_mon",   "toa_lw_all_mon",
                  "toa_sw_clr_c_mon", "toa_lw_clr_c_mon",
                  "toa_cre_sw_mon",   "toa_cre_lw_mon"]
ceres_labels   = ["Net", "SW", "LW", "SW_clr", "LW_clr", "SW_cld (CRE)", "LW_cld (CRE)"]
ceres_data, ceres_coords = load_new_ceres_data(ceres_varnames, date_range)
ceres_coords["time"] = round_dates_down_to_nearest_month(ceres_coords["time"])
common_time = ceres_coords["time"]
@assert all(common_time .== round_dates_down_to_nearest_month(era5_coords["time"])) "ERA5/CERES time mismatch"

# Verify pressure-level time array aligns with common_time so data_valid_idx applies directly
plev_times_rounded = round_dates_down_to_nearest_month(era5_coords["pressure_time"])
@assert plev_times_rounded == common_time "ERA5 pressure-level time mismatch"

println("Deseasonalising & detrending all variables…")
float_time   = calc_float_time.(common_time)
month_groups = groupfind(month, common_time)

for sl in eachslice(era5_data["t2m"]; dims=(1, 2))
    deseasonalize_and_detrend_precalculated_groups_twice!(sl, float_time, month_groups;
        aggfunc=mean, trendfunc=least_squares_fit)
end
for i in axes(era5_data["t"], 1), j in axes(era5_data["t"], 2), p in axes(era5_data["t"], 3)
    sl = @view era5_data["t"][i, j, p, :]
    deseasonalize_and_detrend_precalculated_groups_twice!(sl, float_time, month_groups;
        aggfunc=mean, trendfunc=least_squares_fit)
end
for cvar in ceres_varnames, sl in eachslice(ceres_data[cvar]; dims=(1, 2))
    deseasonalize_and_detrend_precalculated_groups_twice!(sl, float_time, month_groups;
        aggfunc=mean, trendfunc=least_squares_fit)
end

println("Loading ENSO data…")
enso_data, enso_dates_raw = load_enso_data(date_range)
enso_times = round_dates_down_to_nearest_month(enso_dates_raw["time"])
oni_lags   = -24:24
n_enso     = length(enso_times)
enso_lag_full = Matrix{Union{Float64,Missing}}(missing, n_enso, length(oni_lags))
for (j, lag) in enumerate(oni_lags)
    col = "oni_lag_$lag"
    haskey(enso_data, col) && (enso_lag_full[:, j] .= enso_data[col])
end
valid_rows = [all(!ismissing, enso_lag_full[i, :]) for i in 1:n_enso]
enso_matrix      = Float64.(enso_lag_full[valid_rows, :])
enso_mat_matched, data_valid_idx = precompute_enso_match(
    common_time, enso_times[valid_rows], enso_matrix)

# ONI at lag 0 → ENSO "T_region"
lag0_col      = findfirst(==(0), collect(oni_lags))
T_enso_region = Float64.(enso_mat_matched[:, lag0_col])

println("Loading region masks…")
jldpath = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison"
era5_region_mask_dict = jldopen(joinpath(jldpath, "stratocumulus_region_masks.jld2"))["regional_masks_era5"]

era5_lat  = era5_coords["latitude"]
era5_lon  = era5_coords["longitude"]
ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]
plev_vals  = Int.(era5_coords["pressure_level"])    # [1000, 850, 700, 500, 250]
plev_labels = ["T@$(p)hPa" for p in plev_vals]

# ============================================================
# Colormap  (discrete diverging, integer ticks)
# ============================================================
lag_vals   = collect(LAG_RANGE)
boundaries = collect((lag_vals[1] - 0.5) : 1.0 : (lag_vals[end] + 0.5))
cmap_lag   = plt.get_cmap("viridis", length(lag_vals))
lag_norm   = colors.BoundaryNorm(boundaries, cmap_lag.N)

# ============================================================
# Panel layout: 3 rows × 4 cols = 12 panels
#   Panels 1–5  : ERA5 T at plev_vals
#   Panels 6–12 : CERES vars
# ============================================================
nrows, ncols = 3, 4
n_panels = length(plev_vals) + length(ceres_varnames)   # 5 + 7 = 12
@assert n_panels == nrows * ncols

panel_titles = [plev_labels; ceres_labels]

const regions_to_inspect = ["SEPac_feedback_definition", "NEPac", "SEAtl", "ENSO"]

# ============================================================
# Per-region loop
# ============================================================
for region_name in regions_to_inspect
    println("\n=== Region: $region_name ===")
    region_visdir = joinpath(visdir_base, region_name)
    mkpath(region_visdir)

    # ── T_region scalar (ENSO-valid period) ─────────────────────────────
    if region_name == "ENSO"
        T_region = T_enso_region
    else
        temp_mask      = era5_region_mask_dict[region_name]
        T_region_full  = generate_spatial_mean(era5_data["t2m"], era5_lat, temp_mask)
        T_region       = T_region_full[data_valid_idx]
    end

    central_lon = region_name == "SEAtl" ? 0 : 180
    proj = ccrs.Robinson(central_longitude=central_lon)

    fig, axs = plt.subplots(nrows, ncols;
        figsize = (ncols * 5, nrows * 3),
        subplot_kw = Dict("projection" => proj),
        layout = "compressed")
    axes_flat = _flat_axes(axs, nrows, ncols)

    # Closure: draw region mask contour on any axis
    function draw_mask!(ax)
        region_name == "ENSO" && return
        mask = era5_region_mask_dict[region_name]
        ax.contour(convert(Array{Float64}, era5_lon),
                   convert(Array{Float64}, era5_lat),
                   convert(Array{Float64}, collect(Float64.(mask)'));
            transform   = ccrs.PlateCarree(),
            levels      = pylist([0.5]),
            colors      = pylist(["black"]),
            linewidths  = pylist([1.0]),
            linestyles  = pylist(["--"]))
    end

    panel_idx = 1

    # ── ERA5 pressure-level panels ──────────────────────────────────────
    for p_idx in eachindex(plev_vals)
        println("  Lag map: $(plev_labels[p_idx])…")
        G_valid = Float64.(era5_data["t"][:, :, p_idx, data_valid_idx])
        lag_map, best_cor = max_lag_map(T_region, G_valid, LAG_RANGE)
        alpha = r2_alpha(best_cor)

        ax = axes_flat[panel_idx]
        plot_global_heatmap_on_ax!(ax, era5_lat, era5_lon, lag_map;
            cmap=cmap_lag, colornorm=lag_norm, title=plev_labels[p_idx],
            alpha_matrix=alpha)
        draw_mask!(ax)
        panel_idx += 1
        GC.gc(); py_gc.collect()
    end

    # ── CERES panels ─────────────────────────────────────────────────────
    for (cvar, clbl) in zip(ceres_varnames, ceres_labels)
        println("  Lag map: $clbl…")
        G_valid = Float64.(ceres_data[cvar][:, :, data_valid_idx])
        lag_map, best_cor = max_lag_map(T_region, G_valid, LAG_RANGE)
        alpha = r2_alpha(best_cor)

        ax = axes_flat[panel_idx]
        plot_global_heatmap_on_ax!(ax, ceres_lat, ceres_lon, lag_map;
            cmap=cmap_lag, colornorm=lag_norm, title=clbl,
            alpha_matrix=alpha)
        draw_mask!(ax)
        panel_idx += 1
        GC.gc(); py_gc.collect()
    end

    # Hide any unused panels (none expected)
    for i in panel_idx:nrows*ncols
        axes_flat[i].set_visible(false)
    end

    # ── Colorbar & title ─────────────────────────────────────────────────
    region_label = region_name == "ENSO" ? "ONI" : "T_$(region_name)"
    cb_label = "var leads $region_label by _ months  (+ve = var leads, −ve = var lags)"

    plt.colorbar(cm.ScalarMappable(norm=lag_norm, cmap=cmap_lag),
        ax=axs, orientation="horizontal",
        label=cb_label, shrink=0.55, pad=0.06,
        ticks=lag_vals[1:2:end])      # every 2nd tick for readability

    fig.suptitle("[$region_name] Lag of max correlation with $region_label",
        fontsize=13, y=1.01)

    outpath = joinpath(region_visdir, "max_lag_correlation.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close("all")          # clears matplotlib's global figure manager, not just this fig
    PythonCall.pydel!(fig)
    PythonCall.pydel!(axs)
    foreach(PythonCall.pydel!, axes_flat)
    PythonCall.pydel!(proj)
    # Clear cartopy's internal LRU transform caches
    let cartopy_crs = pyimport("cartopy.crs")
        for fn in py_gc.get_referents(cartopy_crs)
            pyconvert(Bool, pyhasattr(fn, "cache_clear")) && fn.cache_clear()
        end
    end
    GC.gc(); py_gc.collect()
    GC.gc(); py_gc.collect()
    println("  Saved: $outpath")
end

println("\nAll plots saved under: $visdir_base")
