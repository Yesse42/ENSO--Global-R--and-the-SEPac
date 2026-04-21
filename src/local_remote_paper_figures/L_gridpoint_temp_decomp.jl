# L_gridpoint_temp_decomp.jl
#
# Per-gridpoint decomposition of λ_eff_SEPac (eq. sepac_decomp):
#
#   λ_eff = f_L / Var(T_global) * Σ_i w_i Cov(T_L, R_i)
#         = f_L / Var(T_global) * Σ_i w_i [ α_i Cov(T_L, T_i) + Cov(T_L, R_{i,res}) ]
#
# where:
#   T_L         = SEPac regional mean t2m (scalar time series)
#   f_L         = fractional area of the SEPac region
#   w_i         = cos-lat area weight of CERES gridpoint i  (Σ w_i = 1)
#   α_i         = OLS slope of R_i ~ T_i  [= Cov(T_i,R_i)/Var(T_i)]
#   R_{i,res}   = R_i − α_i T_i
#   T_i         = ERA5 temperature at gridpoint i (mapped to CERES grid)
#
# ENSO decomposition: T_L = T_{L,E} + T_{L,NE}  via 1-component PLS.
# The ENSO component includes all cross-covariance terms involving T_{L,E}
# (i.e. Cov(T_{L,E}, R_i) — no decomposition of R is required):
#
#   total_E_i  = f_L w_i Cov(T_{L,E},  R_i) / Var(T_global)
#   total_NE_i = f_L w_i Cov(T_{L,NE}, R_i) / Var(T_global)
#
# and likewise for temp and resid (by linearity: resid = total − temp).
#
# Outputs per region × temperature variable:
#   - 9 individual PNG maps  (total, temp, resid) × (full, ENSO, NE)
#     with scalar λ = global sum shown in each title
#   - decomp_3pane.png   — [total | temp | resid]
#   - enso_6pane.png     — [ENSO row | NE row] × [total | temp | resid]

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")

using JLD2, Dates, Statistics
@py import matplotlib.pyplot as plt, matplotlib.colors as colors, matplotlib.cm as cm, cartopy.crs as ccrs

const visdir_base = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/L_gridpoint_temp_decomp"
mkpath(visdir_base)

# ============================================================
# Helper Functions
# ============================================================

"""
Remap 3-D ERA5 array (lon×lat×time) to CERES grid via nearest-neighbour index map.
`idx_map` has CERES grid shape; each element is a Tuple (i_era5, j_era5).
"""
function map_era5_to_ceres(era5_3d, idx_map)
    n_time = size(era5_3d, 3)
    [era5_3d[i, j, t] for (i, j) in idx_map, t in 1:n_time]
end

"""Per-gridpoint α_i = Cov(T_i, R_i)/Var(T_i): OLS slope of R ~ T at each CERES point.
Both arrays are (nlon, nlat, ntime)."""
function compute_alpha_field(T_3d, R_3d)
    [cov(T_3d[i,j,:], R_3d[i,j,:]) / var(T_3d[i,j,:])
     for i in axes(T_3d,1), j in axes(T_3d,2)]
end

"""1-component PLS decomposition of scalar y. Returns (enso_component, residual)."""
function enso_decompose_scalar(enso_matrix, y)
    m = make_pls_regressor(enso_matrix, y, 1; print_updates=false)
    e = vec(predict(m, enso_matrix))
    return e, y .- e
end

"""
    decompose_gridpoint_L(T_L, T_L_E, T_L_NE, T_era5_c, R, f_L, var_T)

Compute per-gridpoint contribution maps (nlon × nlat) to λ_eff_SEPac.
w_i = 1 at every gridpoint; area weighting is handled by the map projection.

Returns a NamedTuple:
  total, temp, resid         — full T_L
  total_E, temp_E, resid_E   — ENSO component of T_L (cross-covs with non-ENSO R included)
  total_NE, temp_NE, resid_NE — non-ENSO component of T_L

Global sum of each map = corresponding scalar λ component.
"""
function decompose_gridpoint_L(T_L, T_L_E, T_L_NE, T_era5_c, R, f_L, var_T)
    alpha = compute_alpha_field(T_era5_c, R)   # (nlon, nlat)
    scale = f_L / var_T                        # scalar

    # Inner helpers: Cov(ts, R_i) and Cov(ts, T_i) at every gridpoint
    cov_with_R(ts) = [cov(ts, R[i,j,:])        for i in axes(R,1),        j in axes(R,2)]
    cov_with_T(ts) = [cov(ts, T_era5_c[i,j,:]) for i in axes(T_era5_c,1), j in axes(T_era5_c,2)]

    total    = scale .* cov_with_R(T_L)
    total_E  = scale .* cov_with_R(T_L_E)
    total_NE = scale .* cov_with_R(T_L_NE)

    temp    = scale .* alpha .* cov_with_T(T_L)
    temp_E  = scale .* alpha .* cov_with_T(T_L_E)
    temp_NE = scale .* alpha .* cov_with_T(T_L_NE)

    # Residual = total − temp  (exact by linearity; avoids allocating R_res)
    resid    = total    .- temp
    resid_E  = total_E  .- temp_E
    resid_NE = total_NE .- temp_NE

    return (; total, temp, resid, total_E, temp_E, resid_E, total_NE, temp_NE, resid_NE)
end

# ── Plot utilities ─────────────────────────────────────────────────────

function _shared_norm(fields...)
    absmax = maximum(abs, filter(!isnan, vcat(vec.(fields)...)))
    colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
end

λ_str(v) = "λ=$(round(v; sigdigits=3))"

"""Save a single-panel global map. λ value should be embedded in `title`."""
function save_single_map(lat, lon, field, fpath;
        title="", colorbar_label="W m⁻² K⁻¹", central_longitude=180, contour_fn=nothing)
    proj = ccrs.Robinson(central_longitude=central_longitude)
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=Dict("projection" => proj))
    absmax = maximum(abs, filter(!isnan, vec(Float64.(field))))
    norm   = colors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    plot_global_heatmap_on_ax!(ax, lat, lon, field;
        cmap=DAVES_CMAP, colornorm=norm, title=title)
    contour_fn !== nothing && try contour_fn(ax) catch end
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=DAVES_CMAP),
        ax=ax, orientation="horizontal", label=colorbar_label, shrink=0.8, pad=0.08)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
end

"""
Multi-pane figure with a shared colour scale.
`panels` is a Vector of (title_string, field_matrix).
`nrows` subdivides panels into rows (default 1 row).
"""
function plot_multipane(lat, lon, panels;
        suptitle="", colorbar_label="W m⁻² K⁻¹",
        central_longitude=180, contour_fn=nothing, nrows=1)
    ncols = length(panels) ÷ nrows
    proj  = ccrs.Robinson(central_longitude=central_longitude)
    fig, axs = plt.subplots(nrows, ncols;
        figsize=(7ncols, 5nrows),
        subplot_kw=Dict("projection" => proj), layout="compressed")

    flat_axs = nrows * ncols == 1 ? [axs] :
               (nrows == 1 || ncols == 1) ? collect(axs) :
               reduce(vcat, [pyconvert(Array, r) for r in axs])

    norm = _shared_norm([f for (_, f) in panels]...)

    for (ax, (title, field)) in zip(flat_axs, panels)
        plot_global_heatmap_on_ax!(ax, lat, lon, field;
            cmap=DAVES_CMAP, colornorm=norm, title=title)
        contour_fn !== nothing && try contour_fn(ax) catch end
    end

    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=DAVES_CMAP),
        ax=axs, orientation="horizontal", label=colorbar_label, shrink=0.6, pad=0.08)
    isempty(suptitle) || fig.suptitle(suptitle, fontsize=11)
    return fig
end

# ============================================================
# Data Loading
# ============================================================

println("Loading ERA5 t2m...")
date_range = (Date(2000, 3), Date(2024, 2, 28))
era5_data, era5_coords = load_era5_data(["t2m"], date_range)

println("Loading ERA5 t@700hPa and t@1000hPa...")
era5_data_pl, era5_coords_pl = load_era5_data(["t"], date_range;
    pressure_level_file="new_pressure_levels.nc")
pressure_levels = era5_coords_pl["pressure_level"]
idx_700  = findfirst(==(700),  Int.(pressure_levels))
idx_1000 = findfirst(==(1000), Int.(pressure_levels))
T_700_raw  = era5_data_pl["t"][:, :, idx_700,  :]
T_1000_raw = era5_data_pl["t"][:, :, idx_1000, :]

pot_temp(T, P) = T .* (1000 / P)^(2/7)
LTS_raw = pot_temp(T_700_raw, 700) .- pot_temp(T_1000_raw, 1000)

println("Loading CERES net radiation...")
ceres_data, ceres_coords = load_new_ceres_data(["toa_net_all_mon"], date_range)
ceres_coords["time"] = round_dates_down_to_nearest_month(ceres_coords["time"])
common_time = ceres_coords["time"]
@assert all(common_time .== round_dates_down_to_nearest_month(era5_coords["time"])) "ERA5/CERES time mismatch"
@assert all(common_time .== round_dates_down_to_nearest_month(era5_coords_pl["pressure_time"])) "ERA5 PL/CERES time mismatch"

println("Deseasonalizing and detrending...")
float_time   = calc_float_time.(common_time)
month_groups = groupfind(month, common_time)
for datadict in [era5_data, ceres_data]
    for var_data in datadict
        for sl in eachslice(var_data; dims=(1, 2))
            deseasonalize_and_detrend_precalculated_groups_twice!(sl, float_time, month_groups;
                aggfunc=mean, trendfunc=least_squares_fit)
        end
    end
end
for arr in [T_700_raw, LTS_raw]
    for sl in eachslice(arr; dims=(1, 2))
        deseasonalize_and_detrend_precalculated_groups_twice!(sl, float_time, month_groups;
            aggfunc=mean, trendfunc=least_squares_fit)
    end
end

println("Loading region masks...")
jldpath = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/"
const regions_to_inspect = ["SEPac_feedback_definition", "NEPac", "SEAtl"]
era5_region_mask_dict = jldopen(joinpath(jldpath, "stratocumulus_region_masks.jld2"))["regional_masks_era5"]
rad_region_mask_dict  = jldopen(joinpath(jldpath, "stratocumulus_exclusion_masks.jld2"))["stratocum_masks"]

println("Loading ENSO data...")
enso_data, enso_dates_raw = load_enso_data(date_range)
enso_times = round_dates_down_to_nearest_month(enso_dates_raw["time"])
lags = -6:6
n_enso = length(enso_times)
enso_lag_matrix_full = Matrix{Union{Float64,Missing}}(missing, n_enso, length(lags))
for (j, lag) in enumerate(lags)
    col = "oni_lag_$lag"
    haskey(enso_data, col) && (enso_lag_matrix_full[:, j] .= enso_data[col])
end
valid_rows = [all(!ismissing, enso_lag_matrix_full[i, :]) for i in 1:n_enso]
enso_matrix = Float64.(enso_lag_matrix_full[valid_rows, :])
enso_matrix_matched, data_valid_idx = precompute_enso_match(
    common_time, enso_times[valid_rows], enso_matrix)

println("Loading ERA5↔CERES coordinate mapping...")
coord_mapping = JLD2.load(joinpath(jldpath, "era5_ceres_coordinate_mapping.jld2"))
era5_to_ceres_indices = Tuple.(coord_mapping["era5_to_ceres_indices"])

# ============================================================
# Shared Quantities (ENSO-valid time steps throughout for consistency)
# ============================================================

ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]
era5_lat  = era5_coords["latitude"]
era5_lon  = era5_coords["longitude"]

# R: CERES net radiation restricted to ENSO-valid time steps
R = ceres_data["toa_net_all_mon"][:, :, data_valid_idx]

# var_T = Var(global mean t2m) — standard feedback denominator
global_mean_t2m = generate_spatial_mean(era5_data["t2m"], era5_lat)
var_T = var(global_mean_t2m[data_valid_idx])

# ============================================================
# Per-Temperature-Variable, Per-Region Analysis
# ============================================================

for (temp_var_name, T_era5_full) in [("t2m", era5_data["t2m"]), ("t700hPa", T_700_raw), ("LTS", LTS_raw)]
    temp_var_name == "t700hPa" && continue
    println("\n" * "="^60)
    println("Temperature variable: $temp_var_name")
    println("="^60)
    temp_visdir = joinpath(visdir_base, temp_var_name)
    mkpath(temp_visdir)

    println("  Mapping ERA5 $temp_var_name to CERES grid (ENSO-valid times)...")
    T_era5_c = map_era5_to_ceres(T_era5_full[:, :, data_valid_idx], era5_to_ceres_indices)

    for region_name in regions_to_inspect
        println("\n  === Region: $region_name ($temp_var_name) ===")
        region_visdir = joinpath(temp_visdir, region_name)
        mkpath(region_visdir)

        temp_mask   = era5_region_mask_dict[region_name]
        rad_mask    = rad_region_mask_dict[region_name]
        central_lon = region_name == "SEAtl" ? 0 : 180

        contour_fn = ax -> add_region_contours!(
            ax, ceres_lat, ceres_lon, rad_mask, era5_lat, era5_lon, temp_mask)

        # T_L: regional mean t2m (always t2m for consistency with λ_eff definition)
        T_L_full = generate_spatial_mean(era5_data["t2m"], era5_lat, temp_mask)
        T_L      = T_L_full[data_valid_idx]
        T_L_E, T_L_NE = enso_decompose_scalar(enso_matrix_matched, T_L)

        # f_L: fractional area of this region
        f_L = calculate_mask_fractional_area(temp_mask, era5_lat)
        println("  f_L = $(round(f_L; sigdigits=4))")

        println("  Computing gridpoint decomposition (α, Cov fields)...")
        D = decompose_gridpoint_L(T_L, T_L_E, T_L_NE, T_era5_c, R, f_L, var_T)

        # Scalar λ components = global sum of each map
        gm(m) = spatial_mean_kernel(m, ceres_lat)
        λ = map(gm, D)
        println("  λ:    total=$(round(λ.total; sigdigits=3))  " *
                "temp=$(round(λ.temp; sigdigits=3))  " *
                "resid=$(round(λ.resid; sigdigits=3))")
        println("  λ_E:  total=$(round(λ.total_E; sigdigits=3))  " *
                "temp=$(round(λ.temp_E; sigdigits=3))  " *
                "resid=$(round(λ.resid_E; sigdigits=3))")
        println("  λ_NE: total=$(round(λ.total_NE; sigdigits=3))  " *
                "temp=$(round(λ.temp_NE; sigdigits=3))  " *
                "resid=$(round(λ.resid_NE; sigdigits=3))")

        tag  = "[$region_name/$temp_var_name]"
        clbl = "W m⁻² K⁻¹"

        # ── Individual maps (9 total) ─────────────────────────────────────
        map_specs = [
            ("total",    D.total,    "Total  $(λ_str(λ.total))"),
            ("temp",     D.temp,     "Temp-mediated  $(λ_str(λ.temp))"),
            ("resid",    D.resid,    "Residual  $(λ_str(λ.resid))"),
            ("total_E",  D.total_E,  "ENSO total  $(λ_str(λ.total_E))"),
            ("temp_E",   D.temp_E,   "ENSO temp  $(λ_str(λ.temp_E))"),
            ("resid_E",  D.resid_E,  "ENSO resid  $(λ_str(λ.resid_E))"),
            ("total_NE", D.total_NE, "Non-ENSO total  $(λ_str(λ.total_NE))"),
            ("temp_NE",  D.temp_NE,  "Non-ENSO temp  $(λ_str(λ.temp_NE))"),
            ("resid_NE", D.resid_NE, "Non-ENSO resid  $(λ_str(λ.resid_NE))"),
        ]
        for (fname, field, ptitle) in map_specs
            save_single_map(ceres_lat, ceres_lon, field,
                joinpath(region_visdir, "$(fname).png");
                title = "$tag $ptitle",
                colorbar_label = clbl,
                central_longitude = central_lon,
                contour_fn = contour_fn)
        end
        println("  Saved 9 individual maps.")

        # ── 3-pane: total | temp | resid ──────────────────────────────────
        fig3 = plot_multipane(ceres_lat, ceres_lon,
            [("Total  $(λ_str(λ.total))",        D.total),
             ("Temp-mediated  $(λ_str(λ.temp))", D.temp),
             ("Residual  $(λ_str(λ.resid))",     D.resid)];
            suptitle = "$tag  λ_eff decomposition  |  f_L=$(round(f_L; sigdigits=4))",
            colorbar_label = clbl,
            central_longitude = central_lon,
            contour_fn = contour_fn)
        fig3.savefig(joinpath(region_visdir, "decomp_3pane.png"), dpi=300, bbox_inches="tight")
        plt.close(fig3)

        # ── 6-pane: [ENSO | NE] × [total | temp | resid] ─────────────────
        fig6 = plot_multipane(ceres_lat, ceres_lon,
            [("ENSO total  $(λ_str(λ.total_E))",    D.total_E),
             ("ENSO temp  $(λ_str(λ.temp_E))",      D.temp_E),
             ("ENSO resid  $(λ_str(λ.resid_E))",    D.resid_E),
             ("NE total  $(λ_str(λ.total_NE))",     D.total_NE),
             ("NE temp  $(λ_str(λ.temp_NE))",       D.temp_NE),
             ("NE resid  $(λ_str(λ.resid_NE))",     D.resid_NE)];
            suptitle = "$tag  ENSO decomposition of λ_eff  |  f_L=$(round(f_L; sigdigits=4))",
            colorbar_label = clbl,
            central_longitude = central_lon,
            contour_fn = contour_fn,
            nrows = 2)
        fig6.savefig(joinpath(region_visdir, "enso_6pane.png"), dpi=300, bbox_inches="tight")
        plt.close(fig6)
        println("  Saved decomp_3pane.png and enso_6pane.png.")
    end
end

println("\nAll plots saved under: $visdir_base")
