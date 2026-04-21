# enso_causes_remote_feedbacks.jl
#
# Investigate how ENSO drives remote radiative feedbacks in the three
# stratocumulus regions (SEPac, NEPac, SEAtl).
#
# Data loaded:
#   - ERA5 t2m (gridded, deseasonalised & detrended)
#   - CERES toa_net_all_mon (gridded, deseasonalised & detrended)
#   - Niño 3.4 index (ONI lag-0, restricted to ENSO-valid period)
#
# Derived regional quantities (ENSO-valid time steps):
#   - region_t2m[r]   : cos-lat weighted mean ERA5 t2m over each region
#   - region_rad[r]   : cos-lat weighted mean CERES net rad over each region
#   - rad_region_frac[r] : fractional area of each region's radiation mask (CERES grid)
#   - t2m_region_frac[r] : fractional area of each region's temperature mask (ERA5 grid)
#   - global_mean_rad : global cos-lat weighted mean CERES net rad
#   - nonlocal_rad    : global_mean_rad − region_rad[r] × rad_region_frac[r]

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")

using JLD2, Dates, Statistics
@py import matplotlib.pyplot as plt, matplotlib.colors as colors, matplotlib.cm as cm, cartopy.crs as ccrs

# ============================================================
# Max-lag correlation utilities  (1-D scalar versions)
# ============================================================

"""
    cor_at_lag(x, y, lag) → Float64

Pearson correlation between `x` and `time_lag(y, lag)`, using only the
overlapping (non-missing) time steps.  Positive lag means `x` leads `y`
(because `time_lag` negates the lag internally before shifting).
"""
function cor_at_lag(x::AbstractVector, y::AbstractVector, lag::Int)
    y_lag = time_lag(y, lag)
    valid = .!ismissing.(y_lag)
    cor(x[valid], Float64.(collect(y_lag[valid])))
end

"""
    max_lag_cor(x, y, lags) → (best_lag, best_cor)

Return the lag in `lags` at which the Pearson correlation between `x` and
`time_lag(y, lag)` explains the most variance (max R²), together with the
correlation at that lag.  Positive lag means `x` leads `y`.
"""
function max_lag_cor(x::AbstractVector, y::AbstractVector, lags)
    best_lag = first(lags)
    best_cor = 0.0
    best_r2  = -Inf
    for lag in lags
        c = cor_at_lag(x, y, lag)
        if isfinite(c) && c^2 > best_r2
            best_r2  = c^2
            best_cor = c
            best_lag = lag
        end
    end
    return best_lag, best_cor
end

# ============================================================
# Data Loading
# ============================================================

println("Loading ERA5 t2m + pressure-level temperature...")
date_range = (Date(2000, 3), Date(2024, 2, 28))
era5_data, era5_coords = load_era5_data(["t2m", "t"], date_range;
                                         pressure_level_file="new_pressure_levels.nc")

println("Loading CERES net radiation...")
ceres_data, ceres_coords = load_new_ceres_data(["toa_net_all_mon"], date_range)
ceres_coords["time"] = round_dates_down_to_nearest_month(ceres_coords["time"])
common_time = ceres_coords["time"]
@assert all(common_time .== round_dates_down_to_nearest_month(era5_coords["time"])) "ERA5/CERES time mismatch"

println("Loading region masks...")
jldpath            = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/"
regions_to_inspect = ["SEPac_feedback_definition", "NEPac", "SEAtl"]
visdir             = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/enso_causes_remote_feedbacks/"
mkpath(visdir)
era5_region_mask_dict = jldopen(joinpath(jldpath, "stratocumulus_region_masks.jld2"))["regional_masks_era5"]
rad_region_mask_dict  = jldopen(joinpath(jldpath, "stratocumulus_exclusion_masks.jld2"))["stratocum_masks"]

# ============================================================
# Coordinate arrays & shared constants
# ============================================================

era5_lat  = era5_coords["latitude"]
era5_lon  = era5_coords["longitude"]
ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]

enso_lags    = -12:12
lags_vec     = collect(enso_lags)
n_regions    = length(regions_to_inspect)
n_panels     = n_regions + 1
_np          = pyimport("numpy")

row_labels_full = ["Local R", "Nonlocal R", "Total"]

# Region area fractions depend only on masks — compute once
# Radiation mask fraction: used to weight region_rad in nonlocal_rad construction
rad_region_frac = Dict{String, Float64}(
    r => calculate_mask_fractional_area(rad_region_mask_dict[r], ceres_lat)
    for r in regions_to_inspect)

# Temperature mask fraction: used to scale the covariance table entries
t2m_region_frac = Dict{String, Float64}(
    r => calculate_mask_fractional_area(era5_region_mask_dict[r], era5_lat)
    for r in regions_to_inspect)

# ============================================================
# Gridded LTS field  (θ_700 − θ_1000, aligned to single-level time)
# ============================================================

println("Computing gridded LTS...")
pot_temp(T, P) = T .* (1000f0 ./ P) .^ (2f0/7f0)

press_levels          = era5_coords["pressure_level"]
idx_700               = findfirst(==(700), press_levels)
idx_1000              = findfirst(==(1000), press_levels)
sfc_times             = round_dates_down_to_nearest_month.(era5_coords["time"])
p_times               = round_dates_down_to_nearest_month.(era5_coords["pressure_time"])
valid_p_times         = findall(t -> t in sfc_times, p_times)
θ_700_gridded         = pot_temp.(era5_data["t"][:, :, idx_700,  valid_p_times], 700f0)
θ_1000_gridded        = pot_temp.(era5_data["t"][:, :, idx_1000, valid_p_times], 1000f0)
lts_gridded           = θ_700_gridded .- θ_1000_gridded   # (lon, lat, n_single_level_times)

# ============================================================
# Local-variable configurations for the outer loop
# ============================================================

local_var_configs = [
    (name    = "T2m",
     gridded = era5_data["t2m"]),
    (name    = "LTS",
     gridded = lts_gridded),
]

# ============================================================
# Lag-correlation plot helper  (defined once, used in both passes)
# ============================================================

function plot_lag_cor_panel!(ax, cors, best_lag, best_cor, title; ylabel=false, xlabel_dir="← ENSO leads  |  region leads →")
    lag_label = best_lag == 0 ? "no lag" :
                best_lag  > 0 ? "region leads ENSO by $(best_lag) mo" :
                                "ENSO leads region by $(-best_lag) mo"
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(best_lag, color="red", linewidth=1.2, linestyle="--", alpha=0.7, label=lag_label)
    ax.plot(_np.asarray(convert(Array{Float64}, lags_vec)),
            _np.asarray(convert(Array{Float64}, cors)),
            color="steelblue", linewidth=1.8, marker="o", markersize=3)
    ax.scatter(pylist([best_lag]), pylist([best_cor]), color="red", zorder=5, s=60)
    ax.set_xlabel("Lag (months,  $(xlabel_dir))", fontsize=10)
    ylabel && ax.set_ylabel("Pearson r", fontsize=10)
    ax.set_title("$(title)\nr=$(round(best_cor;digits=3))  R²=$(round(best_cor^2;digits=3))  ($(lag_label))",
                 fontsize=10)
    ax.set_xlim(first(lags_vec) - 0.5, last(lags_vec) + 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9)
end

# ============================================================
# Main loop: T2m / LTS  ×  RONI / ONI
# ============================================================

enso_index_configs = [
    (name       = "RONI",
     load_fn    = () -> load_roni(date_range; lags=enso_lags),
     col_prefix = "roni_lag_"),
    (name       = "ONI",
     load_fn    = () -> load_enso_data(date_range; lags=enso_lags),
     col_prefix = "oni_lag_"),
]

for lvcfg in local_var_configs
for cfg in enso_index_configs
    println("\n" * "="^60)
    println("  Local var: $(lvcfg.name)  |  ENSO index: $(cfg.name)")
    println("="^60)

    # ----------------------------------------------------------
    # Load index, build lag matrix, match to common_time
    # ----------------------------------------------------------
    println("Loading $(cfg.name)...")
    idx_data, idx_dates_raw = cfg.load_fn()
    idx_times  = round_dates_down_to_nearest_month(idx_dates_raw["time"])
    n_idx      = length(idx_times)
    lag_full   = Matrix{Union{Float64,Missing}}(missing, n_idx, length(enso_lags))
    for (j, lag) in enumerate(enso_lags)
        col = "$(cfg.col_prefix)$(lag)"
        haskey(idx_data, col) && (lag_full[:, j] .= idx_data[col])
    end
    valid_rows     = [all(!ismissing, lag_full[i, :]) for i in 1:n_idx]
    idx_matrix     = Float64.(lag_full[valid_rows, :])
    idx_mat_matched, data_valid_idx = precompute_enso_match(
        common_time, idx_times[valid_rows], idx_matrix)

    lag0_col   = findfirst(==(0), lags_vec)
    enso_index = Float64.(idx_mat_matched[:, lag0_col])

    # ----------------------------------------------------------
    # Spatial means (ENSO-valid time steps)
    # ----------------------------------------------------------
    local_var_gridded = lvcfg.gridded[:, :, data_valid_idx]
    R                 = ceres_data["toa_net_all_mon"][:, :, data_valid_idx]

    global_mean_local = generate_spatial_mean(local_var_gridded, era5_lat)
    global_mean_rad   = generate_spatial_mean(R, ceres_lat)

    region_local = Dict{String, Vector{Float64}}()
    region_rad   = Dict{String, Vector{Float64}}()
    for region_name in regions_to_inspect
        region_local[region_name] = generate_spatial_mean(local_var_gridded, era5_lat, era5_region_mask_dict[region_name])
        region_rad[region_name]   = generate_spatial_mean(R, ceres_lat, rad_region_mask_dict[region_name])
    end

    # ----------------------------------------------------------
    # Deseasonalise and detrend all 1-D spatial-mean time series
    # ----------------------------------------------------------
    valid_time         = common_time[data_valid_idx]
    valid_float_time   = calc_float_time.(valid_time)
    valid_month_groups = groupfind(month, valid_time)

    println("Deseasonalising and detrending spatial-mean time series...")
    for ts in (global_mean_local, global_mean_rad)
        deseasonalize_and_detrend_precalculated_groups_twice!(ts, valid_float_time, valid_month_groups;
            aggfunc=mean, trendfunc=least_squares_fit)
    end
    for region_name in regions_to_inspect
        for ts in (region_local[region_name], region_rad[region_name])
            deseasonalize_and_detrend_precalculated_groups_twice!(ts, valid_float_time, valid_month_groups;
                aggfunc=mean, trendfunc=least_squares_fit)
        end
    end

    # Nonlocal radiation from already-deseasonalised components
    nonlocal_rad = Dict{String, Vector{Float64}}(
        r => global_mean_rad .- region_rad[r] .* rad_region_frac[r]
        for r in regions_to_inspect)

    println("  Time steps (ENSO-valid): $(length(enso_index))")
    println("  Global mean rad std:     $(round(std(global_mean_rad); sigdigits=4)) W m⁻²")
    println("  Global mean $(lvcfg.name) std:   $(round(std(global_mean_local); sigdigits=4))")
    for region_name in regions_to_inspect
        f_rad = round(rad_region_frac[region_name]; sigdigits=4)
        f_t2m = round(t2m_region_frac[region_name]; sigdigits=4)
        lv_std = round(std(region_local[region_name]); sigdigits=4)
        println("  $region_name: f_rad=$(f_rad)  f_t2m=$(f_t2m)  " *
                "$(lvcfg.name)_std=$(lv_std)  " *
                "rad_std=$(round(std(region_rad[region_name]); sigdigits=4)) W m⁻²  " *
                "nonlocal_std=$(round(std(nonlocal_rad[region_name]); sigdigits=4)) W m⁻²")
    end

    # ----------------------------------------------------------
    # Lag-correlation of basin local var with ENSO index
    # ----------------------------------------------------------
    basin_best_lag = Dict{String, Int}()
    basin_best_cor = Dict{String, Float64}()
    basin_lag_cors = Dict{String, Vector{Float64}}()

    println("\n$(cfg.name)–$(lvcfg.name) lag correlations:")
    for region_name in regions_to_inspect
        loc                         = region_local[region_name]
        cors                        = [cor_at_lag(loc, enso_index, lag) for lag in enso_lags]
        best_lag, best_cor          = max_lag_cor(loc, enso_index, enso_lags)
        basin_best_lag[region_name] = best_lag
        basin_best_cor[region_name] = best_cor
        basin_lag_cors[region_name] = cors
        println("  $(rpad(region_name, 30)) best lag=$(lpad(best_lag,3)) months  " *
                "r=$(round(best_cor; digits=3))  R²=$(round(best_cor^2; digits=3))")
    end

    # ----------------------------------------------------------
    # ENSO / residual decomposition of basin local var
    # ----------------------------------------------------------
    basin_local_enso  = Dict{String, Vector{Float64}}()
    basin_local_resid = Dict{String, Vector{Float64}}()

    for region_name in regions_to_inspect
        loc      = region_local[region_name]
        best_lag = basin_best_lag[region_name]
        idx_lag  = time_lag(enso_index, best_lag)
        valid    = .!ismissing.(idx_lag)
        x_valid  = Float64.(collect(idx_lag[valid]))
        β        = least_squares_fit(x_valid, loc[valid]).slope
        enso_comp         = zeros(Float64, length(loc))
        enso_comp[valid] .= β .* x_valid
        basin_local_enso[region_name]  = enso_comp
        basin_local_resid[region_name] = loc .- enso_comp
    end

    # ----------------------------------------------------------
    # Covariance decomposition tables
    # ----------------------------------------------------------
    var_global_local  = var(global_mean_local)
    lv_col_labels     = ["$(lvcfg.name)_ENSO", "$(lvcfg.name)_resid"]

    # Regional tables: 2×2 inner (local/nonlocal R  ×  ENSO/resid T)
    cov_tables = Dict{String, Matrix{Float64}}()
    for region_name in regions_to_inspect
        local_R  = region_rad[region_name] .* rad_region_frac[region_name]
        nonloc_R = nonlocal_rad[region_name]
        L_enso   = basin_local_enso[region_name]
        L_resid  = basin_local_resid[region_name]
        scale    = t2m_region_frac[region_name] / var_global_local
        cov_tables[region_name] = [cov(local_R,  L_enso)  cov(local_R,  L_resid);
                                   cov(nonloc_R, L_enso)  cov(nonloc_R, L_resid)] .* scale
    end

    # Global table: 1×2 inner (global R  ×  ENSO/resid global T)
    # Scale = 1 / Var(global_local); no area fraction for global quantities.
    global_best_lag, _  = max_lag_cor(global_mean_local, enso_index, enso_lags)
    global_idx_lag       = time_lag(enso_index, global_best_lag)
    global_valid         = .!ismissing.(global_idx_lag)
    global_x_valid       = Float64.(collect(global_idx_lag[global_valid]))
    global_β             = least_squares_fit(global_x_valid, global_mean_local[global_valid]).slope
    global_local_enso    = zeros(Float64, length(global_mean_local))
    global_local_enso[global_valid] .= global_β .* global_x_valid
    global_local_resid   = global_mean_local .- global_local_enso
    global_scale         = 1.0 / var_global_local
    cov_global_table     = reshape(
        [cov(global_mean_rad, global_local_enso),
         cov(global_mean_rad, global_local_resid)] .* global_scale, 1, 2)

    # ----------------------------------------------------------
    # Helper: build extended table (inner + marginal totals)
    # ----------------------------------------------------------
    function build_ext_table(inner::Matrix{Float64}; add_row_total::Bool=true)
        nr, nc  = size(inner)
        n_rows  = add_row_total ? nr + 1 : nr
        ext     = zeros(Float64, n_rows, nc + 1)
        ext[1:nr, 1:nc]  = inner
        ext[1:nr, nc+1] .= vec(sum(inner; dims=2))
        if add_row_total
            ext[nr+1, 1:nc] .= vec(sum(inner; dims=1))
            ext[nr+1, nc+1]  = sum(inner)
        end
        return ext
    end

    # ----------------------------------------------------------
    # Helper: draw one covariance table panel
    # ----------------------------------------------------------
    function add_cov_table!(ax, inner::Matrix{Float64}, norm, absmax,
                            row_labels_in, col_labels_in, title; add_row_total::Bool=true)
        nr, nc = size(inner)
        ext    = build_ext_table(inner; add_row_total)
        n_rows = size(ext, 1)

        ax.imshow(_np.asarray(convert(Array{Float64}, ext)),
                  cmap=DAVES_CMAP, norm=norm, aspect="auto")

        for i in 0:n_rows-1, j in 0:nc
            val         = ext[i+1, j+1]
            is_marginal = (add_row_total && i == nr) || j == nc
            text_color  = abs(val) > 0.55 * absmax ? "white" : "black"
            prefix      = is_marginal ? "Σ=" : ""
            ax.text(j, i, "$(prefix)$(round(val; digits=4))",
                    ha="center", va="center", fontsize=10,
                    color=text_color, fontweight=(is_marginal ? "bold" : "normal"))
        end

        add_row_total && ax.axhline(nr - 0.5, color="white", linewidth=2.0)
        ax.axvline(nc - 0.5, color="white", linewidth=2.0)

        ax.set_xticks(pylist(collect(0:nc)))
        ax.set_xticklabels(pylist(vcat(col_labels_in, ["Total"])), fontsize=10)
        ax.set_yticks(pylist(collect(0:n_rows-1)))
        ax.set_yticklabels(pylist(add_row_total ? vcat(row_labels_in, ["Total"]) : row_labels_in),
                           fontsize=10)
        ax.set_title(title, fontsize=10)
    end

    # ----------------------------------------------------------
    # Plot 1: covariance decomposition tables (3 regional + 1 global)
    # ----------------------------------------------------------
    absmax_regional = maximum(abs.(vcat(
        [vec(build_ext_table(cov_tables[r]; add_row_total=true)) for r in regions_to_inspect]...)))
    norm_regional   = colors.TwoSlopeNorm(vmin=-absmax_regional, vcenter=0.0, vmax=absmax_regional)

    absmax_global_t = maximum(abs.(vec(build_ext_table(cov_global_table; add_row_total=false))))
    norm_global_t   = colors.TwoSlopeNorm(vmin=-absmax_global_t, vcenter=0.0, vmax=absmax_global_t)

    n_table_panels = n_regions + 1
    fig, axs = plt.subplots(1, n_table_panels;
                            figsize=pylist([3.8 * n_table_panels, 3.4]),
                            layout="constrained")

    for (idx, region_name) in enumerate(regions_to_inspect)
        best_lag = basin_best_lag[region_name]
        lag_str  = best_lag == 0 ? "no lag" :
                   best_lag  > 0 ? "region leads $(cfg.name) by $(best_lag) mo" :
                                   "$(cfg.name) leads region by $(-best_lag) mo"
        add_cov_table!(axs[idx - 1], cov_tables[region_name], norm_regional, absmax_regional,
                       ["Local R", "Nonlocal R"], lv_col_labels,
                       "$(region_name)  ($(lag_str))"; add_row_total=true)
    end

    global_lag_str = global_best_lag == 0 ? "no lag" :
                     global_best_lag  > 0 ? "global leads $(cfg.name) by $(global_best_lag) mo" :
                                            "$(cfg.name) leads global by $(-global_best_lag) mo"
    add_cov_table!(axs[n_table_panels - 1], cov_global_table, norm_global_t, absmax_global_t,
                   ["Global R"], lv_col_labels,
                   "Global  ($(global_lag_str))"; add_row_total=false)

    regional_axs = pylist([axs[i] for i in 0:n_regions-1])
    plt.colorbar(cm.ScalarMappable(norm=norm_regional, cmap=DAVES_CMAP),
                 ax=regional_axs, orientation="horizontal",
                 label="Regional cov contribution  [W m⁻² K⁻¹]", shrink=0.7, pad=0.14)
    plt.colorbar(cm.ScalarMappable(norm=norm_global_t, cmap=DAVES_CMAP),
                 ax=axs[n_table_panels - 1], orientation="horizontal",
                 label="Global cov contribution  [W m⁻² K⁻¹]", shrink=0.9, pad=0.14)
    fig.suptitle("$(lvcfg.name) / $(cfg.name)  —  Cov(R_global, $(lvcfg.name)) / Var($(lvcfg.name)_global)  [W m⁻² K⁻¹]",
                 fontsize=12)

    let savepath = joinpath(visdir, "$(lvcfg.name)_$(cfg.name)_cov_decomp_table.png")
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        println("\nSaved: $savepath")
    end
    plt.close(fig)

    # ----------------------------------------------------------
    # Plot 2: lag-correlation profiles
    # ----------------------------------------------------------
    global_rad_cors = [cor_at_lag(global_mean_rad, enso_index, lag) for lag in enso_lags]
    global_rad_best_lag, global_rad_best_cor = max_lag_cor(global_mean_rad, enso_index, enso_lags)

    fig2, axs2 = plt.subplots(1, n_panels; figsize=pylist([4.0 * n_panels, 3.2]),
                               sharey=true, layout="constrained")

    for (idx, region_name) in enumerate(regions_to_inspect)
        plot_lag_cor_panel!(axs2[idx - 1], basin_lag_cors[region_name],
                            basin_best_lag[region_name], basin_best_cor[region_name],
                            "$(region_name) $(lvcfg.name)"; ylabel=(idx == 1),
                            xlabel_dir="← $(cfg.name) leads  |  region leads →")
    end
    plot_lag_cor_panel!(axs2[n_panels - 1], global_rad_cors,
                        global_rad_best_lag, global_rad_best_cor,
                        "Global mean R"; xlabel_dir="← $(cfg.name) leads  |  R leads →")

    fig2.suptitle("$(lvcfg.name) / $(cfg.name) lag correlations", fontsize=12)

    let savepath = joinpath(visdir, "$(lvcfg.name)_$(cfg.name)_lag_cors.png")
        fig2.savefig(savepath, dpi=150, bbox_inches="tight")
        println("Saved: $savepath")
    end
    plt.close(fig2)

end  # for cfg in enso_index_configs
end  # for lvcfg in local_var_configs

# ============================================================
# POST-LOOP: Global temperature / radiation diagnostics
# ============================================================

println("\n" * "="^60)
println("  POST-LOOP: Global temperature–radiation diagnostics")
println("="^60)

# Load SW and LW (not needed in the main loop)
println("Loading CERES SW and LW radiation...")
ceres_swlw, _ = load_new_ceres_data(["toa_sw_all_mon", "toa_lw_all_mon"], date_range)

# Re-derive ONI valid time indices for the post-loop block
println("Loading ONI for post-loop diagnostics...")
oni_pl, oni_dates_pl = load_enso_data(date_range; lags=enso_lags)
oni_times_pl  = round_dates_down_to_nearest_month(oni_dates_pl["time"])
n_oni_pl      = length(oni_times_pl)
oni_lagmat_pl = Matrix{Union{Float64,Missing}}(missing, n_oni_pl, length(enso_lags))
for (j, lag) in enumerate(enso_lags)
    col = "oni_lag_$(lag)"
    haskey(oni_pl, col) && (oni_lagmat_pl[:, j] .= oni_pl[col])
end
oni_vr        = [all(!ismissing, oni_lagmat_pl[i, :]) for i in 1:n_oni_pl]
oni_mat_pl    = Float64.(oni_lagmat_pl[oni_vr, :])
oni_m_pl, ev_idx = precompute_enso_match(common_time, oni_times_pl[oni_vr], oni_mat_pl)
oni_ts        = Float64.(oni_m_pl[:, findfirst(==(0), lags_vec)])

ev_time   = common_time[ev_idx]
ev_ft     = calc_float_time.(ev_time)
ev_mg     = groupfind(month, ev_time)

# Global means (ENSO-valid time steps only)
gm_t1000 = generate_spatial_mean(θ_1000_gridded[:, :, ev_idx], era5_lat)
gm_t700  = generate_spatial_mean(θ_700_gridded[:, :,  ev_idx], era5_lat)
gm_lts   = generate_spatial_mean(lts_gridded[:, :,    ev_idx], era5_lat)
gm_net   = generate_spatial_mean(ceres_data["toa_net_all_mon"][:, :, ev_idx], ceres_lat)
gm_sw    = generate_spatial_mean(ceres_swlw["toa_sw_all_mon"][:, :,  ev_idx], ceres_lat) .* -1
gm_lw    = generate_spatial_mean(ceres_swlw["toa_lw_all_mon"][:, :,  ev_idx], ceres_lat) .* -1

println("Deseasonalising and detrending global means...")
for ts in (gm_t1000, gm_t700, gm_lts, gm_net, gm_sw, gm_lw)
    deseasonalize_and_detrend_precalculated_groups_twice!(ts, ev_ft, ev_mg;
        aggfunc=mean, trendfunc=least_squares_fit)
end

# ============================================================
# Plot A: Lag correlations — t1000, t700, LTS vs net/SW/LW rad
#         Three subplots; each subplot has one line per radiation type.
#         Net R uses Pearson r; SW/LW use cov / (σ_net · σ_temp) so all
#         three share the same normalisation and are directly additive.
# ============================================================

# cov normalised by σ_net · σ_temp (= Pearson r when rvec is net rad)
function wcov_at_lag(x, y, lag, norm)
    y_lag = time_lag(y, lag)
    valid = .!ismissing.(y_lag)
    cov(x[valid], Float64.(collect(y_lag[valid]))) / norm
end
function max_wcov_lag(x, y, lags, norm)
    best_lag, best_val, best_r2 = first(lags), 0.0, -Inf
    for lag in lags
        v = wcov_at_lag(x, y, lag, norm)
        isfinite(v) && v^2 > best_r2 && (best_r2 = v^2; best_val = v; best_lag = lag)
    end
    return best_lag, best_val
end

# R² of predicting net rad as SW_hat(lag_sw) + LW_hat(lag_lw), where each
# component is fit independently via OLS from the same temperature predictor.
function net_r2_via_sw_lw(tvec, sw, lw, net, lags, nrm)
    lag_sw, _ = max_wcov_lag(tvec, sw, lags, nrm)
    lag_lw, _ = max_wcov_lag(tvec, lw, lags, nrm)
    t_sw = time_lag(tvec, lag_sw)
    t_lw = time_lag(tvec, lag_lw)
    valid = (.!ismissing.(t_sw)) .& (.!ismissing.(t_lw))
    tsw, tlw = Float64.(collect(t_sw[valid])), Float64.(collect(t_lw[valid]))
    sw_hat = hcat(ones(sum(valid)), tsw) * (hcat(ones(sum(valid)), tsw) \ sw[valid])
    lw_hat = hcat(ones(sum(valid)), tlw) * (hcat(ones(sum(valid)), tlw) \ lw[valid])
    net_hat = sw_hat .+ lw_hat
    net_v   = net[valid]
    r2 = 1.0 - sum((net_v .- net_hat).^2) / sum((net_v .- mean(net_v)).^2)
    return r2, lag_sw, lag_lw
end

temp_pairs = [("t₁₀₀₀ (θ)", gm_t1000), ("t₇₀₀ (θ)", gm_t700), ("LTS", gm_lts)]
# use_wcov=true → normalise by σ_net·σ_temp instead of σ_rad·σ_temp
rad_specs  = [("Net R", gm_net, "steelblue",  false),
              ("SW R",  gm_sw,  "darkorange",  true),
              ("LW R",  gm_lw,  "firebrick",   true)]

fig_lc, axs_lc = plt.subplots(1, 3; figsize=pylist([15.0, 4.2]),
                               sharey=true, layout="constrained")
for (i, (tname, tvec)) in enumerate(temp_pairs)
    ax  = axs_lc[i - 1]
    nrm = std(gm_net) * std(tvec)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    for (rname, rvec, rcol, use_wcov) in rad_specs
        if use_wcov
            vals       = [wcov_at_lag(tvec, rvec, lag, nrm) for lag in enso_lags]
            bl, bv     = max_wcov_lag(tvec, rvec, enso_lags, nrm)
        else
            vals       = [cor_at_lag(tvec, rvec, lag) for lag in enso_lags]
            bl, bv     = max_lag_cor(tvec, rvec, enso_lags)
        end
        ax.plot(_np.asarray(convert(Array{Float64}, lags_vec)),
                _np.asarray(convert(Array{Float64}, vals)),
                linewidth=1.8, marker="o", markersize=3, color=rcol,
                label="$(rname)  $(round(bv;digits=2)) @ $(bl) mo")
        ax.scatter(pylist([bl]), pylist([bv]), color=rcol, zorder=5, s=50)
    end
    r2_net, lag_sw, lag_lw = net_r2_via_sw_lw(tvec, gm_sw, gm_lw, gm_net, enso_lags, nrm)
    ax.set_xlim(first(lags_vec) - 0.5, last(lags_vec) + 0.5)
    ax.set_xlabel("Lag (months,  ← rad leads  |  temp leads →)", fontsize=10)
    i == 1 && ax.set_ylabel("Cov / (σ_net · σ_temp)", fontsize=10)
    ax.set_title("Global mean $(tname) vs radiation\n" *
                 "R²(net via SW@$(lag_sw)+LW@$(lag_lw)) = $(round(r2_net; digits=3))",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)
end
fig_lc.suptitle("Global temperature–radiation lag correlations", fontsize=12)
let sp = joinpath(visdir, "global_temp_rad_lag_cors.png")
    fig_lc.savefig(sp, dpi=150, bbox_inches="tight")
    println("Saved: $sp")
end
plt.close(fig_lc)

# ============================================================
# Plot A2: Same as Plot A but SW and LW further decomposed into
#          ENSO and non-ENSO components of the temperature predictor.
#          Five traces per subplot:
#            Net R (Pearson r), SW·T_enso, SW·T_resid, LW·T_enso, LW·T_resid
#          all normalised by σ_net·σ_temp so they sum to Net R at every lag.
# ============================================================

# Decompose a temperature vector into its ENSO and residual components
# using the ONI index at the lag that maximises R² with that vector.
function enso_decompose(tvec, oni, lags)
    best_lag, _ = max_lag_cor(tvec, oni, lags)
    oni_lagged  = time_lag(oni, best_lag)
    valid       = .!ismissing.(oni_lagged)
    x_valid     = Float64.(collect(oni_lagged[valid]))
    β           = least_squares_fit(x_valid, tvec[valid]).slope
    t_enso      = zeros(Float64, length(tvec))
    t_enso[valid] .= β .* x_valid
    return t_enso, tvec .- t_enso, best_lag
end

fig_lc2, axs_lc2 = plt.subplots(1, 3; figsize=pylist([15.0, 4.2]),
                                 sharey=true, layout="constrained")
for (i, (tname, tvec)) in enumerate(temp_pairs)
    ax  = axs_lc2[i - 1]
    nrm = std(gm_net) * std(tvec)

    t_enso, t_resid, enso_lag = enso_decompose(tvec, oni_ts, enso_lags)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")

    # Net R (full temperature, Pearson r — reference trace that the rest sum to)
    net_cors = [cor_at_lag(tvec, gm_net, lag) for lag in enso_lags]
    bl_net, bv_net = max_lag_cor(tvec, gm_net, enso_lags)
    ax.plot(_np.asarray(convert(Array{Float64}, lags_vec)),
            _np.asarray(convert(Array{Float64}, net_cors)),
            color="steelblue", linewidth=2.0, marker="o", markersize=3,
            label="Net R  $(round(bv_net;digits=2)) @ $(bl_net) mo")

    # SW and LW, each split into ENSO and residual components
    sw_lw_decomps = [
        ("SW·T_enso",  gm_sw, t_enso, "darkorange", "-"),
        ("SW·T_resid", gm_sw, t_resid, "darkorange", "--"),
        ("LW·T_enso",  gm_lw, t_enso, "firebrick",  "-"),
        ("LW·T_resid", gm_lw, t_resid, "firebrick",  "--"),
    ]
    for (lbl, rvec, tcomp, rcol, ls) in sw_lw_decomps
        vals   = [wcov_at_lag(tcomp, rvec, lag, nrm) for lag in enso_lags]
        bl, bv = max_wcov_lag(tcomp, rvec, enso_lags, nrm)
        ax.plot(_np.asarray(convert(Array{Float64}, lags_vec)),
                _np.asarray(convert(Array{Float64}, vals)),
                color=rcol, linestyle=ls, linewidth=1.6, marker="o", markersize=2,
                label="$(lbl)  $(round(bv;digits=2)) @ $(bl) mo")
    end

    r2_net, lag_sw, lag_lw = net_r2_via_sw_lw(tvec, gm_sw, gm_lw, gm_net, enso_lags, nrm)
    ax.set_xlim(first(lags_vec) - 0.5, last(lags_vec) + 0.5)
    ax.set_xlabel("Lag (months,  ← rad leads  |  temp leads →)", fontsize=10)
    i == 1 && ax.set_ylabel("Cov / (σ_net · σ_temp)", fontsize=10)
    ax.set_title("Global mean $(tname)  (ONI decomp @ lag $(enso_lag))\n" *
                 "R²(net via SW@$(lag_sw)+LW@$(lag_lw)) = $(round(r2_net; digits=3))",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(labelsize=9)
end
fig_lc2.suptitle("Global temperature–radiation lag correlations  (ENSO / residual decomposition)", fontsize=12)
let sp = joinpath(visdir, "global_temp_rad_lag_cors_enso_decomp.png")
    fig_lc2.savefig(sp, dpi=150, bbox_inches="tight")
    println("Saved: $sp")
end
plt.close(fig_lc2)

# ============================================================
# Plot A3: Net radiation lag correlation decomposed into ENSO and
#          residual temperature components.
#          Three traces per subplot:
#            Net R total (= ENSO + resid, reference)
#            Net R · T_enso
#            Net R · T_resid
#          all normalised by σ_net·σ_temp so the two components sum to Net R.
# ============================================================

fig_lc3, axs_lc3 = plt.subplots(1, 3; figsize=pylist([15.0, 4.2]),
                                 sharey=true, layout="constrained")
for (i, (tname, tvec)) in enumerate(temp_pairs)
    ax  = axs_lc3[i - 1]
    nrm = std(gm_net) * std(tvec)

    t_enso, t_resid, enso_lag = enso_decompose(tvec, oni_ts, enso_lags)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")

    net_total = [cor_at_lag(tvec,   gm_net, lag) for lag in enso_lags]
    net_enso  = [wcov_at_lag(t_enso,  gm_net, lag, nrm) for lag in enso_lags]
    net_resid = [wcov_at_lag(t_resid, gm_net, lag, nrm) for lag in enso_lags]

    bl_tot, bv_tot = max_lag_cor(tvec,    gm_net, enso_lags)
    bl_ens, bv_ens = max_wcov_lag(t_enso,  gm_net, enso_lags, nrm)
    bl_res, bv_res = max_wcov_lag(t_resid, gm_net, enso_lags, nrm)

    for (vals, bl, bv, col, ls, lbl) in [
            (net_total, bl_tot, bv_tot, "steelblue",   "-",  "Net R (total)"),
            (net_enso,  bl_ens, bv_ens, "darkorange",  "-",  "Net R · T_enso"),
            (net_resid, bl_res, bv_res, "forestgreen", "--", "Net R · T_resid"),
        ]
        ax.plot(_np.asarray(convert(Array{Float64}, lags_vec)),
                _np.asarray(convert(Array{Float64}, vals)),
                color=col, linestyle=ls, linewidth=1.8, marker="o", markersize=3,
                label="$(lbl)  $(round(bv;digits=2)) @ $(bl) mo")
        ax.scatter(pylist([bl]), pylist([bv]), color=col, zorder=5, s=50)
    end

    ax.set_xlim(first(lags_vec) - 0.5, last(lags_vec) + 0.5)
    ax.set_xlabel("Lag (months,  ← rad leads  |  temp leads →)", fontsize=10)
    i == 1 && ax.set_ylabel("Cov / (σ_net · σ_temp)", fontsize=10)
    ax.set_title("Global mean $(tname)  (ONI decomp @ lag $(enso_lag))", fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)
end
fig_lc3.suptitle("Net radiation lag correlations  —  ENSO vs residual temperature", fontsize=12)
let sp = joinpath(visdir, "global_net_rad_lag_cors_enso_decomp.png")
    fig_lc3.savefig(sp, dpi=150, bbox_inches="tight")
    println("Saved: $sp")
end
plt.close(fig_lc3)

# ============================================================
# Plot B: R² heatmap — multilinear regression
#         global net rad ~ lagged t₁₀₀₀ + lagged t₇₀₀
#         x-axis = lag of t₁₀₀₀, y-axis = lag of t₇₀₀
# ============================================================
n_lags = length(enso_lags)
r2_mat = fill(NaN, n_lags, n_lags)   # r2_mat[i=lag2_idx, j=lag1_idx]

for (j, lag1) in enumerate(enso_lags)      # lag1 → t₁₀₀₀ (x-axis)
    yl1 = time_lag(gm_t1000, lag1)
    for (i, lag2) in enumerate(enso_lags)  # lag2 → t₇₀₀ (y-axis)
        yl2   = time_lag(gm_t700, lag2)
        valid = (.!ismissing.(yl1)) .& (.!ismissing.(yl2))
        sum(valid) < 12 && continue
        X  = hcat(ones(sum(valid)),
                  Float64.(collect(yl1[valid])),
                  Float64.(collect(yl2[valid])))
        yv = gm_net[valid]
        β  = X \ yv
        ŷ  = X * β
        ss_res = sum((yv .- ŷ).^2)
        ss_tot = sum((yv .- mean(yv)).^2)
        r2_mat[i, j] = 1.0 - ss_res / ss_tot
    end
end

lv = Float64(first(lags_vec)); rv = Float64(last(lags_vec))
fig_h, ax_h = plt.subplots(1, 1; figsize=pylist([7.5, 6.0]))
im = ax_h.imshow(_np.asarray(convert(Array{Float64}, r2_mat)),
                 origin="lower", aspect="auto",
                 extent=pylist([lv - 0.5, rv + 0.5, lv - 0.5, rv + 0.5]),
                 cmap="YlOrRd", vmin=0.0, vmax=maximum(filter(!isnan, vec(r2_mat))))
plt.colorbar(im, ax=ax_h, label="R²", shrink=0.85)
ax_h.set_xlabel("Lag of t₁₀₀₀ (months, + = t₁₀₀₀ leads rad)", fontsize=11)
ax_h.set_ylabel("Lag of t₇₀₀ (months, + = t₇₀₀ leads rad)", fontsize=11)
ax_h.set_title("Multilinear R² : global net rad ~ lagged t₁₀₀₀ + lagged t₇₀₀", fontsize=12)
let sp = joinpath(visdir, "global_rad_t1000_t700_r2_heatmap.png")
    fig_h.savefig(sp, dpi=150, bbox_inches="tight")
    println("Saved: $sp")
end
plt.close(fig_h)

# ============================================================
# Plot C: Standardised regional T2m vs ONI — 2×1 time series
#         Top panel: regional T2m; bottom panel: ONI
#         Panels share x and y axes; one figure per region
# ============================================================
t2m_ev = era5_data["t2m"][:, :, ev_idx]
t_axis = _np.asarray(convert(Array{Float64}, collect(0:length(ev_time)-1)))

for region_name in regions_to_inspect
    reg_t = generate_spatial_mean(t2m_ev, era5_lat, era5_region_mask_dict[region_name])
    deseasonalize_and_detrend_precalculated_groups_twice!(reg_t, ev_ft, ev_mg;
        aggfunc=mean, trendfunc=least_squares_fit)

    best_lag, best_cor = max_lag_cor(reg_t, oni_ts, enso_lags)
    lag_str = best_lag == 0 ? "no lag" :
              best_lag  > 0 ? "region leads ONI by $(best_lag) mo" :
                              "ONI leads region by $(-best_lag) mo"

    reg_std = (reg_t  .- mean(reg_t))  ./ std(reg_t)
    oni_std = (oni_ts .- mean(oni_ts)) ./ std(oni_ts)

    fig_ts, axs_ts = plt.subplots(2, 1; figsize=pylist([13.0, 5.0]),
                                   sharex=true, sharey=true, layout="constrained")

    axs_ts[0].plot(t_axis, _np.asarray(convert(Array{Float64}, reg_std)),
                   color="steelblue", linewidth=1.5)
    axs_ts[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axs_ts[0].set_ylabel("Std. T2m", fontsize=10)
    axs_ts[0].set_title("$(region_name) T2m  ($(lag_str), r=$(round(best_cor;digits=3)))", fontsize=11)

    axs_ts[1].plot(t_axis, _np.asarray(convert(Array{Float64}, oni_std)),
                   color="firebrick", linewidth=1.5)
    axs_ts[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axs_ts[1].set_ylabel("Std. ONI", fontsize=10)
    axs_ts[1].set_xlabel("Months since $(ev_time[1])", fontsize=10)

    fig_ts.suptitle("$(region_name): standardised T2m vs ONI  (ENSO-valid period)", fontsize=12)
    let sp = joinpath(visdir, "$(region_name)_T2m_vs_ONI.png")
        fig_ts.savefig(sp, dpi=150, bbox_inches="tight")
        println("Saved: $sp")
    end
    plt.close(fig_ts)
end
