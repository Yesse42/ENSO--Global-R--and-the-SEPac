# lts_setup.jl
#
# Load ERA5 temperature data, compute global mean θ_1000 and global mean
# LTS, deseasonalize then detrend both.  Then deseasonalize and detrend
# gridded θ_1000 at every pixel and subtract the global mean θ_1000
# anomaly so that only the spatially-local signal remains.
# Note: θ_1000 ≡ T_1000 because (1000/1000)^(2/7) = 1.

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using Dates, Statistics, JLD2

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_kernel/"
mkpath(visdir)

# ============================================================
# Data loading
# ============================================================

date_range = (Date(2000, 3), Date(2024, 2, 28))

println("Loading ERA5 pressure-level temperature...")
era5_data, era5_coords = load_era5_data(["t"], date_range;
                                         pressure_level_file="new_pressure_levels.nc")

era5_lat = era5_coords["latitude"]
era5_lon = era5_coords["longitude"]

# ============================================================
# Potential temperature → θ_700, θ_1000, LTS
# ============================================================

pot_temp(T, P) = T .* (1000f0 ./ P) .^ (2f0/7f0)

press_levels  = era5_coords["pressure_level"]
idx_700       = findfirst(==(700), press_levels)
idx_1000      = findfirst(==(1000), press_levels)

sfc_times     = round_dates_down_to_nearest_month.(era5_coords["time"])
p_times       = round_dates_down_to_nearest_month.(era5_coords["pressure_time"])
valid_p_times = findall(t -> t in sfc_times, p_times)

θ_700_gridded  = pot_temp.(era5_data["t"][:, :, idx_700,  valid_p_times], 700f0)
θ_1000_gridded = pot_temp.(era5_data["t"][:, :, idx_1000, valid_p_times], 1000f0)
lts_gridded    = θ_700_gridded .- θ_1000_gridded

# ============================================================
# Shared time coordinates for all deseasonalization/detrending
# ============================================================

times        = p_times[valid_p_times]
float_times  = calc_float_time.(times)
month_groups = groupfind(month, times)

# ============================================================
# Global mean T (θ_1000) and LTS — deseasonalize then detrend
# ============================================================

println("Computing global mean θ_1000 and LTS...")
gm_T   = generate_spatial_mean(θ_1000_gridded, era5_lat)
gm_lts = generate_spatial_mean(lts_gridded,    era5_lat)

println("Deseasonalizing and detrending global means...")
deseasonalize_and_detrend_precalculated_groups_twice!(gm_T,   float_times, month_groups;
    aggfunc=mean, trendfunc=least_squares_fit)
deseasonalize_and_detrend_precalculated_groups_twice!(gm_lts, float_times, month_groups;
    aggfunc=mean, trendfunc=least_squares_fit)

println("  gm_T   std: $(round(std(gm_T);   sigdigits=4)) K")
println("  gm_lts std: $(round(std(gm_lts); sigdigits=4)) K")

# ============================================================
# Gridded θ_1000 — deseasonalize then detrend at each pixel,
# then subtract the global mean T anomaly
# ============================================================

println("Deseasonalizing and detrending gridded θ_1000 (this may take a moment)...")
θ_1000_anom = Float64.(θ_1000_gridded)
n_lon, n_lat, _ = size(θ_1000_anom)

for i in 1:n_lon, j in 1:n_lat
    ts = @view θ_1000_anom[i, j, :]
    deseasonalize_and_detrend_precalculated_groups_twice!(ts, float_times, month_groups;
        aggfunc=mean, trendfunc=least_squares_fit)
end

println("Subtracting global mean T from gridded θ_1000 anomaly...")
θ_1000_local = θ_1000_anom .- reshape(gm_T, 1, 1, :)

println("Done.")
println("  θ_1000_anom  global mean std (should ≈ gm_T std): " *
        "$(round(std(generate_spatial_mean(θ_1000_anom, era5_lat)); sigdigits=4)) K")
println("  θ_1000_local global mean std (should ≈ 0):        " *
        "$(round(std(generate_spatial_mean(θ_1000_local, era5_lat)); sigdigits=4)) K")

# ============================================================
# Regression utilities
# ============================================================

using LinearAlgebra
@py import sklearn.linear_model as sklearn_lm, sklearn.decomposition as sklearn_decomp
_np = pyimport("numpy")

"""
    fit_spatial_ridge(X_gridded, y, lats; alphas) → (ŷ, resid, α_opt, Q2)

Predict scalar time series `y` (n_time,) from spatial anomaly field
`X_gridded` (n_lon × n_lat × n_time) using area-weighted ridge regression.

Each feature (grid point) is scaled by √cos(lat) so the ridge penalty
operates in an equal-area-weighted feature space.  The penalty α is
chosen by GCV via the SVD (sklearn RidgeCV, gcv_mode="svd").

Q² is computed from the GCV formula:
    PRESS_GCV = ‖y − ŷ‖² / (1 − tr(Hα)/n)²
    Q² = 1 − PRESS_GCV / SS_tot
using the n×n Gram matrix X_w Xw' to obtain singular values efficiently
when n_features ≫ n_time.
"""
function fit_spatial_ridge(X_gridded::Array{<:Real,3}, y::Vector{<:Real},
                            lats::AbstractVector;
                            n_components = 50,
                            alphas = 10 .^ range(-3, 6; length=100))
    n_lon, n_lat, n_time = size(X_gridded)

    # Flatten (n_lon, n_lat, n_time) → (n_time, n_lon*n_lat)
    X_flat = reshape(permutedims(X_gridded, (3, 1, 2)), n_time, n_lon * n_lat)

    # Area weights: √cos(lat) tiled over all lons (coerce away any Missing)
    lats_f = Float64.(collect(skipmissing(lats)))
    w   = vec(repeat(sqrt.(cosd.(lats_f))', n_lon, 1))   # (n_lon*n_lat,)
    X_w = X_flat .* w'                                    # (n_time, n_lon*n_lat)

    # Project onto leading EOFs — puts us firmly in n_components ≪ n_time
    # where GCV is well-behaved (avoids the p≫n Q²=1 pathology)
    pca    = sklearn_decomp.PCA(n_components = n_components)
    X_pc   = pca.fit_transform(_np.asarray(convert(Array{Float64}, X_w)))
    PythonCall.pydel!(pca)
    X_pc_jl = pyconvert(Matrix{Float64}, X_pc)
    PythonCall.pydel!(X_pc)
    GC.gc()

    # GCV-optimal ridge on PC scores
    rcv = sklearn_lm.RidgeCV(
        alphas           = _np.asarray(convert(Array{Float64}, collect(alphas))),
        gcv_mode         = "svd",
        fit_intercept    = true,
        store_cv_results = false,
    )
    rcv.fit(_np.asarray(convert(Array{Float64}, X_pc_jl)),
            _np.asarray(convert(Array{Float64}, y)))

    α   = pyconvert(Float64,         rcv.alpha_)
    β   = pyconvert(Vector{Float64}, rcv.coef_)
    b   = pyconvert(Float64,         rcv.intercept_)
    PythonCall.pydel!(rcv)
    GC.gc()

    ŷ   = X_pc_jl * β .+ b

    # GCV Q²: SVD of the small (n_time × n_components) PC matrix
    _, sv, _ = svd(X_pc_jl; full = false)
    sv_sq  = sv .^ 2
    tr_H   = sum(sv_sq ./ (sv_sq .+ α))
    h_avg  = tr_H / n_time
    Q2     = 1.0 - sum(((y .- ŷ) ./ (1.0 - h_avg)) .^ 2) /
                   sum((y .- mean(y)) .^ 2)

    return ŷ, y .- ŷ, α, Q2
end

"""
    fit_ols_1d(x, y) → (ŷ, resid, r)

Single-predictor OLS.  Returns prediction, residual, and Pearson r.
"""
function fit_ols_1d(x::Vector{<:Real}, y::Vector{<:Real})
    f = least_squares_fit(x, y)
    ŷ = f.slope .* x .+ f.intercept
    return ŷ, y .- ŷ, cor(x, y)
end

# ============================================================
# Lag loop: run both orderings for LTS lags −4 … +4 months.
#
# Trimming strategy: remove the first and last `max_lag` time steps
# from ALL series so every lagged version of gm_lts is fully valid
# (no Missing values) and all analyses use an identical time window.
#
# Positive lag l → gm_lts is shifted forward l months
#   (temperature today predicts LTS l months later).
# ============================================================

const max_lag    = 4
const lts_lags   = -max_lag:max_lag
const trim_range = (max_lag + 1):(length(gm_lts) - max_lag)   # consistent valid window

gm_T_trim      = gm_T[trim_range]
θ_local_trim   = θ_1000_local[:, :, trim_range]
times_trim     = times[trim_range]

# X-axis tick positions (same for every lag plot)
_t_trim    = convert(Array{Float64}, collect(1:length(trim_range)))
jan_idx_tr = findall(month.(times_trim) .== 1)
jan_labels = string.(year.(times_trim[jan_idx_tr]))

row_colors = ["steelblue", "darkorange", "forestgreen"]

for lag in lts_lags
    println("\n" * "="^60)
    println("  LTS lag = $(lag) months")
    println("="^60)

    # Lagged LTS over the trimmed window (no missing values by construction)
    gm_lts_lag = gm_lts[(trim_range) .+ lag]

    # ----------------------------------------------------------
    # Ordering 1: spatial pattern first, then global mean T
    # ----------------------------------------------------------
    println("--- Ordering 1: spatial pattern first ---")
    pred_sp1, resid_sp1, α1, Q2_1          = fit_spatial_ridge(θ_local_trim, gm_lts_lag, era5_lat)
    pred_gm_on_resid1, final_resid1, r_gm1 = fit_ols_1d(gm_T_trim, resid_sp1)
    println("  Spatial ridge:            α=$(round(α1; sigdigits=3))  Q²=$(round(Q2_1; digits=3))")
    println("  Global mean OLS on resid: r=$(round(r_gm1; digits=3))")

    # ----------------------------------------------------------
    # Ordering 2: global mean T first, then spatial pattern
    # ----------------------------------------------------------
    println("--- Ordering 2: global mean T first ---")
    pred_gm2, resid_gm2, r_gm2                 = fit_ols_1d(gm_T_trim, gm_lts_lag)
    pred_sp_on_resid2, final_resid2, α2, Q2_2   = fit_spatial_ridge(θ_local_trim, resid_gm2, era5_lat)
    println("  Global mean OLS:          r=$(round(r_gm2; digits=3))")
    println("  Spatial ridge on resid:   α=$(round(α2; sigdigits=3))  Q²=$(round(Q2_2; digits=3))")

    # ----------------------------------------------------------
    # 3×2 plot — equivalent predictor steps on the same row
    # ----------------------------------------------------------
    all_vals = vcat(pred_sp1, pred_gm_on_resid1, final_resid1,
                    pred_gm2, pred_sp_on_resid2, final_resid2)
    ylim_abs = maximum(abs, all_vals) * 1.1

    lag_str  = lag == 0 ? "lag 0" : (lag > 0 ? "LTS leads +$(lag) mo" : "T leads +$(-lag) mo")
    lag_sign = lag >= 0 ? "+$(lag)" : "$(lag)"

    plot_rows = [
        (pred_gm2,     pred_gm_on_resid1,
         "gm_T → gm_LTS  (ord.2 step 1)",       "gm_T → resid after spatial  (ord.1 step 2)",
         "gm_T component (K)"),
        (pred_sp1,     pred_sp_on_resid2,
         "spatial → gm_LTS  (ord.1 step 1)",     "spatial → resid after gm_T  (ord.2 step 2)",
         "Spatial component (K)"),
        (final_resid1, final_resid2,
         "Final residual  (ord.1)",               "Final residual  (ord.2)",
         "Final residual (K)"),
    ]

    subplots = []
    for (ri, (lseries, rseries, ltitle, rtitle, ylabel)) in enumerate(plot_rows)
        r_cross = round(cor(lseries, rseries);    digits = 3)
        r_left  = round(cor(lseries, gm_lts_lag); digits = 3)
        r_right = round(cor(rseries, gm_lts_lag); digits = 3)
        col     = row_colors[ri]

        for (ci, (series, title, r_lts)) in enumerate([
                (lseries, ltitle, r_left),
                (rseries, rtitle, r_right)])

            xtick_arg = ri == 3 ? (jan_idx_tr, jan_labels) :
                                   (jan_idx_tr, fill("", length(jan_idx_tr)))

            p_sub = plot(_t_trim, series;
                title         = "$(title)\nr(vs gm_LTS) = $(r_lts)" *
                                (ci == 2 ? "    r(L,R) = $(r_cross)" : ""),
                titlefontsize = 8,
                color         = col,
                linewidth     = 1.5,
                label         = false,
                ylabel        = ci == 1 ? ylabel : "",
                guidefontsize = 8,
                xticks        = xtick_arg,
                xrotation     = 45,
                tickfontsize  = 7,
                ylims         = (-ylim_abs, ylim_abs),
                grid          = true,
                gridalpha     = 0.4,
                gridstyle     = :dash,
                gridlinewidth = 0.5,
            )
            hline!(p_sub, [0.0]; color = :black, linewidth = 0.8,
                   linestyle = :dash, label = false)
            push!(subplots, p_sub)
        end
    end

    fig = plot(subplots...;
               layout             = (3, 2),
               size               = (1300, 950),
               plot_title         = "gm_LTS decomposition — spatial T₁₀₀₀ vs global mean T₁₀₀₀  " *
                                    "[$(lag_str)]",
               plot_titlefontsize = 10,
               bottom_margin      = 8Plots.mm,
               left_margin        = 6Plots.mm,
               )

    let sp = joinpath(visdir, "lts_ordering_comparison_lag$(lag_sign).png")
        savefig(fig, sp)
        println("Saved: $sp")
    end
end
