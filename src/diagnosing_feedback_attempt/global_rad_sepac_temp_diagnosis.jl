#= Following Lorenz and Hartmann 2001, this script determines if there is a feedback
between radiation and temperature/LTS in stratocumulus regions (SEPac, NEPac).
For each region, analyses (power spectrum, lag correlation, autocorrelation) are
performed on total, ENSO, and residual components of global and local radiation
paired against regional T₁₀₀₀ and LTS. =#

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../area_averager/area_averager.jl")
include("../pls_regressor/pls_functions.jl")

using JLD2, Dates, Statistics, NCDatasets, FFTW

#Section 1: Load data

jldpath = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/"
era5_region_mask_dict = jldopen(joinpath(jldpath, "stratocumulus_region_masks.jld2"))["regional_masks_era5"]
rad_region_mask_dict  = jldopen(joinpath(jldpath, "stratocumulus_exclusion_masks.jld2"))["stratocum_masks"]

region_keys = ["SEPac", "NEPac"]
# JLD2 keys differ by region: SEPac uses "SEPac_feedback_definition", NEPac uses "NEPac"
jld2_mask_key = Dict("SEPac" => "SEPac_feedback_definition", "NEPac" => "NEPac")
temp_masks = Dict(r => era5_region_mask_dict[jld2_mask_key[r]] for r in region_keys)
rad_masks  = Dict(r => rad_region_mask_dict[jld2_mask_key[r]]  for r in region_keys)

# CERES: global radiation + per-region local radiation
ceres_daily_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/daily/CERES_SYN1deg-Day_merged.nc"
ceres_ds      = NCDataset(ceres_daily_file)
ceres_rad_var = ceres_ds["toa_net_all_daily"]
ceres_lat     = ceres_ds["lat"][:]
ceres_time    = round.(ceres_ds["time"][:], Day(1), RoundDown)

global_mean_rad = generate_spatial_mean_netcdf_compatible(ceres_rad_var, ceres_lat)
println("Generated global mean radiation")
local_rad_raw = Dict(r => generate_spatial_mean_netcdf_compatible(ceres_rad_var, ceres_lat, rad_masks[r]) for r in region_keys)
println("Generated local radiation for all regions")

close(ceres_ds)

# ERA5: per-region T₁₀₀₀ and LTS
era5_daily_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/ERA5/daily_data/hourly_t_data_2000_2025.nc"
era5_ds    = NCDataset(era5_daily_file)
era5_t_var = era5_ds["t"]
era5_lat   = era5_ds["latitude"][:]
era5_time  = era5_ds["time"][:]
p_levels   = era5_ds["isobaricInhPa"][:]
p_1000_idx = findfirst(==(1000), p_levels)
p_700_idx  = findfirst(==(700),  p_levels)

t_1000_raw = Dict(r => generate_spatial_mean_netcdf_and_p_level_compatible(era5_t_var, era5_lat, temp_masks[r], p_1000_idx) for r in region_keys)
t_700_raw  = Dict(r => generate_spatial_mean_netcdf_and_p_level_compatible(era5_t_var, era5_lat, temp_masks[r], p_700_idx)  for r in region_keys)
close(era5_ds)
theta_700_raw = Dict(r => t_700_raw[r] .* (1000/700)^(2/7) for r in region_keys)
LTS_raw    = Dict(r => (theta_700_raw[r]) .- (t_1000_raw[r]) for r in region_keys)
println("Generated T₁₀₀₀ and LTS for all regions")

# Global (cos-weighted) T₁₀₀₀ and LTS
global_t_1000_raw   = generate_spatial_mean_netcdf_and_p_level_compatible(era5_t_var, era5_lat, nothing, p_1000_idx)
global_t_700_raw    = generate_spatial_mean_netcdf_and_p_level_compatible(era5_t_var, era5_lat, nothing, p_700_idx)
global_theta_700_raw = global_t_700_raw .* (1000/700)^(2/7)
global_LTS_raw      = global_theta_700_raw .- global_t_1000_raw
println("Generated global T₁₀₀₀ and global LTS")

# ENSO lag matrix (monthly → daily via linear interpolation, mid-month centering)
println("Loading ENSO data...")
"""
    prepare_enso_lag_matrix_daily(enso_data, enso_dates_raw, lags)

Build the ENSO lag matrix from monthly data and upsample to daily resolution via
linear interpolation. Monthly values are centered at the 15th of each month before
interpolating so that the value represents the middle of the month rather than the
beginning.
"""
function prepare_enso_lag_matrix_daily(enso_data, enso_dates_raw, lags)
    lag_columns = ["oni_lag_$lag" for lag in lags]
    enso_times_monthly = enso_dates_raw["time"]  # already ~mid-month from load_enso_data
    n_enso = length(enso_times_monthly)
    enso_lag_matrix_full = Matrix{Union{Float64, Missing}}(missing, n_enso, length(lag_columns))
    for (j, col) in enumerate(lag_columns)
        haskey(enso_data, col) && (enso_lag_matrix_full[:, j] .= enso_data[col])
    end
    valid_rows = [all(!ismissing, enso_lag_matrix_full[i, :]) for i in 1:n_enso]
    monthly_matrix = Float64.(enso_lag_matrix_full[valid_rows, :])

    # Center each monthly value at the 15th of its month
    monthly_times = enso_times_monthly[valid_rows]
    mid_month_times = DateTime.(year.(monthly_times), month.(monthly_times), 15)

    # Build daily time axis and linearly interpolate
    daily_times    = minimum(mid_month_times):Day(1):maximum(mid_month_times)
    n_daily        = length(daily_times)
    n_cols         = size(monthly_matrix, 2)
    t0             = minimum(mid_month_times)
    monthly_days   = Dates.value.(mid_month_times .- t0)
    daily_day_vals = Dates.value.(collect(daily_times) .- t0)

    daily_matrix = Matrix{Float64}(undef, n_daily, n_cols)
    for j in 1:n_cols
        vals = monthly_matrix[:, j]
        for (i, d) in enumerate(daily_day_vals)
            idx = searchsortedlast(monthly_days, d)
            if idx == 0
                daily_matrix[i, j] = vals[1]
            elseif idx == length(monthly_days)
                daily_matrix[i, j] = vals[end]
            else
                t1, t2 = monthly_days[idx], monthly_days[idx + 1]
                daily_matrix[i, j] = vals[idx] + (vals[idx + 1] - vals[idx]) * (d - t1) / (t2 - t1)
            end
        end
    end
    return daily_times, daily_matrix
end

enso_data, enso_dates_raw = load_enso_data((Date(0), Date(5000)))
enso_times_valid, enso_lag_matrix = prepare_enso_lag_matrix_daily(enso_data, enso_dates_raw, -12:12)
println("Prepared ENSO lag matrix with daily resolution (linear interpolation)")

# Common time axis (ERA5, CERES, ENSO all aligned)
common_times = sort!(intersect(era5_time, ceres_time, enso_times_valid))
era5_idx   = findall(in(common_times), era5_time)
ceres_idx  = findall(in(common_times), ceres_time)
enso_idx   = findall(in(common_times), enso_times_valid)
enso_lag_matrix_valid = enso_lag_matrix[enso_idx, :]

# Build var_dict: "global_rad" + global T₁₀₀₀/θ₇₀₀/LTS + per-region "$(r)_local_rad", "$(r)_t_1000", "$(r)_theta_700", "$(r)_LTS"
# All sliced to common_times and cast to Float64.
var_keys = ["global_rad", "global_t_1000", "global_theta_700", "global_LTS"]
var_vals = Vector{Float64}[
    Float64.(global_mean_rad[ceres_idx]),
    Float64.(global_t_1000_raw[era5_idx]),
    Float64.(global_theta_700_raw[era5_idx]),
    Float64.(global_LTS_raw[era5_idx]),
]
for r in region_keys
    push!(var_keys, "$(r)_local_rad", "$(r)_t_1000", "$(r)_theta_700", "$(r)_LTS")
    push!(var_vals,
        Float64.(local_rad_raw[r][ceres_idx]),
        Float64.(t_1000_raw[r][era5_idx]),
        Float64.(theta_700_raw[r][era5_idx]),
        Float64.(LTS_raw[r][era5_idx]))
end
var_dict = Dictionary(var_keys, var_vals)
println("Built var_dict: $(join(var_keys, ", "))")

# Detrend and deseasonalize all variables
month_day_groups = precalculate_daily_data_groups(common_times)
float_times = calc_float_time.(common_times)
for (key, v) in pairs(var_dict)
    println("Detrending and deseasonalizing $key...")
    deseasonalize_and_detrend_daily_data_precalculated_groups_twice!(v, float_times, month_day_groups)
end

#Section 2: ENSO decomposition + analysis

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/global_rad_sepac_temp_diagnosis"
mkpath(visdir)

@py import matplotlib.pyplot as plt, matplotlib.ticker as ticker

"""
    enso_decompose_scalar(enso_matrix, y)

Decompose scalar time series `y` into its ENSO component and residual
via 1-component PLS on the ENSO lag matrix.
Returns `(enso_component, residual)`.
"""
function enso_decompose_scalar(enso_matrix::Matrix, y::AbstractVector)
    pls_model = make_pls_regressor(enso_matrix, y, 1; print_updates=false)
    enso_comp = vec(predict(pls_model, enso_matrix))
    return enso_comp, y .- enso_comp
end

"""
    compute_power_spectrum(signal; dt=1.0)

Compute the one-sided power spectral density of a signal.
Returns (frequencies, power) where frequencies are in cycles per unit time (dt units).
"""
function compute_power_spectrum(signal; dt=1.0)
    n = length(signal)
    signal_centered = signal .- mean(signal)
    fft_result = fft(signal_centered)
    power = abs2.(fft_result[1:div(n, 2)+1])
    power[2:end-1] .*= 2
    power ./= n^2
    freqs = fftfreq(n, 1/dt)[1:div(n, 2)+1]
    return freqs, power
end

"""
    compute_lag_corr_contributions(decomp, rad_key, temp_key, lags)

Compute the lag-correlation decomposition between a radiation and temperature variable.
- total: standard Pearson lag correlation of the total series
- enso:  normalized covariance ENSO contribution (enso×enso + both cross terms)
- resid: normalized covariance residual contribution (resid×resid)
The three contributions sum to approximately the total correlation.
"""
function compute_lag_corr_contributions(decomp, rad_key, temp_key, lags)
    rad_total  = decomp[rad_key][:total];  temp_total = decomp[temp_key][:total]
    rad_enso   = decomp[rad_key][:enso];   temp_enso  = decomp[temp_key][:enso]
    rad_resid  = decomp[rad_key][:resid];  temp_resid = decomp[temp_key][:resid]
    norm_cov   = let σ = std(rad_total) * std(temp_total)
        (x, y) -> cov(x, y; corrected=true) / σ
    end

    total = calculate_lag_corrs_lag_in_place(rad_total, temp_total, lags)
    ee    = calculate_lag_func_lag_in_place(rad_enso,  temp_enso,  lags; func=norm_cov)
    er    = calculate_lag_func_lag_in_place(rad_enso,  temp_resid, lags; func=norm_cov)
    re    = calculate_lag_func_lag_in_place(rad_resid, temp_enso,  lags; func=norm_cov)
    enso  = Dictionary(lags, [ee[l] + er[l] + re[l] for l in lags])
    resid = calculate_lag_func_lag_in_place(rad_resid, temp_resid, lags; func=norm_cov)
    return Dict("total" => total, "enso" => enso, "resid" => resid)
end

"""
    compute_lts_autocorr_decomposition(decomp, lts_key, t1000_key, theta700_key, lags)

Decompose the LTS autocorrelation into θ₇₀₀×θ₇₀₀, (-T₁₀₀₀)×(-T₁₀₀₀), and cross-term
contributions, normalized by Var(LTS_total). Applied independently to each component
(total/enso/resid). Cross = -Cov(θ, T+τ) - Cov(T, θ+τ), which together with the
two auto-terms sums to Cov(LTS, LTS+τ) / Var(LTS_total).
"""
function compute_lts_autocorr_decomposition(decomp, lts_key, t1000_key, theta700_key, lags)
    σ2_lts = Statistics.var(decomp[lts_key][:total])
    norm = (x, y) -> cov(x, y; corrected=true) / σ2_lts
    result = Dict{String, Dict{String, Any}}()
    for component in (:total, :enso, :resid)
        θ = decomp[theta700_key][component]
        T = decomp[t1000_key][component]
        θθ = calculate_lag_func_lag_in_place(θ, θ, lags; func=norm)
        TT = calculate_lag_func_lag_in_place(T, T, lags; func=norm)
        θT = calculate_lag_func_lag_in_place(θ, T, lags; func=norm)
        Tθ = calculate_lag_func_lag_in_place(T, θ, lags; func=norm)
        cross = Dictionary(lags, [-θT[l] - Tθ[l] for l in lags])
        result[string(component)] = Dict("theta700" => θθ, "neg_t1000" => TT, "cross" => cross)
    end
    return result
end

"""
    compute_lts_lagcorr_decomposition(decomp, rad_key, lts_key, t1000_key, theta700_key, lags)

Decompose the lag correlation of radiation vs LTS into θ₇₀₀ and -T₁₀₀₀ contributions
(since LTS = θ₇₀₀ - T₁₀₀₀, Cov(R, LTS) = Cov(R, θ₇₀₀) - Cov(R, T₁₀₀₀)).
Follows the same total/enso/resid decomposition as compute_lag_corr_contributions,
normalizing by σ(R_total) * σ(LTS_total).
"""
function compute_lts_lagcorr_decomposition(decomp, rad_key, lts_key, t1000_key, theta700_key, lags)
    σ_rad = std(decomp[rad_key][:total])
    σ_lts = std(decomp[lts_key][:total])
    norm = (x, y) -> cov(x, y; corrected=true) / (σ_rad * σ_lts)

    R_t = decomp[rad_key][:total];  θ_t = decomp[theta700_key][:total];  T_t = decomp[t1000_key][:total]
    R_e = decomp[rad_key][:enso];   θ_e = decomp[theta700_key][:enso];   T_e = decomp[t1000_key][:enso]
    R_r = decomp[rad_key][:resid];  θ_r = decomp[theta700_key][:resid];  T_r = decomp[t1000_key][:resid]

    θ_total = calculate_lag_func_lag_in_place(R_t, θ_t, lags; func=norm)
    T_total = calculate_lag_func_lag_in_place(R_t, T_t, lags; func=norm)

    θ_ee = calculate_lag_func_lag_in_place(R_e, θ_e, lags; func=norm)
    θ_er = calculate_lag_func_lag_in_place(R_e, θ_r, lags; func=norm)
    θ_re = calculate_lag_func_lag_in_place(R_r, θ_e, lags; func=norm)
    T_ee = calculate_lag_func_lag_in_place(R_e, T_e, lags; func=norm)
    T_er = calculate_lag_func_lag_in_place(R_e, T_r, lags; func=norm)
    T_re = calculate_lag_func_lag_in_place(R_r, T_e, lags; func=norm)
    θ_enso = Dictionary(lags, [θ_ee[l] + θ_er[l] + θ_re[l] for l in lags])
    T_enso = Dictionary(lags, [T_ee[l] + T_er[l] + T_re[l] for l in lags])

    θ_resid = calculate_lag_func_lag_in_place(R_r, θ_r, lags; func=norm)
    T_resid = calculate_lag_func_lag_in_place(R_r, T_r, lags; func=norm)

    return Dict(
        "total" => Dict("theta700" => θ_total, "neg_t1000" => Dictionary(lags, [-T_total[l] for l in lags])),
        "enso"  => Dict("theta700" => θ_enso,  "neg_t1000" => Dictionary(lags, [-T_enso[l]  for l in lags])),
        "resid" => Dict("theta700" => θ_resid, "neg_t1000" => Dictionary(lags, [-T_resid[l] for l in lags])),
    )
end

# ENSO decomposition for all variables
println("Decomposing variables into ENSO and residual components...")
decomp = Dict{String, NamedTuple}()
for (key, v) in pairs(var_dict)
    enso_comp, resid = enso_decompose_scalar(enso_lag_matrix_valid, v)
    decomp[key] = (; total=v, enso=enso_comp, resid=resid)
    println("  $key: var(total)=$(round(Statistics.var(v), sigdigits=3)), var(enso)=$(round(Statistics.var(enso_comp), sigdigits=3)), var(resid)=$(round(Statistics.var(resid), sigdigits=3))")
end

lags_days = -60:60
autocorr_lags_days = -120:120

# Power spectra (per variable, shared across regions)
println("Computing power spectra...")
power_spectra = Dict{String, Dict{String, Any}}()
for (key, d) in decomp
    power_spectra[key] = Dict{String, Any}()
    for component in [:total, :enso, :resid]
        freqs, ps = compute_power_spectrum(d[component])
        power_spectra[key][string(component)] = (; freqs=freqs, power=ps)
    end
    println("  Power spectra computed for $key")
end

# Autocorrelations (per variable, shared across regions)
println("Computing autocorrelations...")
autocorr_results = Dict{String, Dict{String, Any}}()
for (key, d) in decomp
    autocorr_results[key] = Dict{String, Any}()
    σ2_total = Statistics.var(d[:total])
    norm_var  = (x, y) -> cov(x, y; corrected=true) / σ2_total
    autocorr_results[key]["total"] = calculate_lag_corrs_lag_in_place(d[:total], d[:total], autocorr_lags_days)
    autocorr_results[key]["enso"]  = calculate_lag_func_lag_in_place(d[:enso],  d[:enso],  autocorr_lags_days; func=norm_var)
    autocorr_results[key]["resid"] = calculate_lag_func_lag_in_place(d[:resid], d[:resid], autocorr_lags_days; func=norm_var)
    println("  Autocorrelations computed for $key")
end

# Lag correlations: per region, for (global_rad, region_local_rad) × (region_t_1000, region_LTS)
println("Computing lag correlations and covariance contributions...")
lag_corr_results = Dict{String, Dict{String, Dict{String, Any}}}()
for r in region_keys
    lag_corr_results[r] = Dict{String, Dict{String, Any}}()
    for rad_key in ["global_rad", "$(r)_local_rad"]
        lag_corr_results[r][rad_key] = Dict{String, Any}()
        for temp_key in ["$(r)_t_1000", "$(r)_LTS"]
            lag_corr_results[r][rad_key][temp_key] = compute_lag_corr_contributions(decomp, rad_key, temp_key, lags_days)
            println("  $r: $rad_key vs $temp_key")
        end
    end
end

#Section 3: Plots

"""
    corr_significance_threshold(n_total, max_lag; α=0.05)

Return the two-tailed significance threshold for a Pearson correlation using
Fisher's z-transform.  Under H₀ (ρ = 0), z = atanh(r) is approximately normal
with variance 1/(n - 3), so the critical r is:

    r_crit = tanh(z_{α/2} / √(n_eff - 3))

where n_eff = n_total - max_lag is the most conservative (smallest) effective
sample size across all lags (n decreases by 1 per lag step).
"""
function corr_significance_threshold(n_total::Int, max_lag::Int; α::Float64=0.05)
    # Normal quantile z_{α/2} via rational approximation (error < 5e-4 for α ∈ [0.001, 0.1])
    p = 1.0 - α / 2
    u = sqrt(-2log(1 - p))
    z_crit = u - (2.515517 + 0.802853u + 0.010328u^2) /
                 (1 + 1.432788u + 0.189269u^2 + 0.001308u^3)
    n_eff = n_total - max_lag
    return tanh(z_crit / sqrt(n_eff - 3))
end

component_colors = Dict("total" => "black", "enso" => "tab:red", "resid" => "tab:blue")
component_labels = Dict("total" => "Total", "enso" => "ENSO", "resid" => "Non-ENSO")
lags_vec = collect(lags_days)
autocorr_lags_vec = collect(autocorr_lags_days)

n_total      = length(common_times)
sig_autocorr = corr_significance_threshold(n_total, maximum(abs.(collect(autocorr_lags_days))))
sig_lagcorr  = corr_significance_threshold(n_total, maximum(abs.(collect(lags_days))))
println("Significance thresholds — autocorr: $(round(sig_autocorr, sigdigits=4)), lagcorr: $(round(sig_lagcorr, sigdigits=4))")

var_titles = Dict{String, String}(
    "global_rad"      => "Global Net Radiation",
    "global_t_1000"   => "Global T₁₀₀₀",
    "global_theta_700" => "Global θ₇₀₀",
    "global_LTS"      => "Global LTS",
)
temp_xlabels = Dict{String, String}(
    "global_t_1000" => "Lag (days)  [positive = R lags T]",
    "global_LTS"    => "Lag (days)  [positive = R lags LTS]",
)
for r in region_keys
    var_titles["$(r)_local_rad"]  = "$(r) Local Net Radiation"
    var_titles["$(r)_t_1000"]     = "$(r) T₁₀₀₀"
    var_titles["$(r)_theta_700"]  = "$(r) θ₇₀₀"
    var_titles["$(r)_LTS"]        = "$(r) LTS"
    temp_xlabels["$(r)_t_1000"]   = "Lag (days)  [positive = R lags T]"
    temp_xlabels["$(r)_LTS"]      = "Lag (days)  [positive = R lags LTS]"
end

# Shorten the radiation key to a filename-safe tag
rad_file_tag(rad_key) = rad_key == "global_rad" ? "global" : split(rad_key, "_")[1] * "_local"

for r in region_keys
    rad_keys_r  = ["global_rad", "$(r)_local_rad"]
    temp_keys_r = ["$(r)_t_1000", "$(r)_LTS"]

    # --- Lag correlations: one figure per radiation type, 2 subplots (one per temp var) ---
    for rad_key in rad_keys_r
        println("Plotting lag correlations: $r / $rad_key vs temperatures...")
        fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=true)
        axs_vec = collect(axs)
        for (ax, temp_key) in zip(axs_vec, temp_keys_r)
            for component in ["total", "enso", "resid"]
                corr_dict = lag_corr_results[r][rad_key][temp_key][component]
                ax.plot(lags_vec, [corr_dict[lag] for lag in lags_days],
                    color=component_colors[component],
                    label=component_labels[component],
                    linewidth=1.8)
            end
            ax.axhline( sig_lagcorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
            ax.axhline(-sig_lagcorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel(temp_xlabels[temp_key])
            ax.set_title("$(var_titles[rad_key]) vs $(var_titles[temp_key])")
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
            ax.grid(true, which="major", alpha=0.3)
        end
        axs_vec[1].set_ylabel("Correlation / Normalized Covariance")
        axs_vec[1].legend()
        fig.suptitle("$r: Lag Correlations — $(var_titles[rad_key])", fontsize=12)
        fig.tight_layout()
        fig.savefig(joinpath(visdir, "lag_corr_$(r)_$(rad_file_tag(rad_key))_vs_temps.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    end

    # --- Power spectra: one figure per region, 4 subplots (2 rad + 2 temp) ---
    println("Plotting power spectra for $r...")
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for (ax, var_key) in zip(collect(axs), [rad_keys_r..., temp_keys_r...])
        for component in ["total", "enso", "resid"]
            ps_data = power_spectra[var_key][component]
            valid   = ps_data.freqs .> 0
            periods = 1.0 ./ ps_data.freqs[valid]
            ax.loglog(periods, ps_data.power[valid],
                color=component_colors[component],
                label=component_labels[component],
                linewidth=1.5, alpha=0.85)
        end
        ax.invert_xaxis()
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Power")
        ax.set_title(var_titles[var_key])
        ax.legend(fontsize=9)
        ax.grid(true, which="both", alpha=0.3)
    end
    fig.suptitle("$r: Power Spectra", fontsize=13)
    fig.tight_layout()
    fig.savefig(joinpath(visdir, "power_spectra_$(r).png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Autocorrelations: one figure per region, 4 subplots (2 rad + 2 temp) ---
    println("Plotting autocorrelations for $r...")
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for (ax, var_key) in zip(collect(axs), [rad_keys_r..., temp_keys_r...])
        for component in ["total", "enso", "resid"]
            corr_dict = autocorr_results[var_key][component]
            ax.plot(autocorr_lags_vec, [corr_dict[lag] for lag in autocorr_lags_days],
                color=component_colors[component],
                label=component_labels[component],
                linewidth=1.8)
        end
        ax.axhline( sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
        ax.axhline(-sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Autocorrelation / Normalized Covariance")
        ax.set_title(var_titles[var_key])
        ax.legend(fontsize=9)
        ax.grid(true, which="major", alpha=0.3)
    end
    fig.suptitle("$r: Autocorrelations", fontsize=13)
    fig.tight_layout()
    fig.savefig(joinpath(visdir, "autocorrelations_$(r).png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
end

# --- ENSO autocorr, Global Rad autocorr, ENSO vs Global Rad lag corr ---
println("Plotting ENSO / Global Rad autocorr and lag corr...")
enso_lag0_col   = findfirst(==(0), collect(-12:12))
enso_index      = enso_lag_matrix_valid[:, enso_lag0_col]   # ONI (lag-0), daily-interpolated
global_rad_total = decomp["global_rad"][:total]

enso_autocorr          = calculate_lag_corrs_lag_in_place(enso_index,      enso_index,       autocorr_lags_days)
enso_globalrad_lagcorr = calculate_lag_corrs_lag_in_place(enso_index,      global_rad_total, autocorr_lags_days)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs_vec  = collect(axs)

let ax = axs_vec[1]
    ax.plot(autocorr_lags_vec, [enso_autocorr[lag] for lag in autocorr_lags_days], color="tab:orange", linewidth=1.8)
    ax.axhline( sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
    ax.axhline(-sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ENSO Index (ONI) Autocorrelation")
    ax.legend(fontsize=9)
    ax.grid(true, which="major", alpha=0.3)
end

let ax = axs_vec[2]
    corr_dict = autocorr_results["global_rad"]["total"]
    ax.plot(autocorr_lags_vec, [corr_dict[lag] for lag in autocorr_lags_days], color="black", linewidth=1.8)
    ax.axhline( sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
    ax.axhline(-sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (days)")
    ax.set_title("Global Net Radiation Autocorrelation")
    ax.legend(fontsize=9)
    ax.grid(true, which="major", alpha=0.3)
end

let ax = axs_vec[3]
    ax.plot(autocorr_lags_vec, [enso_globalrad_lagcorr[lag] for lag in autocorr_lags_days], color="black", linewidth=1.8)
    ax.axhline( sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
    ax.axhline(-sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (days)  [positive = R lags ENSO]")
    ax.set_title("Lag Correlation: ENSO vs Global Net Radiation")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.legend(fontsize=9)
    ax.grid(true, which="major", alpha=0.3)
end

fig.suptitle("ENSO & Global Radiation: Autocorrelations and Lag Correlation", fontsize=12)
fig.tight_layout()
fig.savefig(joinpath(visdir, "enso_globalrad_autocorr_lagcorr.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
println("  Saved enso_globalrad_autocorr_lagcorr.png")

# --- Global autocorrelations: global_rad, global_t_1000, global_LTS ---
println("Plotting global autocorrelations...")
global_autocorr_keys = ["global_rad", "global_t_1000", "global_LTS"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for (ax, var_key) in zip(collect(axs), global_autocorr_keys)
    for component in ["total", "enso", "resid"]
        corr_dict = autocorr_results[var_key][component]
        ax.plot(autocorr_lags_vec, [corr_dict[lag] for lag in autocorr_lags_days],
            color=component_colors[component],
            label=component_labels[component],
            linewidth=1.8)
    end
    ax.axhline( sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
    ax.axhline(-sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation / Normalized Covariance")
    ax.set_title(var_titles[var_key])
    ax.legend(fontsize=9)
    ax.grid(true, which="major", alpha=0.3)
end
fig.suptitle("Global Variables: Autocorrelations", fontsize=13)
fig.tight_layout()
fig.savefig(joinpath(visdir, "autocorrelations_global.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
println("  Saved autocorrelations_global.png")

# Lag correlations: global_rad vs global_t_1000 and global_LTS
println("Computing global lag correlations...")
global_lag_corr = Dict{String, Any}()
for temp_key in ["global_t_1000", "global_LTS"]
    global_lag_corr[temp_key] = compute_lag_corr_contributions(decomp, "global_rad", temp_key, lags_days)
end

println("Plotting global lag correlations...")
fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=true)
axs_vec = collect(axs)
for (ax, temp_key) in zip(axs_vec, ["global_t_1000", "global_LTS"])
    for component in ["total", "enso", "resid"]
        corr_dict = global_lag_corr[temp_key][component]
        ax.plot(lags_vec, [corr_dict[lag] for lag in lags_days],
            color=component_colors[component],
            label=component_labels[component],
            linewidth=1.8)
    end
    ax.axhline( sig_lagcorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
    ax.axhline(-sig_lagcorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(temp_xlabels[temp_key])
    ax.set_title("$(var_titles["global_rad"]) vs $(var_titles[temp_key])")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.grid(true, which="major", alpha=0.3)
end
axs_vec[1].set_ylabel("Correlation / Normalized Covariance")
axs_vec[1].legend()
fig.suptitle("Global: Lag Correlations — Global Net Radiation vs Global T₁₀₀₀ & LTS", fontsize=12)
fig.tight_layout()
fig.savefig(joinpath(visdir, "lag_corr_global_rad_vs_global_temps.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
println("  Saved lag_corr_global_rad_vs_global_temps.png")

# --- LTS, T₁₀₀₀, θ₇₀₀: autocorr (top row) + global rad cross-corr (bottom row) ---
println("Plotting LTS, T₁₀₀₀, θ₇₀₀ autocorrelations and global rad cross-correlations...")
lts_region_info = [
    ("global", "global_rad", "global_LTS", "global_t_1000", "global_theta_700"),
    ("SEPac",  "global_rad", "SEPac_LTS",  "SEPac_t_1000",  "SEPac_theta_700"),
    ("NEPac",  "global_rad", "NEPac_LTS",  "NEPac_t_1000",  "NEPac_theta_700"),
]

for (region_name, rad_key, lts_key, t1000_key, theta700_key) in lts_region_info
    println("  $region_name: computing cross-correlations...")
    temp_keys_row = [lts_key, t1000_key, theta700_key]
    xc = Dict(tk => compute_lag_corr_contributions(decomp, rad_key, tk, lags_days) for tk in temp_keys_row)

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    axs_flat = collect(axs.flatten())  # row-major: top-left…top-right, then bot-left…bot-right

    # Top row: autocorrelations
    for (col, tk) in enumerate(temp_keys_row)
        ax = axs_flat[col]
        for component in ["total", "enso", "resid"]
            corr_dict = autocorr_results[tk][component]
            ax.plot(autocorr_lags_vec, [corr_dict[lag] for lag in autocorr_lags_days],
                color=component_colors[component], label=component_labels[component], linewidth=1.8)
        end
        ax.axhline( sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
        ax.axhline(-sig_autocorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Autocorrelation / Normalized Covariance")
        ax.set_title("$(var_titles[tk]) Autocorrelation")
        ax.legend(fontsize=9)
        ax.grid(true, which="major", alpha=0.3)
    end

    # Bottom row: cross-correlations with global rad
    for (col, tk) in enumerate(temp_keys_row)
        ax = axs_flat[col + 3]
        for component in ["total", "enso", "resid"]
            corr_dict = xc[tk][component]
            ax.plot(lags_vec, [corr_dict[lag] for lag in lags_days],
                color=component_colors[component], label=component_labels[component], linewidth=1.8)
        end
        ax.axhline( sig_lagcorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="95% significance")
        ax.axhline(-sig_lagcorr, color="dimgray", linewidth=1.0, linestyle=":", alpha=0.7, label="_nolegend_")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Lag (days)  [positive = R lags $(var_titles[tk])]")
        ax.set_ylabel("Correlation / Normalized Covariance")
        ax.set_title("Global Rad vs $(var_titles[tk]) Lag Correlation")
        ax.legend(fontsize=9)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax.grid(true, which="major", alpha=0.3)
    end

    fig.suptitle("$region_name: Autocorrelations & Global Rad Cross-Correlations", fontsize=13)
    fig.tight_layout()
    fig.savefig(joinpath(visdir, "lts_decomp_$(region_name).png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    println("  Saved lts_decomp_$(region_name).png")
end

println("All plots saved to $visdir")
