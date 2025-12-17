"""
This script connects theta_1000 to ocean and atmosphere temperature components
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")

using CSV, DataFrames, Dates, Dictionaries, PythonCall
@py import matplotlib.pyplot as plt, cartopy.crs as ccrs, matplotlib.colors as colors, cmasher as cmr

base_vis_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/patrizio_decomp"

!isdir(base_vis_dir) && mkpath(base_vis_dir)

analysis_period = (Date(0), Date(50000))

#Load in the sepac time series, and the global radiation time series, for the full length of their respective time periods. Deseasonalize and detrend them twice.

patrizio_vars = ["SST", "T_ocn", "T_atm"]

# Create base save directory for figures
mkpath(base_vis_dir)

function skipmissing_corr(x,y)
    valid_data = .!ismissing.(x) .& .!ismissing.(y)
    if sum(valid_data) > 0
        return cor(x[valid_data], y[valid_data])
    else
        return NaN
    end
end

# Load in the Patrizio SST partition gridded data - this is the only gridded data we need
patrizio_data, patrizio_coords = load_patrizio_sst_data(patrizio_vars, analysis_period)
patrizio_lat = patrizio_coords["latitude"]
patrizio_lon = patrizio_coords["longitude"]
patrizio_time = round.(patrizio_coords["time"], Month(1), RoundDown)
patrizio_time_valid = in_time_period.(patrizio_time, Ref(analysis_period))
patrizio_time = patrizio_time[patrizio_time_valid]
patrizio_float_time = calc_float_time.(patrizio_time)
patrizio_precalculated_month_groups = groupfind(month, patrizio_time)

kuroshio_lats = (36, 45)
kuroshio_lons = (140, 170)

lon_mask = (patrizio_lon .>= kuroshio_lons[1]) .& (patrizio_lon .<= kuroshio_lons[2])
lat_mask = (patrizio_lat .>= kuroshio_lats[1]) .& (patrizio_lat .<= kuroshio_lats[2])

notmissing_mask = (slice -> !(any(v -> ismissing(v) || isnan(v), slice))).(eachslice(patrizio_data["T_ocn"]; dims = (1,2)))

kuroshio_mask = lon_mask .& lat_mask' .& notmissing_mask

sst_avg = generate_spatial_mean(patrizio_data["SST"], patrizio_lat, kuroshio_mask)
tocn_avg = generate_spatial_mean(patrizio_data["T_ocn"], patrizio_lat, kuroshio_mask)
tatm_avg = generate_spatial_mean(patrizio_data["T_atm"], patrizio_lat, kuroshio_mask)

# Calculate sum
tatm_tocn_sum = tatm_avg .+ tocn_avg

# Create 3-pane comparison plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# First panel: SST (faint) vs T_ocean (bold)
ax1.plot(patrizio_float_time, sst_avg, label="SST", linewidth=1, alpha=0.4, color="blue")
ax1.plot(patrizio_float_time, tocn_avg, label="T_ocean", linewidth=3, color="darkblue")
ax1.set_ylabel("Temperature Anomaly")
ax1.set_title("SST vs T_ocean")
ax1.legend()
ax1.grid(true, alpha=0.3)

# Second panel: SST (faint) vs T_atm (bold)
ax2.plot(patrizio_float_time, sst_avg, label="SST", linewidth=1, alpha=0.4, color="blue")
ax2.plot(patrizio_float_time, tatm_avg, label="T_atm", linewidth=3, color="red")
ax2.set_ylabel("Temperature Anomaly")
ax2.set_title("SST vs T_atm")
ax2.legend()
ax2.grid(true, alpha=0.3)

# Third panel: SST (faint) vs (T_atm + T_ocean) (bold)
ax3.plot(patrizio_float_time, sst_avg, label="SST", linewidth=1, alpha=0.4, color="blue")
ax3.plot(patrizio_float_time, tatm_tocn_sum, label="T_atm + T_ocean", linewidth=3, color="purple")
ax3.set_xlabel("Time")
ax3.set_ylabel("Temperature Anomaly")
ax3.set_title("SST vs (T_atm + T_ocean)")
ax3.legend()
ax3.grid(true, alpha=0.3)

plt.tight_layout()
plt.savefig(joinpath(base_vis_dir, "kuroshio_temperature_3pane_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

