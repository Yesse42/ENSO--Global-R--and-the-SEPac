using JLD2, Statistics, StatsBase, Dates, SplitApplyCombine, CSV, DataFrames

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")
include("../utils/plot_global.jl")

"""
This script calculates a mask based on lat/lon, land_sea_mask, and the PLS coefficients.
It then uses this mask to calculate a spatially weighted average SST index over the SE Pacific region.
"""

# Set up output directory
visdir = "../../vis/sepac_radiation_effects/idx_mask"
if !isdir(visdir)
    mkpath(visdir)
end

savedir = "../../data/SEPac_SST/"

vis = true

idx_time_period = (Date(0), Date(10000000, 12, 31))

PLSdir = "../../data/PLSs/"

#Load in the basic PLS
println("Loading PLS models...")
atmospheric_pls_data = jldopen(joinpath(PLSdir, "atmospheric_ceres_net_pls.jld2"), "r")
atmospheric_pls = atmospheric_pls_data["pls_model"]
atmospheric_idxs = atmospheric_pls_data["predictor_indices"]
atmospheric_shapes = atmospheric_pls_data["predictor_shapes"]

coords = atmospheric_pls_data["coordinates"]
lat = coords["latitude"]
lon = coords["longitude"]

#Now load in the ERA5 data for SST and the CERES data for net global radiation
relevant_var = "sst"
era5data, era5coords = load_era5_data([relevant_var], idx_time_period)
sst_data = era5data[relevant_var]

ceres_var = "gtoa_net_all_mon"
ceres_rad, ceres_coords = load_ceres_data([ceres_var], idx_time_period)
ceres_rad = ceres_rad[ceres_var]
ceres_time = ceres_coords["time"]
ceres_time = floor.(ceres_time, Month(1))
ceres_valid_time = in_time_period.(ceres_time, Ref(time_period))

ceres_rad = ceres_rad[ceres_valid_time]
ceres_time = ceres_time[ceres_valid_time]

era5_time_mask = in_time_period.(era5coords["time"], Ref(time_period))
sst = sst_data[:, :, era5_time_mask]
era5_times = era5coords["time"][era5_time_mask]

#Now detrend and deseasonalize the SST data and rad data
println("Detrending and deseasonalizing data...")

# Prepare time information for detrending and deseasonalizing
months = month.(era5_times)
float_times = calc_float_time.(era5_times)
month_groups = groupfind(months)

# Detrend and deseasonalize SST data
for i in 1:size(sst, 1), j in 1:size(sst, 2)
    slice = view(sst, i, j, :)
    if !any(ismissing, slice)
        detrend_and_deseasonalize_precalculated_groups!(slice, float_times, month_groups)
    end
end

# Detrend and deseasonalize radiation data
detrend_and_deseasonalize_precalculated_groups!(ceres_rad, float_times, month_groups)

#Now calculate the correlation between SST and global radiation
temp_coeff_arr = cor.(eachslice(sst, dims = (1,2,)), Ref(ceres_rad))
temp_coeff_arr[ismissing.(temp_coeff_arr)] .= 0.0  # Replace missing values with 0.0

fig_temp_coeff = plot_global_heatmap(lat, lon, Float64.(temp_coeff_arr); title = "$relevant_var-Global Radiation Correlation", colorbar_label = "Correlation")
fig_temp_coeff.savefig(joinpath(visdir, "$(relevant_var)_global_radiation_correlation.png"))
vis && display(fig_temp_coeff)
plt.close(fig_temp_coeff)

thresh = 0.15
temp_coeff_mask = Float64.(temp_coeff_arr .> thresh)
fig_coeff_mask = plot_global_heatmap(lat, lon, temp_coeff_mask; title = "PLS Coefficient Mask for $relevant_var", colorbar_label = "Mask")
fig_coeff_mask.savefig(joinpath(visdir, "pls_coefficient_mask_$relevant_var.png"))
vis && display(fig_coeff_mask)
plt.close(fig_coeff_mask)

era_vars = ["sst", "lsm"]
era5_data, era5_coords = load_era5_data(era_vars, idx_time_period)

lsm = Float64.(era5_data["lsm"][:,:,1] .== 0)
sst = era5_data["sst"]

fig_lsm = plot_global_heatmap(lat, lon, lsm; title = "Land-Sea Mask", colorbar_label = "LSM")
fig_lsm.savefig(joinpath(visdir, "land_sea_mask.png"))
vis && display(fig_lsm)
plt.close(fig_lsm)

lonbounds = (-130, -69.3) .+ 360
latbounds = (-40, 0)

lonmask = (lon .>= lonbounds[1]) .& (lon .<= lonbounds[2])
latmask = (lat .>= latbounds[1]) .& (lat .<= latbounds[2])
region_mask = lonmask .* latmask'

fig_region = plot_global_heatmap(lat, lon, region_mask; title = "Region Mask", colorbar_label = "Mask")
fig_region.savefig(joinpath(visdir, "region_mask.png"))
vis && display(fig_region)
plt.close(fig_region)

sst_has_missing_data_mask = Float64.(mapslices(slice -> !any(ismissing, slice), sst; dims = (3)))[:,:]

fig_missing = plot_global_heatmap(lat, lon, sst_has_missing_data_mask; title = "SST Missing Data Mask", colorbar_label = "Mask")
fig_missing.savefig(joinpath(visdir, "sst_missing_data_mask.png"))
vis && display(fig_missing)
plt.close(fig_missing)

final_mask = temp_coeff_mask .* lsm .* region_mask .* sst_has_missing_data_mask
bool_mask = final_mask .== 1

fig = plot_global_heatmap(lat, lon, final_mask; title = "Final SEPac SST Mask", colorbar_label = "Mask")
fig.savefig(joinpath(visdir, "final_sepac_sst_mask.png"))
vis && display(fig)
plt.close(fig)

coslat = cosd.(lat)'
area_weights = coslat .* bool_mask
total_weights = sum(area_weights)

sst[ismissing.(sst)] .= 0.0

epac_sst_idx = vec(sum(sst .* area_weights, dims = (1,2)) ./ total_weights)
epac_dates = era5_coords["time"]

df = DataFrame(Date = epac_dates,)
fig_idx = plot(df.Date, epac_sst_idx, label = "SEPac SST Index", xlabel = "Date", ylabel = "Index", title = "SEPac SST Index Time Series")
savefig(fig_idx, joinpath(visdir, "sepac_sst_index_timeseries.png"))
vis && display(fig_idx)

time_lags = -24:24  # in months
for lag in time_lags
    df[!, Symbol("SEPac_SST_Index_Lag$(lag)")] = time_lag(epac_sst_idx, lag)
end

display(df)

CSV.write(joinpath(savedir, "sepac_sst_index.csv"), df)

# Save mask, weights, and coordinates to JLD2 file
jldsave(joinpath(savedir, "sepac_sst_mask_and_weights.jld2");
    final_mask = Float32.(final_mask),
    bool_mask = Float32.(bool_mask),
    area_weights = Float32.(area_weights),
    latitude = Float32.(lat),
    longitude = Float32.(lon)
)