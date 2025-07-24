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

vis = false

idx_time_period = (Date(1980), Date(2024, 12, 31))

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

relevant_var = "t2m"

coeff_matrix = make_matrix_to_multiply_by_X_to_get_Y(atmospheric_pls; components = 1:5)
coeff_dict = reconstruct_spatial_arrays(coeff_matrix, atmospheric_idxs, atmospheric_shapes)
temp_coeff_arr = coeff_dict[relevant_var]

fig_coeff = plot_global_heatmap(lat, lon, temp_coeff_arr; title = "PLS Coefficients for $relevant_var", colorbar_label = "Coefficient")
fig_coeff.savefig(joinpath(visdir, "pls_coefficients_$relevant_var.png"))
vis && display(fig_coeff)
plt.close(fig_coeff)

thresh = 3e-6
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

time_lags = -12:12
for lag in time_lags
    df[!, Symbol("SEPac_SST_Index_Lag$(lag)")] = time_lag(epac_sst_idx, lag)
end

display(df)

CSV.write(joinpath(savedir, "sepac_sst_index.csv"), df)