"""
This script aims to calculate the correlation between global net r and local t2m, as well as regress global r onto local t2m to justify the mask for the sepac sst index. It then plots these globally
"""

using JLD2, Statistics, StatsBase, Dates, SplitApplyCombine, CSV, DataFrames

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")
include("../utils/plot_global.jl")

# Set up output directory
visdir = "../../vis/draft_v1_vis"
if !isdir(visdir)
    mkpath(visdir)
end

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
relevant_var = "t2m"
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

temp_regress_arr = get_lsq_slope.(eachslice(sst, dims = (1,2,)), Ref(ceres_rad))

lonbounds = (-130, -69.3) .+ 360
latbounds = (-40, 0)

lonmask = (lon .>= lonbounds[1]) .& (lon .<= lonbounds[2])
latmask = (lat .>= latbounds[1]) .& (lat .<= latbounds[2])
region_mask = lonmask .* latmask'

temp_coeffs_masked = temp_coeff_arr .* region_mask

corr_contours = (0.1, 0.15, 0.2, 0.25)

# Plot 1: Regression slopes (W/m²/K)
println("Creating regression slopes plot...")
temp_regress_arr[ismissing.(temp_regress_arr)] .= 0.0  # Replace missing values with 0.0

if vis
    fig1 = plot_global_heatmap(lat, lon, temp_regress_arr; 
                              title="Regression Slopes: Global Net Radiation vs Local SST", 
                              colorbar_label="Regression Slope (W/m²/K)",
                              cmap=@py(plt.cm.RdBu_r))
    
    plt.savefig(joinpath(visdir, "regression_slopes_global_r_vs_sst.png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)
end

# Plot 2: Correlations with contours
println("Creating correlations plot with contours...")

if vis
    # Create the correlation plot using plot_global_heatmap
    fig2, ax2 = plt.subplots(subplot_kw=Dict("projection"=>ccrs.Robinson(central_longitude=180)))
    ax2.set_global()
    ax2.coastlines()
    ax2.set_title("Correlations: Global Net Radiation vs Local SST")

    # Set up color normalization
    absmax = max(abs(minimum(temp_coeff_arr)), abs(maximum(temp_coeff_arr)))
    colornorm = @py(plt.cm.colors.Normalize(vmin=-absmax, vmax=absmax))
    
    # Plot the correlation data as contourf
    c = ax2.contourf(lon, lat, temp_coeff_arr'; 
                     transform=ccrs.PlateCarree(), 
                     cmap=@py(plt.cm.RdBu_r), 
                     levels=21, 
                     norm=colornorm)
    
    # Add colorbar
    plt.colorbar(c, ax=ax2, orientation="horizontal", pad=0.05, label="Correlation Coefficient")
    
    # Add contour lines for the masked data
    contour_lines = ax2.contour(lon, lat, temp_coeffs_masked'; 
                               transform=ccrs.PlateCarree(), 
                               levels=collect(corr_contours), 
                               colors="black", 
                               linewidths=0.8,
                               linestyles="solid",
                               alpha=0.5)
    
    # Add contour labels
    labels = ax2.clabel(contour_lines, inline=true, fontsize=6, fmt="%.2f")
    # Set alpha for the text labels
    for label in labels
        label.set_alpha(0.5)
    end
    
    plt.savefig(joinpath(visdir, "correlations_global_r_vs_sst_with_contours.png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)
end

println("Plots saved to: ", visdir)



