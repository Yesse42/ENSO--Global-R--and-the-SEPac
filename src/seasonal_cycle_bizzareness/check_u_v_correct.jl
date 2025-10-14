"""
Investigate if u and v have been accidentally swapped somewhere
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

time_period = (Date(2000, 3), Date(2024, 3, 31))
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/why_ceres_why"

ceres_global_rad, grad_times = load_new_ceres_data(["gtoa_net_all_mon"], time_period)
ceres_global_rad = ceres_global_rad["gtoa_net_all_mon"]
grad_times = round.(grad_times["time"], Month(1), RoundDown)

era5_gridded_u, era5_coords = load_era5_data(["u"], time_period; pressure_level_file = "new_pressure_levels.nc")
lat = era5_coords["latitude"]
lon = era5_coords["longitude"]
plev = era5_coords["pressure_level"]
era_time = era5_coords["time"]
era5_gridded_u = era5_gridded_u["u"]

hpa_1000_idx = findfirst(plev .== 1000)

common_times = sort!(intersect(grad_times, era_time))

era_valid_times = findall(t -> t in common_times, era_time)
grad_valid_times = findall(t -> t in common_times, grad_times)

ceres_global_rad = ceres_global_rad[grad_valid_times]
era5_gridded_u = era5_gridded_u[:, :, hpa_1000_idx, era_valid_times]

# Deseasonalize and detrend both datasets
month_groups = get_seasonal_cycle(month.(common_times))
float_times = calc_float_time.(common_times)
deseasonalize_and_detrend_precalculated_groups_twice!(ceres_global_rad, float_times, month_groups)
deseasonalize_and_detrend_precalculated_groups_twice!.(eachslice(era5_gridded_u, dims = (1, 2)), Ref(float_times), Ref(month_groups))

corr_1000_hpa = cor.(eachslice(era5_gridded_u, dims = (1, 2)), Ref(ceres_global_rad))

#Now plot
fig = plot_global_heatmap(lat, lon, corr_1000_hpa, title = "Correlation between 1000 hPa u and CERES Global Rad", colorbar_label = "Correlation Coefficient")
fig.savefig(joinpath(visdir, "corr_1000_hpa_u_ceres.png"), dpi = 300)
plt.close()
