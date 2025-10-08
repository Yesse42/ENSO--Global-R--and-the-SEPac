"""
In this script I generate a spread of plots exploring the relationship between LTS and t2m in the SEPac and global radiation, to be used for the graduate student showcase
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")

using CSV, DataFrames, Dates, Dictionaries, PythonCall
@py import matplotlib.pyplot as plt, cartopy.crs as ccrs, matplotlib.colors as colors, cmasher as cmr

# Create base save directory for figures
base_save_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/gridded_effects"
mkpath(base_save_dir)

function skipmissing_corr(x,y)
    valid_data = .!ismissing.(x) .& .!ismissing.(y)
    if sum(valid_data) > 0
        return cor(x[valid_data], y[valid_data])
    else
        return NaN
    end
end

function generate_correlations_single_level(grid, lagged_time_series_dict)
    corr_values = [calculate_corrfunc_grid(grid, ts; corrfunc = skipmissing_corr) for ts in lagged_time_series_dict]
end

analysis_bounds = (Date(2000, 3), Date(2024, 3, 31))

# Load in the time series data for the southeast pacific variables
region = "SEPac_feedback_definition"

# Load in the local time series data for SEPac
local_ts_dir = "../../data/sepac_lts_data/local_region_time_series"
era5_local_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
ceres_local_df = CSV.read(joinpath(local_ts_dir, "ceres_region_avg_lagged_$(region).csv"), DataFrame)

ceres_global_df = CSV.read("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/lagged/global_ceres_lagged_detrended_deseasonalized.csv", DataFrame)

# Load in the new gridded ceres data using load_funcs.jl
cre_names = []
toa_rad_names = "toa_" .* ["net_all", "net_lw", "net_sw"] .* "_mon"
ceres_varnames = vcat(cre_names, toa_rad_names)

global_toa_rad_names = "g" .* toa_rad_names

ceres_data, ceres_coords = load_new_ceres_data(ceres_varnames, analysis_bounds)
ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]
ceres_time = round.(ceres_coords["time"], Month(1), RoundDown)
ceres_time_valid = in_time_period.(ceres_time, Ref(analysis_bounds))
ceres_time = ceres_time[ceres_time_valid]
ceres_float_time = calc_float_time.(ceres_time)
ceres_precalculated_month_groups = groupfind(month, ceres_time)

for var in ceres_varnames
    ceres_data[var] = ceres_data[var][:, :, ceres_time_valid]
    detrend_and_deseasonalize_precalculated_groups!.(eachslice(ceres_data[var]; dims = (1,2)), Ref(ceres_float_time),Ref(ceres_precalculated_month_groups))
end

# Load in the ERA5 data using load_funcs.jl
single_level_vars = []
pressure_level_vars = ["t"]
era5_vars = vcat(single_level_vars, pressure_level_vars)
era5_data, era5_coords = load_era5_data(era5_vars, analysis_bounds, pressure_level_file = "new_pressure_levels.nc")
era5_lat = era5_coords["latitude"]
era5_lon = era5_coords["longitude"]
era5_time = round.(era5_coords["time"], Month(1), RoundDown)
era5_press_time = round.(era5_coords["pressure_time"], Month(1), RoundDown)
era5_time = era5_time[in_time_period.(era5_time, Ref(analysis_bounds))]

#Restrict all era5 vars to the analysis time period
for all_var in pressure_level_vars
    era5_data[all_var] = era5_data[all_var][:,:,:,in_time_period.(era5_press_time, Ref(analysis_bounds))]

    detrend_and_deseasonalize_precalculated_groups!.(eachslice(era5_data[all_var]; dims = (1,2,3)), Ref(ceres_float_time),Ref(ceres_precalculated_month_groups))
end
for all_var in single_level_vars
    era5_data[all_var] = era5_data[all_var][:,:,in_time_period.(era5_time, Ref(analysis_bounds))]

    detrend_and_deseasonalize_precalculated_groups!.(eachslice(era5_data[all_var]; dims = (1,2)), Ref(ceres_float_time),Ref(ceres_precalculated_month_groups))
end


# Bring all data into the same analysis bounds time period, after rounding the times down to the nearest month
era5_local_df[!, :date] = Date.(era5_local_df[!, :date])
ceres_local_df[!, :date] = Date.(ceres_local_df[!, :date])
ceres_global_df[!, :date] = Date.(ceres_global_df[!, :date])

# Filter to analysis bounds
era5_local_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], era5_local_df)
ceres_local_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], ceres_local_df)
ceres_global_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], ceres_global_df)

local_df = DataFrames.innerjoin(era5_local_df, ceres_local_df, ceres_global_df, on = :date)

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/LTS_plot_spread"

#Validate that all has gone well with a quick plot of the correlation between gridded net rad and global net rad
println("Generating Sanity Check Plot...")
check_timeseries = local_df[:, :gtoa_net_all_mon_lag_0]
check_corrs = cor.(eachslice(ceres_data["toa_net_all_mon"]; dims=(1,2)), Ref(check_timeseries))[:,:,1]

plot = plot_global_heatmap(ceres_lat, ceres_lon, check_corrs; title="Correlation between SEPac Gridded Net Radiation and SEPac Averaged Global Net Radiation", colorbar_label="Correlation Coefficient")
plt.savefig(joinpath(visdir, "sanity_check_ceres_global_gridded_corr.png"), dpi=300)