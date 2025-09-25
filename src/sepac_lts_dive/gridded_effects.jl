"""
This script is meant to calculate the basic gridded effects of the following variables in the SEPac SST idx on w, u, v, t, z, toa sw, toa lw, and toa net. LTS, Omega, wind speed, sw, lw, and net radiation for the sepac, at a variety of lags
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

function generate_correlations_levels(grids, lagged_time_series_dict)
    corr_values = [calculate_corrfunc_grid(grid, ts; corrfunc = skipmissing_corr) for grid in grids, ts in lagged_time_series_dict]
end

analysis_bounds = (Date(2000, 3), Date(2024, 3, 31))

# Load in the time series data for the southeast pacific variables
region = "SEPac_feedback_definition"

# Load in the local time series data for SEPac
local_ts_dir = "../../data/sepac_lts_data/local_region_time_series"
era5_local_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
ceres_local_df = CSV.read(joinpath(local_ts_dir, "ceres_region_avg_lagged_$(region).csv"), DataFrame)

ceres_global_df = CSV.read("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/lagged/global_ceres_lagged_detrended_deseasonalized.csv", DataFrame)

# Load in the nonlocal radiation time series from CERES
nonlocal_ts_dir = "../../data/sepac_lts_data/nonlocal_radiation_time_series"
nonlocal_rad_df = CSV.read(joinpath(nonlocal_ts_dir, "$(region)_nonlocal_radiation.csv"), DataFrame)

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
single_level_vars = ["t2m", "msl", "u10", "v10"]
pressure_level_vars = ["t", "w", "z", "u", "v"]
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
nonlocal_rad_df[!, :date] = Date.(nonlocal_rad_df[!, :date])
ceres_global_df[!, :date] = Date.(ceres_global_df[!, :date])

# Filter to analysis bounds
era5_local_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], era5_local_df)
ceres_local_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], ceres_local_df)
nonlocal_rad_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], nonlocal_rad_df)
ceres_global_df = filter(row -> analysis_bounds[1] <= row.date <= analysis_bounds[2], ceres_global_df)

lags_of_interest = [-6, -3, 0, 3, 6]

local_df = DataFrames.innerjoin(era5_local_df, ceres_local_df, nonlocal_rad_df, ceres_global_df, on = :date)

single_level_grids = [ceres_data[var] for var in ceres_varnames]

time_series_correlates = ["toa_net_all_mon", "toa_net_lw_mon", "toa_net_sw_mon", "LTS_1000", "θ_1000", "θ_700", "ω_500", "sfc_wind_speed"]
append!(time_series_correlates, global_toa_rad_names)

connected_single_level_var = ["t2m", "msl", "u10", "v10"]
connected_pressure_level_var = ["t", "z", "u", "v"]

level_var_names = ["t", "press_geopotential", "u", "v"]
press_level_names = string.(era5_coords["pressure_level"]) .* " hPa"
press_sfc_joint_names = vcat(["sfc"], press_level_names)

alone_pressure_level_var = setdiff(pressure_level_vars, connected_pressure_level_var)

array_of_level_data = vec([vcat([era5_data[sl_var]], vec(collect(eachslice(era5_data[pl_var]; dims = (3,))))) for (sl_var, pl_var) in zip(connected_single_level_var, connected_pressure_level_var)])
append!(array_of_level_data, vec([vec(collect(eachslice(era5_data[pl_var]; dims = (3,)))) for pl_var in alone_pressure_level_var]))

level_var_names = vcat(level_var_names, alone_pressure_level_var)

# First, make maps showing the connection between the vars and the surface vars at different lags
for var in time_series_correlates
    if var[1] ≠ 'g'
        regionalized_name = region * " " * var
    else
        regionalized_name = var
    end
    if any(occursin.(var, names(local_df)))
        # Create specific save directory for this variable
        var_save_dir = joinpath(base_save_dir, regionalized_name)
        mkpath(var_save_dir)
        
        # Create lagged time series dictionary for different lags of interest using Dictionaries.jl
        lagged_ts_dict = Dictionary()
        for lag in lags_of_interest
            lag_col_name = "$(var)_lag_$(lag)"
            if lag_col_name in names(local_df)
                set!(lagged_ts_dict, lag, local_df[!, lag_col_name])
            end
        end
        
        # Generate correlations for single level variables (CERES TOA radiation)
        for (i, grid_data) in enumerate(single_level_grids)
            var_name = ceres_varnames[i]
            println("Processing correlations for $var with $var_name")
            
            # Generate correlation maps for different lags
            corr_values = [calculate_corrfunc_grid(grid_data, ts; corrfunc = skipmissing_corr) for ts in lagged_ts_dict]
            
            # Create subtitles for each lag
            subtitles = ["$(var) Lag $(lag)" for lag in keys(lagged_ts_dict)]
            plot_layout = (1, length(corr_values))
            
            # Plot using the global plotting function
            fig = plot_multiple_levels(ceres_lat, ceres_lon, corr_values, plot_layout; 
                                      subtitles=subtitles, 
                                      colorbar_label="Correlation with $var_name",
                                      )

            fig.suptitle("Corr between $regionalized_name and $var_name")
            
            # Save figure in variable-specific directory
            filename = "$(regionalized_name)_vs_$(var_name)_correlations.png"
            filepath = joinpath(var_save_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            println("Saved figure to: $filepath")
            plt.close(fig)
        end
    end
end

# Then connect the local vars and the data on all the levels
for var in time_series_correlates
    if var[1] ≠ 'g'
        regionalized_name = region * " " * var
    else
        regionalized_name = var
    end
    if any(occursin.(var, names(local_df)))
        # Create specific save directory for this variable
        var_save_dir = joinpath(base_save_dir, var)
        mkpath(var_save_dir)
        
        # Create lagged time series dictionary for different lags of interest using Dictionaries.jl
        lagged_ts_dict = Dictionary()
        for lag in lags_of_interest
            lag_col_name = "$(var)_lag_$(lag)"
            if lag_col_name in names(local_df)
                set!(lagged_ts_dict, lag, local_df[!, lag_col_name])
            end
        end
        
        # Process multi-level ERA5 data
        for (level_idx, level_grids) in enumerate(array_of_level_data)
            level_var_name = level_var_names[level_idx] 
            println("Processing multi-level correlations for $var with $level_var_name")
            
            n_levels = length(level_grids)
            n_lags = length(lagged_ts_dict)

            out_corr_arr = Array{Any}(undef, n_levels, n_lags)
            plot_labels = String[]

            grid_level_names = nothing
            if n_levels == length(press_sfc_joint_names)
                grid_level_names = press_sfc_joint_names
            elseif n_levels == length(press_level_names)
                grid_level_names = press_level_names
            end

            for level in 1:n_levels
                for (lag_idx, (lag_val, lag_ts)) in enumerate(pairs(lagged_ts_dict))
                    corr_value = calculate_corrfunc_grid(level_grids[level], lag_ts; corrfunc = skipmissing_corr)
                    out_corr_arr[level, lag_idx] = corr_value
                    push!(plot_labels, "$(grid_level_names[level]) Lag $(lag_val)")
                end
            end

            out_corr_arr = vec(out_corr_arr)
            #Generate the plot
            fig = plot_multiple_levels(era5_lat, era5_lon, out_corr_arr, (n_levels, n_lags); 
                                      subtitles=plot_labels, 
                                      colorbar_label="Correlation with $level_var_name",
                                      )

            fig.suptitle("Corr between $var and $level_var_name")
            # Save figure in variable-specific directory
            filename = "$(regionalized_name)_vs_$(level_var_name)_correlations.png"
            filepath = joinpath(var_save_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            println("Saved figure to: $filepath")
            plt.close(fig)
        end
    end
end

#Now perform a more interesting analysis; perform the same decomposition as in src/sepac_lts_dive/compare_surface_aloft.jl but decomposing the relationship between sepac lts and global radiation into the relationship between sepac lts and global net radiation. Further decompose the local relationship between net rad and sepac lts into the relationship between sw, lw, theta_1000, theta_700, and local net rad.

gridded_decomp_visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/lts_global_rad/gridded_effects/gridded_decomposition"

#First thing to do is calculate the correlation between sepac lts and global net radiation.
net_rad = ceres_global_df[!, "gtoa_net_all_mon_lag_0"]
lts_1000 = local_df[!, "LTS_1000_lag_0"]

total_corr = skipmissing_corr(lts_1000, net_rad)
println("Total correlation between SEPac LTS and global net radiation: $total_corr")

area_weighted_net_rad = ceres_data["toa_net_all_mon"] .* cosd.(ceres_lat')
#Now calculate the gridded cor
gridded_net_rad_corr = calculate_corrfunc_grid(area_weighted_net_rad, lts_1000; corrfunc = skipmissing_corr)

#And calculate the gridded weights
net_rad_from_sum = vec(sum(area_weighted_net_rad; dims = (1,2)))
gridded_net_rad_std = mapslices(std, area_weighted_net_rad; dims = 3)[:,:, 1]
net_corr_sum_weights = gridded_net_rad_std ./ std(net_rad_from_sum)

weighted_corr = gridded_net_rad_corr .* net_corr_sum_weights

#Verify that the weighted correlations have a relative error of 1e-4 or less compared to the total correlation
if !isapprox(sum(skipmissing(vec(weighted_corr))), total_corr; rtol=1e-2)
    @warn "Weighted correlation does not match total correlation within tolerance!"
end
#Now make a 3 pane plot showing the total corr in the title, the raw net corr, the weights, and then the weighted corr from left to right
fig, axs = plt.subplots(1, 3, 
                       figsize=(18, 4),
                       subplot_kw=Dict("projection" => ccrs.PlateCarree()),
                       layout="compressed")

plot_data = [gridded_net_rad_corr, net_corr_sum_weights, weighted_corr]
plot_titles = ["Raw Corr", "Weights", "Weighted Corr"]

for (i, (data, title)) in enumerate(zip(plot_data, plot_titles))
    i -= 1 # Adjust for 0-based indexing in Python
    ax = axs[i]
    ax.set_global()
    ax.coastlines()
    
    # Calculate color normalization for each plot individually
    absmax = max(abs(minimum(data)), abs(maximum(data)))
    colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    
    # Plot the data
    c = ax.contourf(ceres_lon, ceres_lat, data', 
                  transform=ccrs.PlateCarree(), 
                  cmap=cmr.prinsenvlag.reversed(), 
                  levels=21, 
                  norm=colornorm)
    
    # Add individual colorbar for this subplot
    plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, 
               label=title)
end
fig.suptitle("Decomposition of SEPac LTS and Global Net Radiation Correlation: Total Corr = $(round(total_corr, digits=3))")
#Save the figure
mkpath(gridded_decomp_visdir)
fig.savefig(joinpath(gridded_decomp_visdir, "sepac_lts_global_net_rad_decomposition.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

#Now we go one step further, and decompose the weighted local correlation between sepac lts and local net radiation into the weighted local correlations between sepac lts and local sw, lw, theta_1000, and theta_700
gridded_sw = ceres_data["toa_net_sw_mon"] .* cosd.(ceres_lat')
gridded_lw = ceres_data["toa_net_lw_mon"] .* cosd.(ceres_lat')

gridded_sw_std = mapslices(std, gridded_sw; dims = 3)
gridded_lw_std = mapslices(std, gridded_lw; dims = 3)

local_neg_theta_1000 = -1 .* local_df[!, "θ_1000_lag_0"]
local_theta_700 = local_df[!, "θ_700_lag_0"]

neg_theta_1000_std = std(local_neg_theta_1000)
theta_700_std = std(local_theta_700)
LTS_std = std(lts_1000)

#Now make the correlation maps and the associated weights
sw_theta_1000_corr = calculate_corrfunc_grid(gridded_sw, local_neg_theta_1000; corrfunc = skipmissing_corr)
sw_theta_1000_weights = @. gridded_sw_std * neg_theta_1000_std / (LTS_std * gridded_net_rad_std) * net_corr_sum_weights

sw_theta_700_corr = calculate_corrfunc_grid(gridded_sw, local_theta_700; corrfunc = skipmissing_corr)
sw_theta_700_weights = @. gridded_sw_std * theta_700_std / (LTS_std * gridded_net_rad_std) * net_corr_sum_weights

lw_theta_1000_corr = calculate_corrfunc_grid(gridded_lw, local_neg_theta_1000; corrfunc = skipmissing_corr)
lw_theta_1000_weights = @. gridded_lw_std * neg_theta_1000_std / (LTS_std * gridded_net_rad_std) * net_corr_sum_weights

lw_theta_700_corr = calculate_corrfunc_grid(gridded_lw, local_theta_700; corrfunc = skipmissing_corr)
lw_theta_700_weights = @. gridded_lw_std * theta_700_std / (LTS_std * gridded_net_rad_std) * net_corr_sum_weights

# Create the 5-panel plot showing the decomposition
decomp_corrs = [weighted_corr, sw_theta_1000_corr .* sw_theta_1000_weights, sw_theta_700_corr .* sw_theta_700_weights, 
               lw_theta_1000_corr .* lw_theta_1000_weights, lw_theta_700_corr .* lw_theta_700_weights]
decomp_corrs = [if ndims(comp) == 2 comp else dropdims(comp;dims=3) end for comp in decomp_corrs]

decomp_subtitles = ["Total Weighted Corr", "SW × θ₁₀₀₀ Component", "SW × θ₇₀₀ Component", 
                   "LW × θ₁₀₀₀ Component", "LW × θ₇₀₀ Component"]

#Divide each component by coslat to undo the area weighting for visualization purposes
decomp_corrs = [comp ./ cosd.(ceres_lat') for comp in decomp_corrs]

fig = plot_multiple_levels(ceres_lat, ceres_lon, decomp_corrs, (1, 5); 
                          subtitles=decomp_subtitles, 
                          colorbar_label="Weighted Correlation Components",
                          proj = ccrs.Sinusoidal(;central_longitude = -160))

fig.suptitle("SEPac LTS - Global Net Radiation Decomposition into SW/LW and θ Components")

# Save the decomposition figure
fig.savefig(joinpath(gridded_decomp_visdir, "sepac_lts_radiation_component_decomposition.png"), dpi=300, bbox_inches="tight")
plt.close(fig)





