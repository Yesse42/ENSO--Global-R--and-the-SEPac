"""
This script is meant to calculate the basic gridded effects of the following variables in the SEPac SST idx on w, u, v, t, z, toa sw, toa lw, and toa net. LTS, Omega, wind speed, sw, lw, and net radiation for the sepac, at a variety of lags
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")
include("../pls_regressor/pls_functions.jl")

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

calc_theta(T, P) = T .* (1000 ./ P).^(2/7)

analysis_bounds = (Date(2000, 3), Date(2023, 2, 28))

# Load in the time series data for the southeast pacific variables
region = "SEPac_feedback_definition"

# Load in the local time series data for SEPac
local_ts_dir = "../../data/sepac_lts_data/local_region_time_series"
era5_local_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
ceres_local_df = CSV.read(joinpath(local_ts_dir, "ceres_region_avg_lagged_$(region).csv"), DataFrame)

# Load in the ENSO-removed local time series data
era5_enso_removed_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_enso_removed_$(region).csv"), DataFrame)

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
    deseasonalize_and_detrend_precalculated_groups_twice!.(eachslice(ceres_data[var]; dims = (1,2)), Ref(ceres_float_time),Ref(ceres_precalculated_month_groups))
end

# Load in the ERA5 data using load_funcs.jl
single_level_vars = []
pressure_level_vars = ["t", "z", "u", "v", "w"]
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

    deseasonalize_and_detrend_precalculated_groups_twice!.(eachslice(era5_data[all_var]; dims = (1,2,3)), Ref(ceres_float_time),Ref(ceres_precalculated_month_groups))
end
for all_var in single_level_vars
    era5_data[all_var] = era5_data[all_var][:,:,in_time_period.(era5_time, Ref(analysis_bounds))]

    deseasonalize_and_detrend_precalculated_groups_twice!.(eachslice(era5_data[all_var]; dims = (1,2)), Ref(ceres_float_time),Ref(ceres_precalculated_month_groups))
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

# Filter ENSO-removed data and convert date column

local_df = DataFrames.innerjoin(era5_local_df, ceres_local_df, nonlocal_rad_df, ceres_global_df, on = :date)

# Load ENSO data (ONI 3.4) and convert to DataFrame
all_time = analysis_bounds
enso_data, enso_dates = load_enso_data(all_time)
enso_dates = enso_dates["time"]
enso_dates .= round.(enso_dates, Dates.Month(1), RoundDown)

# Pregenerate complete ENSO DataFrame with all lag columns outside the loop
enso_df = DataFrame(date = enso_dates)
for (key, values) in pairs(enso_data)
    enso_df[!, string(key)] = values
end

dropmissing!(enso_df)

enso_X = reduce(hcat, eachcol(enso_df[:, Not(:date)]))

# Join ENSO data with local data
local_df.date = Date.(local_df.date)
enso_df.date = Date.(enso_df.date)
local_df = DataFrames.innerjoin(local_df, enso_df, on = :date)

lags_of_interest = [-24, -12, -6, -3, 0, 3, 6, 12, 24]
single_level_grids = [ceres_data[var] for var in ceres_varnames]

time_series_correlates = ["toa_net_all_mon", "toa_net_lw_mon", "toa_net_sw_mon", "LTS_1000", "θ_1000", "θ_700"]
append!(time_series_correlates, global_toa_rad_names)

connected_single_level_var = []
connected_pressure_level_var = []

level_var_names = []
press_level_names = string.(era5_coords["pressure_level"]) .* " hPa"
press_sfc_joint_names = vcat(["sfc"], press_level_names)

alone_pressure_level_var = setdiff(pressure_level_vars, connected_pressure_level_var)

array_of_level_data = vec([vcat([era5_data[sl_var]], vec(collect(eachslice(era5_data[pl_var]; dims = (3,))))) for (sl_var, pl_var) in zip(connected_single_level_var, connected_pressure_level_var)])
append!(array_of_level_data, vec([vec(collect(eachslice(era5_data[pl_var]; dims = (3,)))) for pl_var in alone_pressure_level_var]))

level_var_names = vcat(level_var_names, alone_pressure_level_var)

#Now write functions to remove ENSO from the local variables via PLS

function remove_enso_gridded_data!(enso_X, Y; lats = nothing, n_components = 1)
    #First fit the model
    original_dims = size(Y)
    spatial_dims = 1:3
    reshaped_Y = reshape(Y, product(original_dims[spatial_dims]), original_dims[4])
    reshaped_Y = permutedims(reshaped_Y, (2,1)) #Now time is first dimension

    #Now normalize X and Y
    enso_X = copy(enso_X)
    enso_X, means_X, stds_X = normalize_input!(enso_X, meanfunc, nostd)

    reshaped_Y, means_Y, stds_Y = normalize_input!(reshaped_Y, meanfunc, my_std_func)

    #apply the cos weighting if desired
    root_cos_weights = nothing
    if !isnothing(lats)
        root_cos_weights = sqrt.(cosd.(lats'))
        reshaped_Y .*= root_cos_weights
    end
    #Calculate the model and predict the ENSO component
    pls_model = make_pls_regressor(enso_X, reshaped_Y, n_components)
    Y_predicted = predict(pls_model, enso_X)
    Y_deflated = reshaped_Y - Y_predicted

    #undo the cosine weighting and normalization
    if !isnothing(lats)
        Y_deflated ./= root_cos_weights
    end
    Y_deflated = denormalize_output!(Y_deflated, means_Y, stds_Y)
end

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
        lagged_ts_dict = Dictionary{Int, Vector{Union{Float64, Missing}}}()
        for lag in lags_of_interest
            lag_col_name = "$(var)_lag_$(lag)"
            if lag_col_name in names(local_df)
                set!(lagged_ts_dict, lag, local_df[!, lag_col_name])
            end
        end
        
        # Generate correlations for single level variables (CERES TOA radiation)
        for (i, grid_data) in enumerate(single_level_grids)
            var_name = ceres_varnames[i]
            filename = "$(regionalized_name)_vs_$(var_name)_correlations.png"
            filepath = joinpath(var_save_dir, filename)

            if isfile(filepath)
                println("File $filepath already exists, skipping...")
                continue
            end

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
        var_save_dir = joinpath(base_save_dir, regionalized_name)
        mkpath(var_save_dir)
        
        # Create lagged time series dictionary for different lags of interest using Dictionaries.jl
        lagged_ts_dict = Dictionary{Int, Vector{Union{Float64, Missing}}}()
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

            filename = "$(regionalized_name)_vs_$(level_var_name)_correlations.png"
            filepath = joinpath(var_save_dir, filename)

            if isfile(filepath)
                println("File $filepath already exists, skipping...")
                continue
            end
            
            n_levels = length(level_grids)
            n_lags = length(lagged_ts_dict)

            out_corr_arr = Array{Matrix{Float64}}(undef, n_levels, n_lags)
            plot_labels = Array{String}(undef, n_levels, n_lags)

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
                    plot_labels[level, lag_idx] = "$(grid_level_names[level]) Lag $(lag_val)"
                end
            end

            out_corr_arr = vec(out_corr_arr)
            #Generate the plot
            fig = plot_multiple_levels(era5_lat, era5_lon, out_corr_arr, (n_levels, n_lags); 
                                      subtitles=vec(plot_labels), 
                                      colorbar_label="Correlation with $level_var_name",
                                      )

            fig.suptitle("Corr between $var and $level_var_name")
            # Save figure in variable-specific directory
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            println("Saved figure to: $filepath")
            plt.close(fig)
        end
    end
end





