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

analysis_period = (Date(2000, 3), Date(2017, 2, 28))

#Load in the sepac time series, and the global radiation time series, for the full length of their respective time periods. Deseasonalize and detrend them twice.

patrizio_vars = ["SST", "T_ocn", "T_atm"]

#Now load in the patrizio sst partition data.

ceres_vars_of_interest = ["toa_net_sw_mon", "toa_net_lw_mon", "toa_net_all_mon"]
append!(ceres_vars_of_interest, "g" .* ceres_vars_of_interest)
era5_vars_of_interest = ["LTS_1000", "θ_1000", "θ_700", "θ_1000_enso_removed", "θ_700_enso_removed"]

local_vars_of_interest = vcat(ceres_vars_of_interest, era5_vars_of_interest)

lags = [-24, -12, -6, -3, 0, 3, 6, 12, 24]  #in months

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

# Load in the SEPac SST index time series
region = "SEPac_feedback_definition"
sepac_sst_data, sepac_coords = load_sepac_sst_index(analysis_period; lags=lags)

# Load in the local time series data for SEPac from the specified directory
local_ts_dir = "../../data/sepac_lts_data/local_region_time_series"
era5_local_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
ceres_local_df = CSV.read(joinpath(local_ts_dir, "ceres_region_avg_lagged_$(region).csv"), DataFrame)

# Load in the ENSO-removed local time series data
era5_enso_removed_df = CSV.read(joinpath(local_ts_dir, "era5_region_avg_enso_removed_$(region).csv"), DataFrame)

# Load in the lagged global CERES time series
ceres_global_df = CSV.read("../../data/CERES/lagged/global_ceres_lagged_detrended_deseasonalized.csv", DataFrame)

# Load in the Patrizio SST partition gridded data - this is the only gridded data we need
patrizio_data, patrizio_coords = load_patrizio_sst_data(patrizio_vars, analysis_period)
patrizio_lat = patrizio_coords["latitude"]
patrizio_lon = patrizio_coords["longitude"]
patrizio_time = round.(patrizio_coords["time"], Month(1), RoundDown)
patrizio_time_valid = in_time_period.(patrizio_time, Ref(analysis_period))
patrizio_time = patrizio_time[patrizio_time_valid]
patrizio_float_time = calc_float_time.(patrizio_time)
patrizio_precalculated_month_groups = groupfind(month, patrizio_time)

# Bring all time series data into the same analysis bounds time period
era5_local_df[!, :date] = Date.(era5_local_df[!, :date])
ceres_local_df[!, :date] = Date.(ceres_local_df[!, :date])
ceres_global_df[!, :date] = Date.(ceres_global_df[!, :date])
era5_enso_removed_df[!, :date] = Date.(era5_enso_removed_df[!, :date])

# Filter all dataframes to analysis bounds
era5_local_df = filter(row -> analysis_period[1] <= row.date <= analysis_period[2], era5_local_df)
ceres_local_df = filter(row -> analysis_period[1] <= row.date <= analysis_period[2], ceres_local_df)
ceres_global_df = filter(row -> analysis_period[1] <= row.date <= analysis_period[2], ceres_global_df)
era5_enso_removed_df = filter(row -> analysis_period[1] <= row.date <= analysis_period[2], era5_enso_removed_df)

# Extract theta ENSO residuals from the ENSO-removed dataframe
theta_residual_cols = filter(name -> contains(name, "θ_1000") || contains(name, "θ_700"), names(era5_enso_removed_df))
theta_residuals_df = era5_enso_removed_df[:, vcat([:date], Symbol.(theta_residual_cols))]

# Rename theta residual columns to distinguish them as ENSO-removed
rename_dict = Dict{Symbol, Symbol}()
for col in theta_residual_cols
    if contains(col, "θ_1000")
        rename_dict[Symbol(col)] = Symbol(replace(col, "θ_1000" => "θ_1000_enso_removed"))
    elseif contains(col, "θ_700")
        rename_dict[Symbol(col)] = Symbol(replace(col, "θ_700" => "θ_700_enso_removed"))
    end
end
rename!(theta_residuals_df, collect(pairs(rename_dict)))

# Join all local time series data
local_df = DataFrames.innerjoin(era5_local_df, ceres_local_df, ceres_global_df, theta_residuals_df, on = :date)

# Calculate standard deviations gridpoint by gridpoint (along time dimension only)
skipnanmissing(x) = filter(x -> !ismissing(x) && !isnan(x), x)
sst_std_grid = mapslices(x -> std(skipnanmissing(x)), patrizio_data["SST"], dims=3)[:, :, 1]
t_ocn_std_grid = mapslices(x -> std(skipnanmissing(x)), patrizio_data["T_ocn"], dims=3)[:, :, 1]
t_atm_std_grid = mapslices(x -> std(skipnanmissing(x)), patrizio_data["T_atm"], dims=3)[:, :, 1]

# Verification plot: Check if SST = T_ocn + T_atm over time at a single equatorial Pacific point
# Find indices for equatorial Pacific point (approximately 0°N, 180°W)
target_lat = 35
target_lon = 160
lat_idx = argmin(abs.(patrizio_lat .- target_lat))
lon_idx = argmin(abs.(patrizio_lon .- target_lon))

# Extract time series at this point
sst_timeseries = patrizio_data["SST"][lon_idx, lat_idx, :]
t_ocn_timeseries = patrizio_data["T_ocn"][lon_idx, lat_idx, :]
t_atm_timeseries = patrizio_data["T_atm"][lon_idx, lat_idx, :]
sum_timeseries = t_ocn_timeseries .+ t_atm_timeseries
diff_timeseries = sst_timeseries .- sum_timeseries

# Convert missings to NaNs
sst_timeseries = replace(sst_timeseries, missing => NaN)
t_ocn_timeseries = replace(t_ocn_timeseries, missing => NaN)
t_atm_timeseries = replace(t_atm_timeseries, missing => NaN)
sum_timeseries = replace(sum_timeseries, missing => NaN)
diff_timeseries = replace(diff_timeseries, missing => NaN)

# Create time series plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Top panel: All components
axes[0].plot(1:length(sst_timeseries), sst_timeseries, label="SST", linewidth=2)
axes[0].plot(1:length(t_ocn_timeseries), t_ocn_timeseries, label="T_ocn", linewidth=1.5)
axes[0].plot(1:length(t_atm_timeseries), t_atm_timeseries, label="T_atm", linewidth=1.5)
axes[0].plot(1:length(sum_timeseries), sum_timeseries, label="T_ocn + T_atm", linestyle="--", linewidth=2)
axes[0].set_ylabel("Temperature Anomaly")
axes[0].set_title("SST Components at $(patrizio_lat[lat_idx])°N, $(patrizio_lon[lon_idx])°E")
axes[0].legend()
axes[0].grid(true, alpha=0.3)

# Bottom panel: Difference
axes[1].plot(1:length(diff_timeseries), diff_timeseries, color="red", linewidth=1.5)
axes[1].set_ylabel("Difference")
axes[1].set_xlabel("Time Index")
axes[1].set_title("SST - (T_ocn + T_atm)")
axes[1].grid(true, alpha=0.3)

fig.suptitle("Verification: SST vs T_ocn + T_atm Time Series")
plt.tight_layout()
fig.savefig(joinpath(base_vis_dir, "verification_sst_tocn_tatm_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Now calculate gridded correlations between the local variables and all Patrizio SST grids together
for var in local_vars_of_interest
    if var[1] ≠ 'g'
        regionalized_name = region * " " * var
    else
        regionalized_name = var
    end
    
    if any(occursin.(var, names(local_df)))
        # Create specific save directory for this variable
        var_save_dir = joinpath(base_vis_dir, regionalized_name)
        mkpath(var_save_dir)
        
        # Create lagged time series dictionary for different lags of interest using Dictionaries.jl
        lagged_ts_dict = Dictionary{Int, Vector{Union{Float64, Missing}}}()
        for lag in lags
            lag_col_name = "$(var)_lag_$(lag)"
            if lag_col_name in names(local_df)
                set!(lagged_ts_dict, lag, local_df[!, lag_col_name])
            end
        end
        
        # Generate combined correlation plot for all three Patrizio variables (3x7 layout)
        filename = "$(regionalized_name)_vs_all_patrizio_vars_correlations.png"
        filepath = joinpath(var_save_dir, filename)

        if isfile(filepath)
            println("File $filepath already exists, skipping...")
            continue
        end

        println("Processing correlations for $var with all Patrizio variables")
        
        # Calculate correlations for all three variables and all lags
        all_corr_values = Array{Array{Float64,2}}(undef, length(patrizio_vars), length(lags))
        all_subtitles = Array{String, 2}(undef, length(patrizio_vars), length(lags))
        
        # Order: SST (top row), T_atm (middle row), T_ocn (bottom row)
        patrizio_ordered = ["SST", "T_atm", "T_ocn"]
        patrizio_labels = ["SST", "T_atm", "T_ocn"]
        
        for (i, patrizio_var) in enumerate(patrizio_ordered)
            if haskey(patrizio_data, patrizio_var)
                grid_data = patrizio_data[patrizio_var]
                
                for (j, (lag_key, lag_ts)) in enumerate(pairs(lagged_ts_dict))
                    corr_map = calculate_corrfunc_grid(grid_data, lag_ts; corrfunc = skipmissing_corr)
                    
                    # Apply scaling factors using gridpoint-specific standard deviations
                    if patrizio_var == "T_ocn"
                        corr_map = corr_map .* (t_ocn_std_grid ./ sst_std_grid)
                    elseif patrizio_var == "T_atm"
                        corr_map = corr_map .* (t_atm_std_grid ./ sst_std_grid)
                    end
                    # SST remains unscaled
                    
                    all_corr_values[i, j] = corr_map
                    all_subtitles[i, j] = "$(patrizio_labels[i]) Lag $(lag_key)"
                end
            end
        end
        
        # Calculate colorbar limits based on SST correlations only (first 7 maps)
        sst_corr_values = all_corr_values[1, :]  # First row is SST
        sst_maxabsval = maximum([maximum(abs(val) for val in arr if !isnan(val)) for arr in sst_corr_values])
        vmin = -sst_maxabsval
        vmax = sst_maxabsval
        colornorm = colors.Normalize(vmin, vmax)

        #Manually clamp t_atm and t_ocn corrs to the vmin/vmax range as well
        for i in 1:size(all_corr_values, 1)
            for j in 1:size(all_corr_values, 2)
                all_corr_values[i, j] = clamp.(all_corr_values[i, j], vmin, vmax)
            end
        end
        
        # Create 3x7 plot layout
        plot_layout = (3, length(lags))
        
        # Plot using the global plotting function
        fig = plot_multiple_levels(patrizio_lat, patrizio_lon, 
                                  all_corr_values, plot_layout; 
                                  subtitles=permutedims(all_subtitles), 
                                  colorbar_label="Correlation (scaled)",
                                  colornorm=colornorm)

        fig.suptitle("Corr between $regionalized_name and Patrizio SST components")
        
        # Save figure in variable-specific directory
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        println("Saved figure to: $filepath")
        plt.close(fig)
    end
end

