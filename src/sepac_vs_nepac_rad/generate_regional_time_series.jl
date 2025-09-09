"""
This script does these things:
1. Generates the spatial means for the three stratocumulus regions (SEPac, NEPac, SEAtl)
    For t2m, LTS, CRE, and SW, LW, and Net TOA fluxes
2. Detrend and deseasonalize those time series
3. Generates another time series which has the optimal ENSO lag removed from the original time series
4. Generates lags of all time series from -24 to 24 months
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using JLD2

function generate_spatial_mean(data, latitudes, mask)
    weights = mask .* cosd.(latitudes')
    return vec(sum(data .* weights; dims=(1, 2)) ./ sum(weights))
end

function pot_temp(T, P)
    return T .* (1000 ./ P).^(2/7)
end

era5_vars = ["t2m", "msl", "t"]

all_time = (Date(0), Date(100000))

era5_data, era5_coords = load_era5_data(era5_vars, all_time)

era5_lat = era5_coords["latitude"]
era5_lon = era5_coords["longitude"]
press_levels = era5_coords["pressure_level"]

idx_700_hpa = findfirst(press_levels .== 700)

sfc_pot_temp = pot_temp.(era5_data["t2m"], era5_data["msl"]/100)  # Convert Pa to hPa

pot_temp_700 = pot_temp.(era5_data["t"][:,:,idx_700_hpa, 1:end-1], 700)

LTS = pot_temp_700 .- sfc_pot_temp[:,:,:]

#Open the precalculated masks via jld2
mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)

era5_data_dict = Dictionary(["t2m", "LTS"], [era5_data["t2m"], LTS])

era5_time = era5_coords["time"][1:end]
era5_float_times = calc_float_time.(era5_time)
era5_months = month.(era5_time)

region_names = keys(region_data["bounds"])
era5_lat = region_data["latitude_era5"]
era5_lon = region_data["longitude_era5"]
ceres_lon = region_data["longitude_ceres"]
ceres_lat = region_data["latitude_ceres"]

era5_masks = region_data["regional_masks_era5"]
ceres_masks = region_data["regional_masks_ceres"]

cre_names = "toa_cre_" .* ["sw", "lw", "net"] .* "_mon"
toa_rad_names = "toa_" .* ["net_all", "net_lw", "net_sw"] .* "_mon"
ceres_varnames = vcat(cre_names, toa_rad_names)

#Load in the ceres data
ceres_period = (Date(0), Date(2025, 2, 28))
ceres_data, ceres_coords = load_new_ceres_data(ceres_varnames, ceres_period)
ceres_time = ceres_coords["time"]
ceres_time .= round.(ceres_time, Dates.Month(1), RoundDown)

enso_data, enso_dates = load_enso_data(all_time)
enso_dates = enso_dates["time"]
enso_dates .= round.(enso_dates, Dates.Month(1), RoundDown)
enso_df = DataFrame(["date" => enso_dates, pairs(enso_data)...])

function missing_sensitive_corr(x, y)
    valid_idx = .!(ismissing.(x) .| ismissing.(y))
    if sum(valid_idx) < 2
        return NaN
    else
        return cor(x[valid_idx], y[valid_idx])
    end
end

function remove_x_from_y_via_linear_regression(x,y)
    @assert length(x) == length(y)
    out = Array{Union{Missing, Float64}}(undef, length(y))
    both_valid = .!(ismissing.(x) .| ismissing.(y))
    x = x[both_valid]
    y = y[both_valid]
    slope, intercept = least_squares_fit(x, y)
    out[both_valid] .= y .- (slope .* x .+ intercept)
    out[(!).(both_valid)] .= missing
    return out
end

detrend_deseason_suffix = "_detrend_deseason"
lag_suffix = "_lag_"

enso_base_name = "oni_lag_"

standard_lags = -24:24

enso_compare_dates = (Date(2000), Date(2022, 4))

savedir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/region_average_time_series"

for region in region_names
    #Load in the era5 mask for the region
    era5_mask = era5_masks[region]

    era5_df = DataFrame(date = era5_time)
    era5_lagged_df = DataFrame(date = era5_time)
    era5_enso_lagged_df = DataFrame(date = era5_time)
    #Now form the weighted average time series for all era5 fluxes
    for var in keys(era5_data_dict)
        println("Processing $var for $region")
        data = era5_data_dict[var]
        mean_ts = generate_spatial_mean(data, era5_lat, era5_mask)
        era5_df[!, var] = mean_ts

        #Add a detrended and deseasonalized version of the time series
        detrend_deseason_name = Symbol(var * detrend_deseason_suffix)
        detrend_deseasoned, _ = detrend_and_deseasonalize!(copy(mean_ts), era5_float_times, era5_months)
        era5_df[!, detrend_deseason_name] = detrend_deseasoned

        #Now, generate the lagged time series
        for lag in standard_lags
            lagged_name = Symbol(var * lag_suffix * string(lag))
            era5_lagged_df[!, lagged_name] = time_lag(detrend_deseasoned, lag)
        end

        #Now compute the lagged enso index which has maximum correlation with the detrended and deseasonalized time series
        joined_data_df = DataFrames.innerjoin(enso_df, era5_df[:, [:date, detrend_deseason_name]], on = :date)
        oni_compare_dates_df = filter(row -> in_time_period(row.date, enso_compare_dates), joined_data_df)
        opt_corr, opt_lag_idx = findmax([abs(missing_sensitive_corr(oni_compare_dates_df[!, "oni_lag_$lag"], oni_compare_dates_df[!, detrend_deseason_name])) for lag in standard_lags])
        opt_lag = standard_lags[opt_lag_idx]

        display("The optimal lag for $var in $region is $opt_lag months, with corr of $opt_corr")

        #Now remove the optimal enso lag from the detrend deseasoned
        enso_residual = remove_x_from_y_via_linear_regression(joined_data_df[!, "oni_lag_$opt_lag"], joined_data_df[!, detrend_deseason_name])

        #Now lag the residual time series
        for lag in standard_lags
            lagged_name = Symbol(var * "_detrend_deseason_enso_removed" * lag_suffix * string(lag))
            era5_enso_lagged_df[!, lagged_name] = time_lag(enso_residual, lag)
        end
    end

    #Now do the same for the ceres data
    #Load in the ceres mask for the region
    ceres_mask = ceres_masks[region]


    ceres_df = DataFrame(date = ceres_time)
    ceres_lagged_df = DataFrame(date = ceres_time)
    ceres_enso_lagged_df = DataFrame(date = ceres_time)

    #Now form the weighted average time series for all ceres fluxes
    for var in ceres_varnames
        println("Processing $var for $region")
        data = ceres_data[var]
        mean_ts = generate_spatial_mean(data, ceres_lat, ceres_mask)
        ceres_df[!, var] = mean_ts

        #Add a detrended and deseasonalized version of the time series
        ceres_float_times = calc_float_time.(ceres_time)
        ceres_months = month.(ceres_time)
        detrend_deseason_name = Symbol(var * detrend_deseason_suffix)
        detrend_deseasoned, _ = detrend_and_deseasonalize!(copy(mean_ts), ceres_float_times, ceres_months)
        ceres_df[!, detrend_deseason_name] = detrend_deseasoned

        #Now, generate the lagged time series
        for lag in standard_lags
            lagged_name = Symbol(var * lag_suffix * string(lag))
            ceres_lagged_df[!, lagged_name] = time_lag(detrend_deseasoned, lag)
        end

        #Now compute the lagged enso index which has maximum correlation with the detrended and deseasonalized time series
        joined_data_df = DataFrames.innerjoin(enso_df, ceres_df[:, [:date, detrend_deseason_name]], on = :date)
        oni_compare_dates_df = filter(row -> in_time_period(row.date, enso_compare_dates), joined_data_df)
        opt_lag_idx = argmax([abs(missing_sensitive_corr(oni_compare_dates_df[!, "oni_lag_$lag"], oni_compare_dates_df[!, detrend_deseason_name])) for lag in standard_lags])
        opt_lag = standard_lags[opt_lag_idx]

        #Now remove the optimal enso lag from the detrend deseasoned
        enso_residual = remove_x_from_y_via_linear_regression(joined_data_df[!, "oni_lag_$opt_lag"], joined_data_df[!, detrend_deseason_name])

        #Now lag the residual time series
        for lag in standard_lags
            lagged_name = Symbol(var * "_detrend_deseason_enso_removed" * lag_suffix * string(lag))
            ceres_enso_lagged_df[!, lagged_name] = time_lag(enso_residual, lag)
        end
    end


    #Now save both dataframes to CSV 
    era5_savepath = joinpath(savedir, "era5_region_avg_" * region * ".csv")
    ceres_savepath = joinpath(savedir, "ceres_region_avg_" * region * ".csv")
    era5_lagged_savepath = joinpath(savedir, "era5_region_avg_lagged_" * region * ".csv")
    ceres_lagged_savepath = joinpath(savedir, "ceres_region_avg_lagged_" * region * ".csv")
    era5_enso_residual_savepath = joinpath(savedir, "era5_region_avg_enso_removed_" * region * ".csv")
    ceres_enso_residual_savepath = joinpath(savedir, "ceres_region_avg_enso_removed_" * region * ".csv")

    CSV.write(era5_savepath, era5_df)
    CSV.write(ceres_savepath, ceres_df)
    CSV.write(era5_lagged_savepath, era5_lagged_df)
    CSV.write(ceres_lagged_savepath, ceres_lagged_df)
    CSV.write(era5_enso_residual_savepath, era5_enso_lagged_df)
    CSV.write(ceres_enso_residual_savepath, ceres_enso_lagged_df)
end