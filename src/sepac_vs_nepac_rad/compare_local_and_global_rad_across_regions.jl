"""
This script aims to calculate the correlation between local radiation and t2m, and global radiation and t2m, across the three stratocumulus regions (SEPac, NEPac, SEAtl).
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using CSV, DataFrames, Dates, JLD2, Statistics

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/compare_stratocumulus_regions/comparison_plots"

mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)

region_names = sort(collect(keys(region_data["regional_masks_ceres"])))

cre_names = "toa_cre_" .* ["sw", "lw", "net"] .* "_mon"
toa_rad_names = "toa_" .* ["net_all", "net_lw", "net_sw"] .* "_mon"
global_rad_names = "g" .* toa_rad_names
toa_rad_names .*= "_detrend_deseason"

date_range = (Date(2000, 3), Date(2022, 3, 31)) #This time range needs to account for the 24 month lag shortening the POR
is_analysis_time(t) = in_time_period(t, date_range)

#Load in global radiation
ceres_global_rad, ceres_global_coords = load_new_ceres_data(global_rad_names, date_range)
analysis_times = ceres_global_coords["time"]
ceres_float_times = calc_float_time.(analysis_times)
ceres_months = month.(analysis_times)
detrend_and_deseasonalize!.(ceres_global_rad, Ref(ceres_float_times), Ref(ceres_months))

# Create regional correlation plot
using Plots
gr()

# Define region names and lags
regions = region_names
region_colors = [:blue, :red, :green]
lags = -24:24

plot_arr = Array{Any}(undef, 2, length(region_names))

datadir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/region_average_time_series"

check_if_lw_sw(str) = any(occursin.(["lw", "sw"], [str]))

function map_to_shortlabel(name)
    if occursin("sw", name)
        return "SW"
    elseif occursin("lw", name)
        return "LW"
    elseif occursin("net", name)
        return "Net"
    else
        return "Unknown"
    end
end

function map_to_color(name)
    if occursin("sw", name)
        return :green
    elseif occursin("lw", name)
        return :red
    elseif occursin("net", name)
        return :blue
    else
        return :black
    end
end

maxabsval = 0.0

for (j, region) in enumerate(regions)
    local_rad_datafile = joinpath(datadir, "ceres_region_avg_$(region).csv")
    local_rad_df = CSV.read(local_rad_datafile, DataFrame)
    filter!(row -> is_analysis_time(row.date), local_rad_df)
    local_global_rad = [(collect(ceres_global_rad), global_rad_names), ([local_rad_df[!, name] for name in toa_rad_names], toa_rad_names)]

    t2m_lagged_df = CSV.read(joinpath(datadir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
    filter!(row -> is_analysis_time(row.date), t2m_lagged_df)
    for (i, (rad_data, rad_names)) in enumerate(local_global_rad)
        p = plot(title = "$(region) - $(i == 1 ? "Global" : "Local") Radiation vs. T2M", xlabel = "Lag (months), radiation lags to right", ylabel = "(Weighted) Correlation", legend = :topright, size = (600, 400), titlefontsize = 8, yticks = -1:0.05:1)

        net_rad_idx = only(findfirst(!check_if_lw_sw, rad_names))
        net_rad_data = rad_data[net_rad_idx]
        net_rad_std = std(skipmissing(net_rad_data))

        for (rad, rad_name) in zip(rad_data, rad_names)
            corrs = Float64[]
            for lag in lags
                lagged_t2m = t2m_lagged_df[!, "t2m_lag_$(lag)"]
                valid_indices = .!(ismissing.(rad) .| ismissing.(lagged_t2m))
                if sum(valid_indices) > 0
                    push!(corrs, cor(rad[valid_indices], lagged_t2m[valid_indices]))
                else
                    push!(corrs, NaN)
                end
            end
            if check_if_lw_sw(rad_name)
                weight = std(skipmissing(rad)) / net_rad_std
                corrs .*= weight
            end
            global maxabsval = maximum([maxabsval; abs.(corrs)...])
            plot!(p, lags, corrs, label = map_to_shortlabel(rad_name), color = map_to_color(rad_name), lw = 2)
        end
        plot_arr[i, j] = p
    end
end

ylims!.(plot_arr, Ref((-maxabsval - 0.05, maxabsval + 0.05)))

final_plot = plot(permutedims(plot_arr)..., layout = (2, length(region_names)), size = (1800, 800); plot_title = "Correlation between Local and Global Radiation vs. T2M across Stratocumulus Regions", titlefontsize = 8)

savefig(joinpath(visdir, "local_vs_global_rad_correlation_across_regions.png"))