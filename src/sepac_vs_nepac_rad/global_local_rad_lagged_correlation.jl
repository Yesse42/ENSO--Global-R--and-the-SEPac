"""
This script computes the lagged correlation between global net radiation (lags -24 to 24 months) 
and lag 0 local shortwave, longwave, and net radiation for each region.
Local SW and LW correlations are weighted by their standard deviation relative to net radiation.
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using CSV, DataFrames, Dates, JLD2, Statistics

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/compare_stratocumulus_regions/global_local_rad_connection"

# Create output directory if it doesn't exist
if !isdir(visdir)
    mkpath(visdir)
end

mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)

region_names = sort(collect(keys(region_data["regional_masks_ceres"])))

# Define radiation variable names
toa_rad_names = "toa_" .* ["net_all", "net_lw", "net_sw"] .* "_mon"
global_net_rad_name = "gtoa_net_all_mon"
toa_rad_names .*= "_detrend_deseason"

date_range = (Date(2000, 3), Date(2022, 3, 31)) # Time range accounting for 24 month lag
is_analysis_time(t) = in_time_period(t, date_range)

# Load global net radiation data
println("Loading global net radiation data...")
global_rad_data, global_coords = load_new_ceres_data([global_net_rad_name], date_range)
global_net_rad = global_rad_data[global_net_rad_name]
analysis_times = global_coords["time"]
global_float_times = calc_float_time.(analysis_times)
global_months = month.(analysis_times)

# Detrend and deseasonalize global net radiation
detrend_and_deseasonalize!(global_net_rad, global_float_times, global_months)

# Create lagged global net radiation array
println("Creating lagged global net radiation arrays...")
lags = -24:24
n_times = length(analysis_times)
global_net_rad_lagged = Dict{Int, Vector{Float64}}()

for lag in lags
    if lag == 0
        global_net_rad_lagged[lag] = global_net_rad
    elseif lag > 0
        # Positive lag: global radiation leads (shift global data backward)
        padded_array = vcat(fill(NaN, lag), global_net_rad[1:end-lag])
        global_net_rad_lagged[lag] = padded_array
    else
        # Negative lag: global radiation lags (shift global data forward)
        abs_lag = abs(lag)
        padded_array = vcat(global_net_rad[abs_lag+1:end], fill(NaN, abs_lag))
        global_net_rad_lagged[lag] = padded_array
    end
end

# Create regional correlation plot
using Plots
pythonplot()

# Define region names and plotting setup
regions = region_names
plot_arr = Array{Any}(undef, 1, length(region_names))

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

println("Computing correlations for each region...")
for (j, region) in enumerate(regions)
    println("Processing region: $region")
    
    # Load local radiation data for this region
    local_rad_datafile = joinpath(datadir, "ceres_region_avg_$(region).csv")
    local_rad_df = CSV.read(local_rad_datafile, DataFrame)
    filter!(row -> is_analysis_time(row.date), local_rad_df)
    
    # Get local radiation arrays (lag 0)
    local_rad_data = [local_rad_df[!, name] for name in toa_rad_names]
    
    # Create plot for this region
    p = plot(title = "$(region) - Global Net Radiation vs. Local Radiation", 
             xlabel = "Lag (months), global net radiation lags to right", 
             ylabel = "(Weighted) Correlation", 
             legend = :topright, 
             size = (600, 400), 
             titlefontsize = 8, 
             yticks = -1:0.05:1)

    # Find net radiation index for weighting
    net_rad_idx = findfirst(!check_if_lw_sw, toa_rad_names)
    net_rad_data = local_rad_data[net_rad_idx]
    net_rad_std = std(skipmissing(net_rad_data))

    # Compute correlations for each local radiation type
    for (rad, rad_name) in zip(local_rad_data, toa_rad_names)
        corrs = Float64[]
        
        for lag in lags
            # Get lagged global net radiation
            lagged_global_net = global_net_rad_lagged[lag]
            
            # Find valid indices (both local and global data available)
            valid_indices = .!(ismissing.(rad) .| ismissing.(lagged_global_net) .| isnan.(lagged_global_net))
            
            if sum(valid_indices) > 10  # Need sufficient data points
                push!(corrs, cor(rad[valid_indices], lagged_global_net[valid_indices]))
            else
                push!(corrs, NaN)
            end
        end
        
        # Apply weighting for SW and LW components
        if check_if_lw_sw(rad_name)
            weight = std(skipmissing(rad)) / net_rad_std
            corrs .*= weight
        end
        
        # Track maximum absolute value for y-axis scaling
        valid_corrs = corrs[.!isnan.(corrs)]
        if !isempty(valid_corrs)
            global maxabsval = maximum([maxabsval; abs.(valid_corrs)...])
        end
        
        # Plot the correlation time series
        plot!(p, lags, corrs, 
              label = map_to_shortlabel(rad_name), 
              color = map_to_color(rad_name), 
              lw = 2)
    end
    
    plot_arr[1, j] = p
end

# Apply consistent y-axis limits across all plots
println("Applying consistent y-axis limits...")
ylims!.(plot_arr, Ref((-maxabsval - 0.05, maxabsval + 0.05)))

# Create the final combined plot
final_plot = plot(plot_arr..., layout = (1, length(region_names)), size = (1800, 600),
                 plot_title = "Lagged Correlation: Global Net Radiation vs Local Radiation Components", 
                 titlefontsize = 8)

# Save the plot
output_filename = joinpath(visdir, "global_net_vs_local_rad_lagged_correlations.png")
savefig(final_plot, output_filename)

println("Plot saved to: $output_filename")
println("Analysis complete!")
println("Maximum absolute correlation found: $(round(maxabsval, digits=3))")
