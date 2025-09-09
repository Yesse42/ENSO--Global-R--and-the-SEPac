"""
This script creates two 4-pane plots showing LTS vs Net CRE scatter plots for different seasons:
- First plot: SEPac region with 4 seasons (DJF, MAM, JJA, SON)
- Second plot: SEPac_feedback_only region with 4 seasons (DJF, MAM, JJA, SON)
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using CSV, DataFrames, Dates, JLD2, Statistics
using StatsBase: corspearman

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/compare_stratocumulus_regions/comparison_plots"

date_range = (Date(2000, 3), Date(2022, 3, 31))
is_analysis_time(t) = in_time_period(t, date_range)

# Create scatterplot array
using Plots
gr()

# Define region names
mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)

region_names = sort(collect(keys(region_data["regional_masks_ceres"])))
regions = region_names
plot_arr = Array{Any}(undef, 2, length(region_names))

# Variables to track global axis limits for the original plots
all_lts_values = Float64[]
all_net_cre_values = Float64[]
all_lts_enso_values = Float64[]

# Define regions and seasons
target_regions = ["SEPac", "SEPac_feedback_only"]
seasons = ["DJF", "MAM", "JJA", "SON"]
season_names = ["Dec-Jan-Feb", "Mar-Apr-May", "Jun-Jul-Aug", "Sep-Oct-Nov"]

datadir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/region_average_time_series"

# Function to determine season from month
function get_season(month_val)
    if month_val in [12, 1, 2]
        return "DJF"
    elseif month_val in [3, 4, 5]
        return "MAM"
    elseif month_val in [6, 7, 8]
        return "JJA"
    elseif month_val in [9, 10, 11]
        return "SON"
    end
end

# First pass: collect all data to determine global axis limits
for (j, region) in enumerate(regions)
    # Load CERES data for this region
    ceres_datafile = joinpath(datadir, "ceres_region_avg_$(region).csv")
    ceres_df = CSV.read(ceres_datafile, DataFrame)
    filter!(row -> is_analysis_time(row.date), ceres_df)
    
    # Load regular LTS data (lagged)
    lts_lagged_df = CSV.read(joinpath(datadir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
    filter!(row -> is_analysis_time(row.date), lts_lagged_df)
    
    # Load ENSO residual LTS data
    lts_enso_removed_df = CSV.read(joinpath(datadir, "era5_region_avg_enso_removed_$(region).csv"), DataFrame)
    filter!(row -> is_analysis_time(row.date), lts_enso_removed_df)
    
    # Collect data ranges
    net_cre = ceres_df[!, "toa_cre_net_mon_detrend_deseason"]
    lts_lag0 = lts_lagged_df[!, "LTS_lag_0"]
    lts_enso_removed_lag0 = lts_enso_removed_df[!, "LTS_detrend_deseason_enso_removed_lag_0"]
    
    # Add valid data to global collections
    valid_indices_regular = .!(ismissing.(net_cre) .| ismissing.(lts_lag0))
    if sum(valid_indices_regular) > 0
        append!(all_net_cre_values, net_cre[valid_indices_regular])
        append!(all_lts_values, lts_lag0[valid_indices_regular])
    end
    
    valid_indices_enso = .!(ismissing.(net_cre) .| ismissing.(lts_enso_removed_lag0))
    if sum(valid_indices_enso) > 0
        append!(all_net_cre_values, net_cre[valid_indices_enso])
        append!(all_lts_enso_values, lts_enso_removed_lag0[valid_indices_enso])
    end
end

# Calculate global axis limits for original plots
lts_xlims = (minimum([all_lts_values; all_lts_enso_values]), maximum([all_lts_values; all_lts_enso_values]))
net_cre_ylims = (minimum(all_net_cre_values), maximum(all_net_cre_values))

println("Global axis limits for original plots:")
println("  LTS xlims: $(lts_xlims)")
println("  Net CRE ylims: $(net_cre_ylims)")

# Second pass: create plots with consistent axis limits
for (j, region) in enumerate(regions)
    # Load CERES data for this region
    ceres_datafile = joinpath(datadir, "ceres_region_avg_$(region).csv")
    ceres_df = CSV.read(ceres_datafile, DataFrame)
    filter!(row -> is_analysis_time(row.date), ceres_df)
    
    # Load regular LTS data (lagged)
    lts_lagged_df = CSV.read(joinpath(datadir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
    filter!(row -> is_analysis_time(row.date), lts_lagged_df)
    
    # Load ENSO residual LTS data
    lts_enso_removed_df = CSV.read(joinpath(datadir, "era5_region_avg_enso_removed_$(region).csv"), DataFrame)
    filter!(row -> is_analysis_time(row.date), lts_enso_removed_df)
    
    # Get the data arrays
    net_cre = ceres_df[!, "toa_cre_net_mon_detrend_deseason"]
    lts_lag0 = lts_lagged_df[!, "LTS_lag_0"]
    lts_enso_removed_lag0 = lts_enso_removed_df[!, "LTS_detrend_deseason_enso_removed_lag_0"]
    
    # Top row: Regular LTS vs Net CRE
    valid_indices_regular = .!(ismissing.(net_cre) .| ismissing.(lts_lag0))
    if sum(valid_indices_regular) > 0
        net_cre_valid = net_cre[valid_indices_regular]
        lts_valid = lts_lag0[valid_indices_regular]
        correlation_regular = cor(net_cre_valid, lts_valid)
        spearman_regular = corspearman(net_cre_valid, lts_valid)
        
        p1 = Plots.scatter(lts_valid, net_cre_valid, 
                    title = "$(region) - Regular LTS vs Net CRE\nPearson's r = $(round(correlation_regular, digits=3)), Spearman's ρ = $(round(spearman_regular, digits=3))", 
                    xlabel = "LTS (K)", 
                    ylabel = "Net CRE (W/m²)", 
                    alpha = 0.6,
                    color = :blue,
                    markersize = 3,
                    size = (400, 300), titlefontsize = 6, label = "",
                    xlims = lts_xlims, ylims = net_cre_ylims)
        
        # Add trend line using simple linear regression
        n = length(lts_valid)
        x_mean = mean(lts_valid)
        y_mean = mean(net_cre_valid)
        slope = sum((lts_valid .- x_mean) .* (net_cre_valid .- y_mean)) / sum((lts_valid .- x_mean).^2)
        intercept = y_mean - slope * x_mean
        lts_range = range(minimum(lts_valid), maximum(lts_valid), length=100)
        trend_line = intercept .+ slope .* lts_range
        plot!(p1, lts_range, trend_line, color = :red, linewidth = 2, label = "")
        
        plot_arr[1, j] = p1
    else
        plot_arr[1, j] = plot(title = "$(region) - No valid data", size = (400, 300))
    end
    
    # Bottom row: ENSO residual LTS vs Net CRE
    valid_indices_enso = .!(ismissing.(net_cre) .| ismissing.(lts_enso_removed_lag0))
    if sum(valid_indices_enso) > 0
        net_cre_valid_enso = net_cre[valid_indices_enso]
        lts_enso_valid = lts_enso_removed_lag0[valid_indices_enso]
        correlation_enso = cor(net_cre_valid_enso, lts_enso_valid)
        spearman_enso = corspearman(net_cre_valid_enso, convert(Array{Float64}, lts_enso_valid))

        p2 = scatter(lts_enso_valid, net_cre_valid_enso, 
                    title = "$(region) - ENSO Residual LTS vs Net CRE\nPearson's r = $(round(correlation_enso, digits=3)), Spearman's ρ = $(round(spearman_enso, digits=3))", 
                    xlabel = "ENSO Residual LTS (K)", 
                    ylabel = "Net CRE (W/m²)", 
                    alpha = 0.6,
                    color = :green,
                    markersize = 3,
                    size = (400, 300),
                    titlefontsize = 6, label = "",
                    xlims = lts_xlims, ylims = net_cre_ylims)
        
        # Add trend line using simple linear regression
        n_enso = length(lts_enso_valid)
        x_mean_enso = mean(lts_enso_valid)
        y_mean_enso = mean(net_cre_valid_enso)
        slope_enso = sum((lts_enso_valid .- x_mean_enso) .* (net_cre_valid_enso .- y_mean_enso)) / sum((lts_enso_valid .- x_mean_enso).^2)
        intercept_enso = y_mean_enso - slope_enso * x_mean_enso
        lts_enso_range = range(minimum(lts_enso_valid), maximum(lts_enso_valid), length=100)
        trend_line_enso = intercept_enso .+ slope_enso .* lts_enso_range
        plot!(p2, lts_enso_range, trend_line_enso, color = :red, linewidth = 2, label = "")
        
        plot_arr[2, j] = p2
    else
        plot_arr[2, j] = plot(title = "$(region) - No valid ENSO residual data", size = (400, 300))
    end
end

# Create the final combined plot
final_plot = plot(permutedims(plot_arr)..., layout = (2, length(region_names)), size = (1200, 800),
                 plot_title = "LTS vs Net CRE Relationships: Regular (top) vs ENSO Residual (bottom)")

# Save the plot
savefig(joinpath(visdir, "LTS_vs_NetCRE_scatterplots_regular_vs_ENSO_residual.png"))

println("Six-panel scatterplot saved to: $(joinpath(visdir, "LTS_vs_NetCRE_scatterplots_regular_vs_ENSO_residual.png"))")

#########################################
# NEW SEASONAL ANALYSIS FOR SEPAC REGIONS
#########################################

# Create seasonal scatter plots for SEPac and SEPac_feedback_only
for region in target_regions
    println("Creating seasonal plots for region: $region")
    
    # Load CERES data for this region
    ceres_datafile = joinpath(datadir, "ceres_region_avg_$(region).csv")
    if !isfile(ceres_datafile)
        println("Warning: File not found: $ceres_datafile")
        continue
    end
    
    ceres_df = CSV.read(ceres_datafile, DataFrame)
    filter!(row -> is_analysis_time(row.date), ceres_df)
    
    # Load LTS data
    lts_lagged_df = CSV.read(joinpath(datadir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
    filter!(row -> is_analysis_time(row.date), lts_lagged_df)
    
    # Add season column
    ceres_df[!, :season] = get_season.(month.(ceres_df[!, :date]))
    lts_lagged_df[!, :season] = get_season.(month.(lts_lagged_df[!, :date]))
    
    # First pass: collect all seasonal data to determine axis limits for this region
    seasonal_lts_values = Float64[]
    seasonal_net_cre_values = Float64[]
    
    for season in seasons
        ceres_season = filter(row -> row.season == season, ceres_df)
        lts_season = filter(row -> row.season == season, lts_lagged_df)
        
        net_cre_season = ceres_season[!, "toa_cre_net_mon_detrend_deseason"]
        lts_season_data = lts_season[!, "LTS_lag_0"]
        
        valid_indices = .!(ismissing.(net_cre_season) .| ismissing.(lts_season_data))
        if sum(valid_indices) > 5
            append!(seasonal_lts_values, lts_season_data[valid_indices])
            append!(seasonal_net_cre_values, net_cre_season[valid_indices])
        end
    end
    
    # Calculate axis limits for this region's seasonal plots
    if !isempty(seasonal_lts_values) && !isempty(seasonal_net_cre_values)
        seasonal_lts_xlims = (minimum(seasonal_lts_values), maximum(seasonal_lts_values))
        seasonal_net_cre_ylims = (minimum(seasonal_net_cre_values), maximum(seasonal_net_cre_values))
    else
        seasonal_lts_xlims = (0, 1)  # Default if no data
        seasonal_net_cre_ylims = (0, 1)
    end
    
    println("Seasonal axis limits for $region:")
    println("  LTS xlims: $(seasonal_lts_xlims)")
    println("  Net CRE ylims: $(seasonal_net_cre_ylims)")
    
    # Create 4-pane plot for this region
    seasonal_plots = Array{Any}(undef, 2, 2)  # 2x2 grid
    
    for (i, season) in enumerate(seasons)
        # Filter data for this season
        ceres_season = filter(row -> row.season == season, ceres_df)
        lts_season = filter(row -> row.season == season, lts_lagged_df)
        
        # Get the data arrays for this season
        net_cre_season = ceres_season[!, "toa_cre_net_mon_detrend_deseason"]
        lts_season_data = lts_season[!, "LTS_lag_0"]
        
        # Check for valid data
        valid_indices = .!(ismissing.(net_cre_season) .| ismissing.(lts_season_data))
        
        if sum(valid_indices) > 5  # Need at least 5 points for meaningful correlation
            net_cre_valid = net_cre_season[valid_indices]
            lts_valid = lts_season_data[valid_indices]
            correlation = cor(net_cre_valid, lts_valid)
            spearman_corr = corspearman(net_cre_valid, convert(Array{Float64}, lts_valid))
            
            # Calculate grid position (row, col)
            row = div(i-1, 2) + 1
            col = mod(i-1, 2) + 1
            
            p = scatter(lts_valid, net_cre_valid,
                       title = "$(season_names[i]) (n=$(length(lts_valid)))\nPearson's r = $(round(correlation, digits=3)), Spearman's ρ = $(round(spearman_corr, digits=3))",
                       xlabel = "LTS (K)",
                       ylabel = "Net CRE (W/m²)",
                       alpha = 0.6,
                       color = [:blue, :red, :green, :orange][i],
                       markersize = 3,
                       size = (400, 300),
                       titlefontsize = 8, label = "",
                       xlims = seasonal_lts_xlims, ylims = seasonal_net_cre_ylims)
            
            # Add trend line
            if length(lts_valid) > 1
                x_mean = mean(lts_valid)
                y_mean = mean(net_cre_valid)
                slope = sum((lts_valid .- x_mean) .* (net_cre_valid .- y_mean)) / sum((lts_valid .- x_mean).^2)
                intercept = y_mean - slope * x_mean
                lts_range = range(minimum(lts_valid), maximum(lts_valid), length=100)
                trend_line = intercept .+ slope .* lts_range
                plot!(p, lts_range, trend_line, color = :red, linewidth = 2, label = "")
            end
            
            seasonal_plots[row, col] = p
        else
            row = div(i-1, 2) + 1
            col = mod(i-1, 2) + 1
            seasonal_plots[row, col] = plot(title = "$(season_names[i]) - No data", size = (400, 300), titlefontsize = 8)
        end
    end
    
    # Create final seasonal plot for this region
    seasonal_final = plot(seasonal_plots..., layout = (2, 2), size = (800, 600),
                         plot_title = "$region - LTS vs Net CRE by Season", titlefontsize = 10)
    
    # Save the seasonal plot
    savefig(seasonal_final, joinpath(visdir, "LTS_vs_NetCRE_seasonal_$(region).png"))
    println("Seasonal plot saved for $region: $(joinpath(visdir, "LTS_vs_NetCRE_seasonal_$(region).png"))")
end

println("All seasonal analysis complete!")

#########################################
# ENSO STATE ANALYSIS FOR SEPAC REGIONS
#########################################

# Load ENSO data for the analysis period
enso_data, enso_coords = load_enso_data(date_range; 
                                       data_dir="/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/ENSO", 
                                       filename="enso_data.csv", 
                                       lags=[0])

oni_values = enso_data["oni_lag_0"]
enso_times = enso_coords["time"]

# Function to classify ENSO states using 0.5K threshold
function classify_enso_state(oni_value, threshold=0.5)
    if ismissing(oni_value)
        return "Unknown"
    elseif oni_value >= threshold
        return "El Niño"
    elseif oni_value <= -threshold
        return "La Niña"
    else
        return "Neutral"
    end
end

# Create ENSO state classifications
enso_states = classify_enso_state.(oni_values, 0.5)
enso_state_names = ["El Niño", "La Niña", "Neutral"]
enso_colors = [:red, :blue, :gray]

println("ENSO state distribution:")
for state in enso_state_names
    count = sum(enso_states .== state)
    println("  $state: $count months")
end

# Create ENSO state scatter plots for SEPac and SEPac_feedback_only
for region in target_regions
    println("Creating ENSO state plots for region: $region")
    
    # Load CERES data for this region
    ceres_datafile = joinpath(datadir, "ceres_region_avg_$(region).csv")
    if !isfile(ceres_datafile)
        println("Warning: File not found: $ceres_datafile")
        continue
    end
    
    ceres_df = CSV.read(ceres_datafile, DataFrame)
    filter!(row -> is_analysis_time(row.date), ceres_df)
    
    # Load LTS data
    lts_lagged_df = CSV.read(joinpath(datadir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
    filter!(row -> is_analysis_time(row.date), lts_lagged_df)
    
    # Match ENSO states with CERES/LTS data by time
    # Convert CERES dates to DateTime for matching
    ceres_times = DateTime.(ceres_df[!, :date] .+ Day(14))
    
    # Find matching indices between ENSO and CERES/LTS data
    matched_indices = []
    matched_enso_states = String[]
    
    for (i, ceres_time) in enumerate(ceres_times)
        enso_idx = findfirst(x -> x == ceres_time, enso_times)
        if enso_idx !== nothing
            push!(matched_indices, i)
            push!(matched_enso_states, enso_states[enso_idx])
        end
    end
    
    println("Matched $(length(matched_indices)) time points for $region")
    
    # Get the matched data
    net_cre_matched = ceres_df[matched_indices, "toa_cre_net_mon_detrend_deseason"]
    lts_matched = lts_lagged_df[matched_indices, "LTS_lag_0"]
    
    # First pass: collect all ENSO state data to determine axis limits for this region
    enso_lts_values = Float64[]
    enso_net_cre_values = Float64[]
    
    for state in enso_state_names
        state_indices = matched_enso_states .== state
        net_cre_state = net_cre_matched[state_indices]
        lts_state = lts_matched[state_indices]
        
        valid_indices = .!(ismissing.(net_cre_state) .| ismissing.(lts_state))
        if sum(valid_indices) > 5
            append!(enso_lts_values, lts_state[valid_indices])
            append!(enso_net_cre_values, net_cre_state[valid_indices])
        end
    end
    
    # Calculate axis limits for this region's ENSO state plots
    if !isempty(enso_lts_values) && !isempty(enso_net_cre_values)
        enso_lts_xlims = (minimum(enso_lts_values), maximum(enso_lts_values))
        enso_net_cre_ylims = (minimum(enso_net_cre_values), maximum(enso_net_cre_values))
    else
        enso_lts_xlims = (0, 1)  # Default if no data
        enso_net_cre_ylims = (0, 1)
    end
    
    println("ENSO state axis limits for $region:")
    println("  LTS xlims: $(enso_lts_xlims)")
    println("  Net CRE ylims: $(enso_net_cre_ylims)")
    
    # Create 3-pane plot for this region (El Niño, La Niña, Neutral)
    enso_plots = Array{Any}(undef, 1, 3)  # 1x3 grid
    
    for (i, state) in enumerate(enso_state_names)
        # Filter data for this ENSO state
        state_indices = matched_enso_states .== state
        net_cre_state = net_cre_matched[state_indices]
        lts_state = lts_matched[state_indices]
        
        # Check for valid data
        valid_indices = .!(ismissing.(net_cre_state) .| ismissing.(lts_state))
        
        if sum(valid_indices) > 5  # Need at least 5 points for meaningful correlation
            net_cre_valid = net_cre_state[valid_indices]
            lts_valid = lts_state[valid_indices]
            correlation = cor(net_cre_valid, lts_valid)
            spearman_corr = corspearman(net_cre_valid, lts_valid)
            
            p = scatter(lts_valid, net_cre_valid,
                       title = "$state (n=$(length(lts_valid)))\nPearson's r = $(round(correlation, digits=3)), Spearman's ρ = $(round(spearman_corr, digits=3))",
                       xlabel = "LTS (K)",
                       ylabel = "Net CRE (W/m²)",
                       alpha = 0.6,
                       color = enso_colors[i],
                       markersize = 3,
                       size = (400, 300),
                       titlefontsize = 8, label = "",
                       xlims = enso_lts_xlims, ylims = enso_net_cre_ylims)
            
            # Add trend line
            if length(lts_valid) > 1
                x_mean = mean(lts_valid)
                y_mean = mean(net_cre_valid)
                slope = sum((lts_valid .- x_mean) .* (net_cre_valid .- y_mean)) / sum((lts_valid .- x_mean).^2)
                intercept = y_mean - slope * x_mean
                lts_range = range(minimum(lts_valid), maximum(lts_valid), length=100)
                trend_line = intercept .+ slope .* lts_range
                plot!(p, lts_range, trend_line, color = :black, linewidth = 2, label = "")
            end
            
            enso_plots[1, i] = p
        else
            enso_plots[1, i] = plot(title = "$state - No data", size = (400, 300), titlefontsize = 8)
        end
    end
    
    # Create final ENSO state plot for this region
    enso_final = plot(enso_plots..., layout = (1, 3), size = (1200, 400),
                     plot_title = "$region - LTS vs Net CRE by ENSO State (±0.5K threshold)", titlefontsize = 10)
    
    # Save the ENSO state plot
    savefig(enso_final, joinpath(visdir, "LTS_vs_NetCRE_ENSO_states_$(region).png"))
    println("ENSO state plot saved for $region: $(joinpath(visdir, "LTS_vs_NetCRE_ENSO_states_$(region).png"))")
end

println("All ENSO state analysis complete!")

#########################################
# DJF-ONLY ENSO STATE ANALYSIS FOR SEPAC REGIONS
#########################################

# Create DJF-only ENSO state scatter plots for SEPac and SEPac_feedback_only
for region in target_regions
    println("Creating DJF-only ENSO state plots for region: $region")
    
    # Load CERES data for this region
    ceres_datafile = joinpath(datadir, "ceres_region_avg_$(region).csv")
    if !isfile(ceres_datafile)
        println("Warning: File not found: $ceres_datafile")
        continue
    end
    
    ceres_df = CSV.read(ceres_datafile, DataFrame)
    filter!(row -> is_analysis_time(row.date), ceres_df)
    
    # Load LTS data
    lts_lagged_df = CSV.read(joinpath(datadir, "era5_region_avg_lagged_$(region).csv"), DataFrame)
    filter!(row -> is_analysis_time(row.date), lts_lagged_df)
    
    # Filter for DJF season only
    ceres_df[!, :season] = get_season.(month.(ceres_df[!, :date]))
    lts_lagged_df[!, :season] = get_season.(month.(lts_lagged_df[!, :date]))
    
    ceres_djf = filter(row -> row.season == "DJF", ceres_df)
    lts_djf = filter(row -> row.season == "DJF", lts_lagged_df)
    
    # Match ENSO states with DJF CERES/LTS data by time
    ceres_djf_times = DateTime.(ceres_djf[!, :date] .+ Day(14))
    
    # Find matching indices between ENSO and DJF CERES/LTS data
    matched_djf_indices = []
    matched_djf_enso_states = String[]
    
    for (i, ceres_time) in enumerate(ceres_djf_times)
        enso_idx = findfirst(x -> x == ceres_time, enso_times)
        if enso_idx !== nothing
            push!(matched_djf_indices, i)
            push!(matched_djf_enso_states, enso_states[enso_idx])
        end
    end
    
    println("Matched $(length(matched_djf_indices)) DJF time points for $region")
    
    # Get the matched DJF data
    net_cre_djf_matched = ceres_djf[matched_djf_indices, "toa_cre_net_mon_detrend_deseason"]
    lts_djf_matched = lts_djf[matched_djf_indices, "LTS_lag_0"]
    
    # First pass: collect all DJF ENSO state data to determine axis limits for this region
    djf_enso_lts_values = Float64[]
    djf_enso_net_cre_values = Float64[]
    
    for state in enso_state_names
        state_indices = matched_djf_enso_states .== state
        net_cre_state = net_cre_djf_matched[state_indices]
        lts_state = lts_djf_matched[state_indices]
        
        valid_indices = .!(ismissing.(net_cre_state) .| ismissing.(lts_state))
        if sum(valid_indices) > 2  # Need at least 2 points for plotting
            append!(djf_enso_lts_values, lts_state[valid_indices])
            append!(djf_enso_net_cre_values, net_cre_state[valid_indices])
        end
    end
    
    # Calculate axis limits for this region's DJF ENSO state plots
    if !isempty(djf_enso_lts_values) && !isempty(djf_enso_net_cre_values)
        djf_enso_lts_xlims = (minimum(djf_enso_lts_values), maximum(djf_enso_lts_values))
        djf_enso_net_cre_ylims = (minimum(djf_enso_net_cre_values), maximum(djf_enso_net_cre_values))
    else
        djf_enso_lts_xlims = (0, 1)  # Default if no data
        djf_enso_net_cre_ylims = (0, 1)
    end
    
    println("DJF ENSO state axis limits for $region:")
    println("  LTS xlims: $(djf_enso_lts_xlims)")
    println("  Net CRE ylims: $(djf_enso_net_cre_ylims)")
    
    # Create 3-pane plot for DJF ENSO states (El Niño, La Niña, Neutral)
    djf_enso_plots = Array{Any}(undef, 1, 3)  # 1x3 grid
    
    for (i, state) in enumerate(enso_state_names)
        # Filter data for this ENSO state
        state_indices = matched_djf_enso_states .== state
        net_cre_state = net_cre_djf_matched[state_indices]
        lts_state = lts_djf_matched[state_indices]
        
        # Check for valid data
        valid_indices = .!(ismissing.(net_cre_state) .| ismissing.(lts_state))
        
        if sum(valid_indices) > 2  # Need at least 2 points for meaningful analysis
            net_cre_valid = net_cre_state[valid_indices]
            lts_valid = lts_state[valid_indices]
            
            if length(lts_valid) > 2  # Need at least 3 points for correlation
                correlation = cor(net_cre_valid, lts_valid)
                spearman_corr = corspearman(net_cre_valid, lts_valid)
                
                p = scatter(lts_valid, net_cre_valid,
                           title = "DJF $state (n=$(length(lts_valid)))\nPearson's r = $(round(correlation, digits=3)), Spearman's ρ = $(round(spearman_corr, digits=3))",
                           xlabel = "LTS (K)",
                           ylabel = "Net CRE (W/m²)",
                           alpha = 0.6,
                           color = enso_colors[i],
                           markersize = 4,
                           size = (400, 300),
                           titlefontsize = 8, label = "",
                           xlims = djf_enso_lts_xlims, ylims = djf_enso_net_cre_ylims)
                
                # Add trend line
                if length(lts_valid) > 1
                    x_mean = mean(lts_valid)
                    y_mean = mean(net_cre_valid)
                    slope = sum((lts_valid .- x_mean) .* (net_cre_valid .- y_mean)) / sum((lts_valid .- x_mean).^2)
                    intercept = y_mean - slope * x_mean
                    lts_range = range(minimum(lts_valid), maximum(lts_valid), length=100)
                    trend_line = intercept .+ slope .* lts_range
                    plot!(p, lts_range, trend_line, color = :black, linewidth = 2, label = "")
                end
            else
                # If only 1-2 points, show scatter without correlation
                p = scatter(lts_valid, net_cre_valid,
                           title = "DJF $state (n=$(length(lts_valid)))\nInsufficient data for correlation",
                           xlabel = "LTS (K)",
                           ylabel = "Net CRE (W/m²)",
                           alpha = 0.6,
                           color = enso_colors[i],
                           markersize = 4,
                           size = (400, 300),
                           titlefontsize = 8, label = "",
                           xlims = djf_enso_lts_xlims, ylims = djf_enso_net_cre_ylims)
            end
            
            djf_enso_plots[1, i] = p
        else
            djf_enso_plots[1, i] = plot(title = "DJF $state - No data", size = (400, 300), titlefontsize = 8,
                                       xlims = djf_enso_lts_xlims, ylims = djf_enso_net_cre_ylims)
        end
    end
    
    # Create final DJF ENSO state plot for this region
    djf_enso_final = plot(djf_enso_plots..., layout = (1, 3), size = (1200, 400),
                         plot_title = "$region - DJF LTS vs Net CRE by ENSO State (±0.5K threshold)", titlefontsize = 10)
    
    # Save the DJF ENSO state plot
    savefig(djf_enso_final, joinpath(visdir, "LTS_vs_NetCRE_DJF_ENSO_states_$(region).png"))
    println("DJF ENSO state plot saved for $region: $(joinpath(visdir, "LTS_vs_NetCRE_DJF_ENSO_states_$(region).png"))")
end

println("All DJF ENSO state analysis complete!")

