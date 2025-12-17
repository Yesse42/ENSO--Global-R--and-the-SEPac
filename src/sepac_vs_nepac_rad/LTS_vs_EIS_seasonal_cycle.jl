"""
This script creates a scatterplot showing the seasonal cycle of LTS vs EIS for different regions.
- LTS on y-axis, EIS on x-axis
- Different colors for each region
- DJF: empty circles (outline only)
- JJA: filled circles
- MAM/SON: half-filled circles
- Arrows connect the seasons in order (DJF -> MAM -> JJA -> SON -> DJF)
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using CSV, DataFrames, Dates, JLD2, Statistics
using Plots, Colors

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/compare_stratocumulus_regions/seasonal_cycle_plots"

# Create the visualization directory if it doesn't exist
mkpath(visdir)

date_range = (Date(2000, 3), Date(2022, 3, 31))
is_analysis_time(t) = in_time_period(t, date_range)

# Create scatterplot array
gr()

# Define region names - get all available regions from the mask file
mask_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2"
region_data = JLD2.load(mask_file)
all_regions = sort(collect(keys(region_data["regional_masks_ceres"])))

# Define seasons and their plotting order
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

# Define colors for each region (using a colormap that will work for all regions)
region_colors = palette(:tab10, length(all_regions))

println("Found $(length(all_regions)) regions: $(all_regions)")

# Data structure to store seasonal means for each region
regional_seasonal_data = Dict()

# First pass: collect seasonal mean data for each region
for (region_idx, region) in enumerate(all_regions)
    println("Processing region: $region")
    
    # Load ERA5 data for this region
    era5_datafile = joinpath(datadir, "era5_region_avg_$(region).csv")
    if !isfile(era5_datafile)
        println("Warning: File not found: $era5_datafile")
        continue
    end
    
    era5_df = CSV.read(era5_datafile, DataFrame)
    filter!(row -> is_analysis_time(row.date), era5_df)
    
    # Add season column
    era5_df[!, :season] = get_season.(month.(era5_df[!, :date]))
    
    # Calculate seasonal means
    seasonal_means = Dict()
    
    for season in seasons
        season_data = filter(row -> row.season == season, era5_df)
        
        if nrow(season_data) > 0
            # Get LTS and EIS data for this season
            lts_values = season_data[!, "LTS"]
            eis_values = season_data[!, "EIS"]
            
            # Remove missing values
            valid_indices = .!(ismissing.(lts_values) .| ismissing.(eis_values))
            
            if sum(valid_indices) > 0
                lts_clean = lts_values[valid_indices]
                eis_clean = eis_values[valid_indices]
                
                seasonal_means[season] = Dict(
                    "LTS" => mean(lts_clean),
                    "EIS" => mean(eis_clean),
                    "n_points" => length(lts_clean)
                )
            end
        end
    end
    
    # Only store regions that have data for all 4 seasons
    if length(seasonal_means) == 4
        regional_seasonal_data[region] = seasonal_means
        println("  Successfully processed all seasons for $region")
    else
        println("  Warning: Missing seasonal data for $region (only $(length(seasonal_means))/4 seasons)")
    end
end

println("Successfully processed $(length(regional_seasonal_data)) regions with complete seasonal data")

# Calculate global axis limits
all_lts_values = Float64[]
all_eis_values = Float64[]

for (region, seasonal_data) in regional_seasonal_data
    for season in seasons
        if haskey(seasonal_data, season)
            push!(all_lts_values, seasonal_data[season]["LTS"])
            push!(all_eis_values, seasonal_data[season]["EIS"])
        end
    end
end

if !isempty(all_lts_values) && !isempty(all_eis_values)
    # Add some padding to the limits
    lts_range = maximum(all_lts_values) - minimum(all_lts_values)
    eis_range = maximum(all_eis_values) - minimum(all_eis_values)
    
    lts_ylims = (minimum(all_lts_values) - 0.1 * lts_range, maximum(all_lts_values) + 0.1 * lts_range)
    eis_xlims = (minimum(all_eis_values) - 0.1 * eis_range, maximum(all_eis_values) + 0.1 * eis_range)
    
    println("Global axis limits:")
    println("  EIS xlims: $(eis_xlims)")
    println("  LTS ylims: $(lts_ylims)")
else
    lts_ylims = (0, 20)
    eis_xlims = (0, 15)
    println("Warning: No data found, using default axis limits")
end

# Create the seasonal cycle plot
p = plot(size = (1000, 800), 
         xlabel = "EIS (K)", 
         ylabel = "LTS (K)", 
         title = "Seasonal Cycle: LTS vs EIS for Different Stratocumulus Regions",
         titlefontsize = 14,
         xlims = eis_xlims,
         ylims = lts_ylims,
         legend = :outertopright,
         legendtitlefontsize = 10,
         legendfontsize = 8)

# Plot data for each region
for (region_idx, region) in enumerate(sort(collect(keys(regional_seasonal_data))))
    seasonal_data = regional_seasonal_data[region]
    region_color = region_colors[region_idx]
    region != "SEPac" && continue
    
    # Extract seasonal data in order
    eis_seasonal = Float64[]
    lts_seasonal = Float64[]
    
    for season in seasons
        push!(eis_seasonal, seasonal_data[season]["EIS"])
        push!(lts_seasonal, seasonal_data[season]["LTS"])
    end
    
    # Plot different marker styles for each season
    for (season_idx, season) in enumerate(seasons)
        eis_val = eis_seasonal[season_idx]
        lts_val = lts_seasonal[season_idx]
        
        if season == "DJF"
            # Empty circle (outline only)
            scatter!([eis_val], [lts_val], 
                    color = :white, 
                    markerstrokecolor = region_color,
                    markerstrokewidth = 2,
                    markersize = 8,
                    label = season_idx == 1 ? region : "")
        elseif season == "JJA"
            # Filled circle
            scatter!([eis_val], [lts_val], 
                    color = region_color, 
                    markerstrokecolor = region_color,
                    markerstrokewidth = 1,
                    markersize = 8,
                    label = season_idx == 1 ? region : "")
        else  # MAM or SON - half-filled circles
            # Create half-filled effect by plotting a filled circle with a white overlay
            scatter!([eis_val], [lts_val], 
                    color = region_color, 
                    markerstrokecolor = region_color,
                    markerstrokewidth = 1,
                    markersize = 8,
                    alpha = 0.5,
                    label = season_idx == 1 ? region : "")
        end
    end
    
    # Add arrows connecting the seasons in order (DJF -> MAM -> JJA -> SON -> DJF)
    for i in 1:4
        start_idx = i
        end_idx = i == 4 ? 1 : i + 1  # Connect SON back to DJF
        
        start_eis = eis_seasonal[start_idx]
        start_lts = lts_seasonal[start_idx]
        end_eis = eis_seasonal[end_idx]
        end_lts = lts_seasonal[end_idx]
        
        # Calculate arrow direction
        arrow_eis = end_eis - start_eis
        arrow_lts = end_lts - start_lts
        
        # Make arrows slightly shorter so they don't overlap markers
        arrow_length = 0.8
        arrow_start_eis = start_eis + 0.1 * arrow_eis
        arrow_start_lts = start_lts + 0.1 * arrow_lts
        arrow_end_eis = start_eis + arrow_length * arrow_eis
        arrow_end_lts = start_lts + arrow_length * arrow_lts
        
        # Plot arrow
        plot!([arrow_start_eis, arrow_end_eis], [arrow_start_lts, arrow_end_lts],
              arrow = true,
              color = region_color,
              linewidth = 2,
              alpha = 0.7,
              label = "")
    end
    
    # Add season labels near the points for the first region (to avoid clutter)
    if region_idx == 1
        for (season_idx, season) in enumerate(seasons)
            eis_val = eis_seasonal[season_idx]
            lts_val = lts_seasonal[season_idx]
            
            # Offset the labels slightly to avoid overlap
            annotate!(eis_val + 0.1, lts_val + 0.1, text(season, 8, :left, :bottom))
        end
    end
end

# Add legend explanation for marker styles
plot!([], [], color = :white, markerstrokecolor = :black, markerstrokewidth = 2, 
      markersize = 8, label = "DJF (outline)", linestyle = :solid)
plot!([], [], color = :gray, markerstrokecolor = :gray, markerstrokewidth = 1, 
      markersize = 8, alpha = 0.5, label = "MAM/SON (half-filled)", linestyle = :solid)
plot!([], [], color = :black, markerstrokecolor = :black, markerstrokewidth = 1, 
      markersize = 8, label = "JJA (filled)", linestyle = :solid)

# Save the plot
savefig(joinpath(visdir, "LTS_vs_EIS_seasonal_cycle_all_regions.png"))
println("Seasonal cycle plot saved to: $(joinpath(visdir, "LTS_vs_EIS_seasonal_cycle_all_regions.png"))")

# Also create a version with just the most important regions (SEPac and NEPac)
important_regions = ["SEPac", "NEPac", "SEPac_feedback_only"]
filtered_regional_data = Dict()

for region in important_regions
    if haskey(regional_seasonal_data, region)
        filtered_regional_data[region] = regional_seasonal_data[region]
    end
end

if !isempty(filtered_regional_data)
    println("Creating focused plot with $(length(filtered_regional_data)) key regions: $(collect(keys(filtered_regional_data)))")
    
    # Create focused plot
    p_focused = plot(size = (800, 600), 
                    xlabel = "EIS (K)", 
                    ylabel = "LTS (K)", 
                    title = "Seasonal Cycle: LTS vs EIS for Key Pacific Stratocumulus Regions",
                    titlefontsize = 14,
                    xlims = eis_xlims,
                    ylims = lts_ylims,
                    legend = :topright,
                    legendfontsize = 10)
    
    # Define specific colors for the key regions
    key_colors = Dict(
        "SEPac" => :red,
        "NEPac" => :blue, 
        "SEPac_feedback_only" => :green
    )
    
    # Plot data for key regions only
    for region in sort(collect(keys(filtered_regional_data)))
        seasonal_data = filtered_regional_data[region]
        region_color = key_colors[region]
        
        # Extract seasonal data in order
        eis_seasonal = Float64[]
        lts_seasonal = Float64[]
        
        for season in seasons
            push!(eis_seasonal, seasonal_data[season]["EIS"])
            push!(lts_seasonal, seasonal_data[season]["LTS"])
        end
        
        # Plot different marker styles for each season
        for (season_idx, season) in enumerate(seasons)
            eis_val = eis_seasonal[season_idx]
            lts_val = lts_seasonal[season_idx]
            
            if season == "DJF"
                # Empty circle (outline only)
                scatter!([eis_val], [lts_val], 
                        color = :white, 
                        markerstrokecolor = region_color,
                        markerstrokewidth = 3,
                        markersize = 10,
                        label = season_idx == 1 ? region : "")
            elseif season == "JJA"
                # Filled circle
                scatter!([eis_val], [lts_val], 
                        color = region_color, 
                        markerstrokecolor = region_color,
                        markerstrokewidth = 2,
                        markersize = 10,
                        label = season_idx == 1 ? region : "")
            else  # MAM or SON - half-filled circles
                # Create half-filled effect
                scatter!([eis_val], [lts_val], 
                        color = region_color, 
                        markerstrokecolor = region_color,
                        markerstrokewidth = 2,
                        markersize = 10,
                        alpha = 0.5,
                        label = season_idx == 1 ? region : "")
            end
        end
        
        # Add arrows connecting the seasons
        for i in 1:4
            start_idx = i
            end_idx = i == 4 ? 1 : i + 1
            
            start_eis = eis_seasonal[start_idx]
            start_lts = lts_seasonal[start_idx]
            end_eis = eis_seasonal[end_idx]
            end_lts = lts_seasonal[end_idx]
            
            arrow_eis = end_eis - start_eis
            arrow_lts = end_lts - start_lts
            
            arrow_length = 0.8
            arrow_start_eis = start_eis + 0.15 * arrow_eis
            arrow_start_lts = start_lts + 0.15 * arrow_lts
            arrow_end_eis = start_eis + arrow_length * arrow_eis
            arrow_end_lts = start_lts + arrow_length * arrow_lts
            
            plot!([arrow_start_eis, arrow_end_eis], [arrow_start_lts, arrow_end_lts],
                  arrow = true,
                  color = region_color,
                  linewidth = 3,
                  alpha = 0.8,
                  label = "")
        end
        
        # Add season labels for each region
        for (season_idx, season) in enumerate(seasons)
            eis_val = eis_seasonal[season_idx]
            lts_val = lts_seasonal[season_idx]
            
            # Offset labels based on region to avoid overlap
            if region == "SEPac"
                offset_eis, offset_lts = 0.2, 0.2
            elseif region == "NEPac"
                offset_eis, offset_lts = -0.3, 0.2
            else  # SEPac_feedback_only
                offset_eis, offset_lts = 0.2, -0.3
            end
            
            annotate!(eis_val + offset_eis, lts_val + offset_lts, 
                     text(season, 9, :center, region_color))
        end
    end
    
    # Add marker style legend
    plot!([], [], color = :white, markerstrokecolor = :black, markerstrokewidth = 3, 
          markersize = 10, label = "DJF", linestyle = :solid)
    plot!([], [], color = :gray, markerstrokecolor = :gray, markerstrokewidth = 2, 
          markersize = 10, alpha = 0.5, label = "MAM/SON", linestyle = :solid)
    plot!([], [], color = :black, markerstrokecolor = :black, markerstrokewidth = 2, 
          markersize = 10, label = "JJA", linestyle = :solid)
    
    # Save the focused plot
    savefig(p_focused, joinpath(visdir, "LTS_vs_EIS_seasonal_cycle_key_regions.png"))
    println("Focused seasonal cycle plot saved to: $(joinpath(visdir, "LTS_vs_EIS_seasonal_cycle_key_regions.png"))")
end

# Print summary statistics
println("\n=== SUMMARY ===")
println("Seasonal cycle analysis complete!")
println("Processed $(length(regional_seasonal_data)) regions total")
println("Key findings:")

for (region, seasonal_data) in sort(regional_seasonal_data)
    println("\n$region:")
    for season in seasons
        data = seasonal_data[season]
        println("  $season: LTS = $(round(data["LTS"], digits=2)) K, EIS = $(round(data["EIS"], digits=2)) K (n=$(data["n_points"]))")
    end
    
    # Calculate seasonal ranges
    lts_values = [seasonal_data[season]["LTS"] for season in seasons]
    eis_values = [seasonal_data[season]["EIS"] for season in seasons]
    
    lts_range = maximum(lts_values) - minimum(lts_values)
    eis_range = maximum(eis_values) - minimum(eis_values)
    
    println("  Seasonal ranges: LTS = $(round(lts_range, digits=2)) K, EIS = $(round(eis_range, digits=2)) K")
end

println("\nPlots saved to: $visdir")