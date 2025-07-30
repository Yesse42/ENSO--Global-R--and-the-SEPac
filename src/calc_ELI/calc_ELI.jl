"""
This script aims to calculate the ENSO Longitude Index (ELI) basert på ERA5 sst data

The procedure is as follows:
1. Calculate the mean tropical surface temperature between 5N and 5S
2. Use that as the threshold of convective initiation
3. 
"""

using CSV, DataFrames, Plots, Statistics, StatsBase

visdir = joinpath("../../vis", "ELI")
!isdir(visdir) && mkpath(visdir)

# Save data to ENSO directory since ELI is an ENSO-related index
savedir = "../../data/ENSO/"

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

lonbounds = (140, -75+360)
latbounds = (-5, 5)
times = (Date(0), Date(10000000))

var = "sst"
era_vars = ["sst", "lsm"]
era5_data, era5_coords = load_era5_data(era_vars, times)

sst_data = era5_data["sst"][:,:,:]
lsm = era5_data["lsm"][:,:,1] .== 0

sst_notmissing_mask = mapslices(slice -> !any(ismissing, slice), sst_data; dims = (3))[:,:] .&& lsm

for sst_slice in eachslice(sst_data, dims = 3)
    sst_slice[(!).(sst_notmissing_mask)] .= 0.0
end

lat = era5_coords["latitude"]
lon = era5_coords["longitude"]

in_lonbounds = lonbounds[1] .<= lon .<= lonbounds[2]
in_latbounds = latbounds[1] .<= lat .<= latbounds[2]
pac_region_mask = in_lonbounds .* in_latbounds'

#Now calculate the tropical mean sst
"w must be in the appropriate dimensions for the slice"
function weighted_mean_slice(X, w; dims)
    #The dimensions in dim must have a size in w equal to the size in X
    w_factor = prod(size.(Ref(X), dims))/prod(size(w))
    return sum(X .* w; dims = dims) ./ (sum(w) * w_factor)
end

total_sst_average_mask = in_latbounds' .* sst_notmissing_mask
total_pac_mask = pac_region_mask .* sst_notmissing_mask

reduced_weights = cosd.(lat)'
full_weights = repeat(reduced_weights, size(sst_data, 1), 1)
full_weights_masked = full_weights .* total_sst_average_mask

fig = plot_global_heatmap(era5_coords["latitude"], era5_coords["longitude"], total_sst_average_mask; title = "SST AVG Mask", colorbar_label = "Mask")
fig.savefig(joinpath(visdir, "sst_avg_mask.png"))
plt.close(fig)

tropical_mean_sst = weighted_mean_slice(sst_data .* total_sst_average_mask, full_weights_masked; dims = (1,2))

pac_mask_fig = plot_global_heatmap(era5_coords["latitude"], era5_coords["longitude"], total_pac_mask; title = "Pacific Region Mask", colorbar_label = "Mask")
pac_mask_fig.savefig(joinpath(visdir, "pac_region_mask.png"))
plt.close(pac_mask_fig)

"Now we must calculate the average longitude of the parts of the pacific region that are above the tropical mean sst"
function calc_ELI(sst_data, region_mask, tropical_mean_sst, lat, lon)
    coslat = cosd.(lat)'
    full_lat_weights = repeat(coslat, size(sst_data, 1), 1)

    full_lon = repeat(lon, 1, size(sst_data, 2))

    ELI = zeros(size(sst_data, 3))
    for i in 1:size(sst_data, 3)
        sst_slice = sst_data[:,:,i]
        masked_sst = sst_slice .* region_mask
        sst_greater_than_mean = masked_sst .> tropical_mean_sst[i]
        lon_avg_mask = sst_greater_than_mean .* region_mask

        full_mask_for_this_iteration = lon_avg_mask .* full_lat_weights

        weights_for_this_iteration = full_lat_weights .* full_mask_for_this_iteration

        ELI[i] = only(weighted_mean_slice(full_lon .* full_mask_for_this_iteration, weights_for_this_iteration; dims = (1,2)))
    end

    return ELI
end

ELI = calc_ELI(sst_data, total_pac_mask, tropical_mean_sst, lat, lon)

# Get the time coordinates for the ELI data
eli_dates = era5_coords["time"]

# Create DataFrame with ELI data
println("Creating ELI DataFrame with lags...")
df = DataFrame(Date = eli_dates, ELI = ELI)

# Create ELI time series plot
eli_fig = plot(df.Date, df.ELI, 
               label="ELI", 
               xlabel="Date", 
               ylabel="ELI (degrees longitude)", 
               title="ENSO Longitude Index (ELI) Time Series",
               linewidth=2)
savefig(eli_fig, joinpath(visdir, "eli_timeseries.png"))
display(eli_fig)

# Add lagged versions of ELI
lags = -24:24
for lag in lags
    df[!, Symbol("ELI_Lag$(lag)")] = time_lag(df.ELI, lag)
end

# Save ELI data to CSV in ENSO directory
eli_csv_path = joinpath(savedir, "eli_data.csv")
CSV.write(eli_csv_path, df)
println("Saved ELI data to: $eli_csv_path")

# Compare ELI with ONI at lag 0
println("\nComparing ELI with ONI at lag 0...")

# Load ENSO data for comparison
enso_data, enso_coords = load_enso_data(times; lags=[0])
oni_lag0 = enso_data["oni_lag_0"]
enso_times = enso_coords["time"]

# Get the ELI times and data
eli_times = df.Date
eli_values = df.ELI

# Round dates to nearest month for proper alignment
eli_times_rounded = Date.(year.(eli_times), month.(eli_times), 1)
enso_times_rounded = Date.(year.(enso_times), month.(enso_times), 1)

println("ELI time range: $(minimum(eli_times_rounded)) to $(maximum(eli_times_rounded)) ($(length(eli_times_rounded)) points)")
println("ONI time range: $(minimum(enso_times_rounded)) to $(maximum(enso_times_rounded)) ($(length(enso_times_rounded)) points)")

# Find overlapping time period
overlap_start = max(minimum(eli_times_rounded), minimum(enso_times_rounded))
overlap_end = min(maximum(eli_times_rounded), maximum(enso_times_rounded))

println("Overlap period: $(overlap_start) to $(overlap_end)")

# Filter both datasets to overlapping period
eli_mask = (eli_times_rounded .>= overlap_start) .& (eli_times_rounded .<= overlap_end)
enso_mask = (enso_times_rounded .>= overlap_start) .& (enso_times_rounded .<= overlap_end)

# Check mask sizes
println("ELI mask: $(sum(eli_mask)) true values out of $(length(eli_mask))")
println("ENSO mask: $(sum(enso_mask)) true values out of $(length(enso_mask))")

# Extract overlapping data
eli_overlap = eli_values[eli_mask]
oni_overlap = oni_lag0[enso_mask]
eli_times_overlap = eli_times_rounded[eli_mask]
enso_times_overlap = enso_times_rounded[enso_mask]

# Further align by matching exact dates (in case of slight misalignments)
# Find common dates
common_dates = intersect(eli_times_overlap, enso_times_overlap)
println("Common dates found: $(length(common_dates))")

if length(common_dates) == 0
    println("ERROR: No common dates found between ELI and ONI datasets!")
    println("ELI dates sample: $(eli_times_overlap[1:min(5, end)])")
    println("ONI dates sample: $(enso_times_overlap[1:min(5, end)])")
else
    # Extract data for common dates only
    eli_common_mask = in.(eli_times_overlap, Ref(Set(common_dates)))
    oni_common_mask = in.(enso_times_overlap, Ref(Set(common_dates)))
    
    eli_final = eli_overlap[eli_common_mask]
    oni_final = oni_overlap[oni_common_mask]
    times_final = common_dates
    
    # Sort by date to ensure proper alignment
    sort_idx = sortperm(times_final)
    times_final = times_final[sort_idx]
    eli_final = eli_final[sort_idx]
    oni_final = oni_final[sort_idx]
    
    println("Final aligned data: $(length(eli_final)) points")
    
    # Calculate correlation
    eli_oni_correlation = cor(eli_final, oni_final)
    println("Correlation between ELI and ONI (lag 0): $(round(eli_oni_correlation, digits=4))")

    # Create comparison plot
    comparison_fig = plot(size=(1000, 600))

    # Plot ELI (scaled for comparison)
    eli_scaled = (eli_final .- mean(eli_final)) ./ std(eli_final)
    plot!(comparison_fig, times_final, eli_scaled,
          label="ELI (standardized)", 
          color=:blue, 
          linewidth=2)

    # Plot ONI (scaled for comparison)
    oni_scaled = (oni_final .- mean(oni_final)) ./ std(oni_final)
    plot!(comparison_fig, times_final, oni_scaled,
          label="ONI Lag 0 (standardized)", 
          color=:red, 
          linewidth=2)

    plot!(comparison_fig,
          xlabel="Date",
          ylabel="Standardized Values",
          title="ELI vs ONI (Lag 0) - Correlation: $(round(eli_oni_correlation, digits=3))",
          legend=:topright,
          grid=true)

    savefig(comparison_fig, joinpath(visdir, "eli_oni_comparison.png"))
    display(comparison_fig)

    # Create scatter plot
    scatter_fig = scatter(oni_final, eli_final,
                         xlabel="ONI (Lag 0)",
                         ylabel="ELI (degrees longitude)",
                         title="ELI vs ONI Scatter Plot (r = $(round(eli_oni_correlation, digits=3)))",
                         alpha=0.6,
                         markersize=3,
                         color=:blue)

    # Add trend line
    trend_coeff = [ones(length(oni_final)) oni_final] \ eli_final
    trend_line = trend_coeff[1] .+ trend_coeff[2] .* oni_final
    plot!(scatter_fig, oni_final, trend_line, 
          color=:red, 
          linewidth=2, 
          label="Trend line")

    savefig(scatter_fig, joinpath(visdir, "eli_oni_scatter.png"))
    display(scatter_fig)

    # Save comparison statistics
    stats_file = joinpath(visdir, "eli_oni_comparison_stats.txt")
    open(stats_file, "w") do io
        println(io, "ELI vs ONI (Lag 0) Comparison Statistics")
        println(io, "=" ^ 50)
        println(io, "Time period: $(overlap_start) to $(overlap_end)")
        println(io, "Number of overlapping observations: $(length(eli_final))")
        println(io, "")
        println(io, "ELI Statistics:")
        println(io, "  Mean: $(round(mean(eli_final), digits=3)) degrees")
        println(io, "  Std:  $(round(std(eli_final), digits=3)) degrees")
        println(io, "  Min:  $(round(minimum(eli_final), digits=3)) degrees")
        println(io, "  Max:  $(round(maximum(eli_final), digits=3)) degrees")
        println(io, "")
        println(io, "ONI (Lag 0) Statistics:")
        println(io, "  Mean: $(round(mean(oni_final), digits=3))")
        println(io, "  Std:  $(round(std(oni_final), digits=3))")
        println(io, "  Min:  $(round(minimum(oni_final), digits=3))")
        println(io, "  Max:  $(round(maximum(oni_final), digits=3))")
        println(io, "")
        println(io, "Correlation: $(round(eli_oni_correlation, digits=4))")
        println(io, "")
        println(io, "Linear regression: ELI = $(round(trend_coeff[1], digits=3)) + $(round(trend_coeff[2], digits=3)) * ONI")
        println(io, "")
        println(io, "Note: ELI measures the longitude centroid of warm SST anomalies")
        println(io, "in the tropical Pacific, providing information about the")
        println(io, "east-west displacement of convective activity during ENSO events.")
    end

    println("Saved comparison statistics to: $stats_file")

end

# Compare with original ELI from CSV file
println("\nComparing calculated ELI with original ELI from CSV file...")

# Load original ELI data
original_eli_path = joinpath(savedir, "ELI_Original.csv")
if isfile(original_eli_path)
    try
        # Read the CSV file - this has a special structure (years as columns, months as rows)
        raw_df = CSV.read(original_eli_path, DataFrame; header=1)
        
        println("Original ELI raw DataFrame size: ", size(raw_df))
        println("Original ELI raw DataFrame columns: ", names(raw_df)[1:5], "...")  # Show first 5 columns
        
        # Extract years from column names (skip the first column which contains month names)
        year_cols = names(raw_df)[2:end]  # Skip first column (month names)
        
        # Parse years from column names (remove quotes and commas, convert to integers)
        years = []
        for col in year_cols
            # Remove quotes and commas, then parse
            year_str = replace(string(col), "\"" => "", "," => "")
            try
                push!(years, parse(Int, year_str))
            catch
                println("Warning: Could not parse year from column: $col")
            end
        end
        
        println("Years found: $(minimum(years)) to $(maximum(years)) ($(length(years)) years)")
        
        # Get month names from first column
        months = raw_df[!, 1]
        println("Months found: ", months)
        
        # Reconstruct time series in standard format (Date, ELI)
        dates = Date[]
        eli_values = Float64[]
        
        for (year_idx, year) in enumerate(years)
            for (month_idx, month_name) in enumerate(months)
                # Convert month name to month number
                month_num = findfirst(==(month_name), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
                if month_num !== nothing
                    try
                        date = Date(year, month_num, 1)
                        # Get ELI value from the matrix (row = month, column = year+1 because first col is month names)
                        eli_val = raw_df[month_idx, year_idx + 1]
                        
                        # Only add if the value is not missing
                        if !ismissing(eli_val) && !isnan(eli_val)
                            push!(dates, date)
                            push!(eli_values, Float64(eli_val))
                        end
                    catch e
                        println("Warning: Error processing $month_name $year: $e")
                    end
                end
            end
        end
        
        println("Reconstructed $(length(dates)) ELI data points")
        println("Original ELI time range: $(minimum(dates)) to $(maximum(dates))")
        
        # Convert to standard format
        original_dates = dates
        original_eli_values = eli_values
        
        # Round original dates to nearest month for comparison
        original_dates_rounded = Date.(year.(original_dates), month.(original_dates), 1)
        
        println("Original ELI time range: $(minimum(original_dates_rounded)) to $(maximum(original_dates_rounded)) ($(length(original_dates_rounded)) points)")
        
        # Get calculated ELI data (rounded)
        calc_eli_times_rounded = Date.(year.(df.Date), month.(df.Date), 1)
        calc_eli_values = df.ELI
        
        # Find overlapping time period
        overlap_start_orig = max(minimum(calc_eli_times_rounded), minimum(original_dates_rounded))
        overlap_end_orig = min(maximum(calc_eli_times_rounded), maximum(original_dates_rounded))
        
        println("ELI comparison overlap period: $(overlap_start_orig) to $(overlap_end_orig)")
        
        # Filter both datasets to overlapping period
        calc_mask = (calc_eli_times_rounded .>= overlap_start_orig) .& (calc_eli_times_rounded .<= overlap_end_orig)
        orig_mask = (original_dates_rounded .>= overlap_start_orig) .& (original_dates_rounded .<= overlap_end_orig)
        
        calc_eli_overlap = calc_eli_values[calc_mask]
        orig_eli_overlap = original_eli_values[orig_mask]
        calc_times_overlap = calc_eli_times_rounded[calc_mask]
        orig_times_overlap = original_dates_rounded[orig_mask]
        
        # Find common dates
        common_dates_eli = intersect(calc_times_overlap, orig_times_overlap)
        println("Common dates for ELI comparison: $(length(common_dates_eli))")
        
        if length(common_dates_eli) > 0
            # Extract data for common dates only
            calc_eli_common_mask = in.(calc_times_overlap, Ref(Set(common_dates_eli)))
            orig_eli_common_mask = in.(orig_times_overlap, Ref(Set(common_dates_eli)))
            
            calc_eli_final = calc_eli_overlap[calc_eli_common_mask]
            orig_eli_final = orig_eli_overlap[orig_eli_common_mask]
            eli_times_final = common_dates_eli
            
            # Sort by date to ensure proper alignment
            sort_idx_eli = sortperm(eli_times_final)
            eli_times_final = eli_times_final[sort_idx_eli]
            calc_eli_final = calc_eli_final[sort_idx_eli]
            orig_eli_final = orig_eli_final[sort_idx_eli]
            
            println("Final aligned ELI data: $(length(calc_eli_final)) points")
            
            # Calculate correlation
            eli_correlation = cor(calc_eli_final, orig_eli_final)
            println("Correlation between calculated and original ELI: $(round(eli_correlation, digits=4))")
            
            # Create comparison plot
            eli_comparison_fig = plot(size=(1000, 600))
            
            plot!(eli_comparison_fig, eli_times_final, calc_eli_final,
                  label="Calculated ELI", 
                  color=:blue, 
                  linewidth=2)
            
            plot!(eli_comparison_fig, eli_times_final, orig_eli_final,
                  label="Original ELI", 
                  color=:red, 
                  linewidth=2,
                  linestyle=:dash)
            
            plot!(eli_comparison_fig,
                  xlabel="Date",
                  ylabel="ELI (degrees longitude)",
                  title="Calculated vs Original ELI - Correlation: $(round(eli_correlation, digits=3))",
                  legend=:topright,
                  grid=true)
            
            savefig(eli_comparison_fig, joinpath(visdir, "eli_calculated_vs_original.png"))
            display(eli_comparison_fig)
            
            # Create scatter plot
            eli_scatter_fig = scatter(orig_eli_final, calc_eli_final,
                                     xlabel="Original ELI (degrees longitude)",
                                     ylabel="Calculated ELI (degrees longitude)",
                                     title="Calculated vs Original ELI Scatter Plot (r = $(round(eli_correlation, digits=3)))",
                                     alpha=0.6,
                                     markersize=3,
                                     color=:blue)
            
            # Add trend line
            eli_trend_coeff = [ones(length(orig_eli_final)) orig_eli_final] \ calc_eli_final
            eli_trend_line = eli_trend_coeff[1] .+ eli_trend_coeff[2] .* orig_eli_final
            plot!(eli_scatter_fig, orig_eli_final, eli_trend_line, 
                  color=:red, 
                  linewidth=2, 
                  label="Trend line")
            
            # Add 1:1 line for reference
            eli_range = [minimum([orig_eli_final; calc_eli_final]), maximum([orig_eli_final; calc_eli_final])]
            plot!(eli_scatter_fig, eli_range, eli_range, 
                  color=:black, 
                  linewidth=1, 
                  linestyle=:dot,
                  label="1:1 line")
            
            savefig(eli_scatter_fig, joinpath(visdir, "eli_calculated_vs_original_scatter.png"))
            display(eli_scatter_fig)
            
            # Save ELI comparison statistics
            eli_stats_file = joinpath(visdir, "eli_calculated_vs_original_stats.txt")
            open(eli_stats_file, "w") do io
                println(io, "Calculated vs Original ELI Comparison Statistics")
                println(io, "=" ^ 60)
                println(io, "Time period: $(overlap_start_orig) to $(overlap_end_orig)")
                println(io, "Number of overlapping observations: $(length(calc_eli_final))")
                println(io, "")
                println(io, "Calculated ELI Statistics:")
                println(io, "  Mean: $(round(mean(calc_eli_final), digits=3)) degrees")
                println(io, "  Std:  $(round(std(calc_eli_final), digits=3)) degrees")
                println(io, "  Min:  $(round(minimum(calc_eli_final), digits=3)) degrees")
                println(io, "  Max:  $(round(maximum(calc_eli_final), digits=3)) degrees")
                println(io, "")
                println(io, "Original ELI Statistics:")
                println(io, "  Mean: $(round(mean(orig_eli_final), digits=3)) degrees")
                println(io, "  Std:  $(round(std(orig_eli_final), digits=3)) degrees")
                println(io, "  Min:  $(round(minimum(orig_eli_final), digits=3)) degrees")
                println(io, "  Max:  $(round(maximum(orig_eli_final), digits=3)) degrees")
                println(io, "")
                println(io, "Correlation: $(round(eli_correlation, digits=4))")
                println(io, "")
                println(io, "Linear regression: Calculated = $(round(eli_trend_coeff[1], digits=3)) + $(round(eli_trend_coeff[2], digits=3)) * Original")
                println(io, "")
                # Calculate additional metrics
                rmse = sqrt(mean((calc_eli_final .- orig_eli_final).^2))
                mae = mean(abs.(calc_eli_final .- orig_eli_final))
                bias = mean(calc_eli_final .- orig_eli_final)
                println(io, "Error Metrics:")
                println(io, "  RMSE: $(round(rmse, digits=3)) degrees")
                println(io, "  MAE:  $(round(mae, digits=3)) degrees")
                println(io, "  Bias: $(round(bias, digits=3)) degrees")
                println(io, "")
                println(io, "Note: This comparison validates the calculated ELI against")
                println(io, "the original ELI implementation to ensure consistency.")
            end
            
            println("Saved ELI comparison statistics to: $eli_stats_file")
        else
            println("ERROR: No common dates found between calculated and original ELI!")
        end
        
    catch e
        println("ERROR loading original ELI file: ", e)
        println("Please check the CSV file format and column structure.")
    end
else
    println("WARNING: Original ELI file not found at: $original_eli_path")
end
println("\nELI calculation and comparison complete!")




