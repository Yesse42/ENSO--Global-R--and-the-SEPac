"""
This script replicates the Wang et al. analysis comparing gridded fields
between normal and elevated SAM years across different temporal averaging periods.
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

using CSV, DataFrames, Dates, Dictionaries, PythonCall, Statistics
@py import matplotlib.pyplot as plt, cartopy.crs as ccrs, matplotlib.colors as colors, cmasher as cmr

# ============================================================================
# PARAMETERS
# ============================================================================

# Analysis time range
ANALYSIS_PERIOD = (Date(1980, 1, 1), Date(2020, 12, 31))

# ============================================================================
# LOAD DATA
# ============================================================================

println("Loading SAM time series...")
sam_df = CSV.read("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/SAM/calculated_sam.csv", DataFrame)
sam_df[!, :date] = Date.(sam_df.year, sam_df.month, sam_df.day)

# Filter SAM data to analysis period
sam_df = filter(row -> ANALYSIS_PERIOD[1] <= row.date < ANALYSIS_PERIOD[2], sam_df)
sort!(sam_df, :date)

println("Loading gridded fields (SST, MSL, Z500)...")

# Load SST (sea surface temperature)
sst_data, sst_coords = load_era5_data(["sst"], ANALYSIS_PERIOD)
sst = sst_data["sst"]
sst_lat = sst_coords["latitude"]
sst_lon = sst_coords["longitude"]
sst_time = sst_coords["time"]

# Load MSL (mean sea level pressure)
msl_data, msl_coords = load_era5_data(["msl"], ANALYSIS_PERIOD)
msl = msl_data["msl"]
msl_lat = msl_coords["latitude"]
msl_lon = msl_coords["longitude"]
msl_time = msl_coords["time"]

# Load Z (geopotential height at pressure levels, then index for 500 hPa)
z_data, z_coords = load_era5_data(["z"], ANALYSIS_PERIOD)
z_full = z_data["z"]
pressure_levels = z_coords["pressure_level"]
z_lat = z_coords["latitude"]
z_lon = z_coords["longitude"]
z_time = z_coords["pressure_time"]

# Find index for 500 hPa
z500_idx = findfirst(pressure_levels .== 500)
if z500_idx === nothing
    error("500 hPa level not found in pressure levels: $pressure_levels")
end

# Extract Z500 (lon, lat, time)
z500 = z_full[:, :, z500_idx, :]

# Convert all missings to NaNs
println("\nConverting missing values to NaNs...")
sst = replace(sst, missing => NaN)
msl = replace(msl, missing => NaN)
z500 = replace(z500, missing => NaN)

println("Loaded dimensions:")
println("  SST: ", size(sst))
println("  MSL: ", size(msl))
println("  Z500: ", size(z500))

# ============================================================================
# DESEASONALIZE GRIDDED FIELDS
# ============================================================================

println("\nDeseasonalizing gridded fields...")

# Get month information for each field
sst_months = month.(sst_time)
msl_months = month.(msl_time)
z_months = month.(z_time)

# Process SST
println("  Deseasonalizing SST...")
for slice in eachslice(sst; dims = (1,2))
    deseasonalize!(slice, sst_months)
end

# Process MSL
println("  Deseasonalizing MSL...")
for slice in eachslice(msl; dims = (1,2))
    deseasonalize!(slice, msl_months)
end

# Process Z500
println("  Deseasonalizing Z500...")
for slice in eachslice(z500; dims = (1,2))
    deseasonalize!(slice, z_months)
end

println("  Deseasonalization complete!")

# ============================================================================
# CALCULATE ANNUAL SAM AVERAGES
# ============================================================================

println("\nCalculating annual SAM averages (January-December)...")

# Extract years
sam_df[!, :year_only] = year.(sam_df.date)
years = sort(unique(sam_df.year_only))

# Calculate annual averages
annual_sam = DataFrame(year = Int[], sam_annual = Float64[])

for yr in years
    year_data = filter(row -> row.year_only == yr, sam_df)

    # Only include complete years (12 months)
    if nrow(year_data) == 12
        sam_avg = mean(year_data.sam_calculated)
        push!(annual_sam, (year = yr, sam_annual = sam_avg))
    end
end

println("Annual SAM averages calculated for $(nrow(annual_sam)) years")

# ============================================================================
# IDENTIFY NORMAL AND ELEVATED SAM YEARS
# ============================================================================

println("\nIdentifying normal and elevated SAM years...")

# Calculate percentiles
sam_values = annual_sam.sam_annual
p40 = quantile(sam_values, 0.40)
p60 = quantile(sam_values, 0.60)
p90 = quantile(sam_values, 0.90)

println("  40th percentile: $p40")
println("  60th percentile: $p60")
println("  90th percentile: $p90")

# Identify normal years (40th-60th percentile)
normal_years = annual_sam.year[p40 .<= annual_sam.sam_annual .<= p60]
println("  Normal SAM years: $normal_years")

# Identify elevated years (90th percentile or above)
elevated_years = annual_sam.year[annual_sam.sam_annual .>= p90]
println("  Elevated SAM years: $elevated_years")

# ============================================================================
# HELPER FUNCTION FOR TEMPORAL AVERAGING
# ============================================================================

"""
Calculate temporal average of gridded field for specified months and years.
Also standardizes by the climatology of those specific months.
"""
function calculate_temporal_average_standardized(field, time_vec, months_range, target_years)
    # Create mask for desired months and years
    field_years = year.(time_vec)
    field_months = month.(time_vec)

    # Mask for the target years and months
    target_mask = falses(length(time_vec))
    for i in 1:length(time_vec)
        if field_years[i] in target_years && field_months[i] in months_range
            target_mask[i] = true
        end
    end

    if sum(target_mask) == 0
        @warn "No data found for specified months and years"
        return fill(NaN, size(field, 1), size(field, 2))
    end

    # Mask for ALL years but same months (for climatology)
    climatology_mask = falses(length(time_vec))
    for i in 1:length(time_vec)
        if field_months[i] in months_range
            climatology_mask[i] = true
        end
    end

    # Calculate climatological mean and std for these months across all years
    climatology_mean = mean(field[:, :, climatology_mask], dims=3)[:, :, 1]
    climatology_std = std(field[:, :, climatology_mask], dims=3)[:, :, 1]

    # Get the data for target years/months and standardize
    target_data = field[:, :, target_mask]

    # Standardize each time slice
    standardized_data = similar(target_data)
    for t in 1:size(target_data, 3)
        standardized_data[:, :, t] = (target_data[:, :, t] .- climatology_mean) ./ climatology_std
    end

    # Average the standardized data over time
    return mean(standardized_data, dims=3)[:, :, 1]
end

# ============================================================================
# CALCULATE TEMPORAL AVERAGES FOR EACH GRIDDED FIELD
# ============================================================================

println("\nCalculating temporal averages for each gridded field...")

# Initialize dictionaries to store results
normal_averages = Dictionary{String, Array{Float64, 2}}()
elevated_averages = Dictionary{String, Array{Float64, 2}}()

gridded_fields = ["SST", "MSL", "Z500"]
field_data = [sst, msl, z500]
field_times = [sst_time, msl_time, z_time]

for (field_name, field, field_time) in zip(gridded_fields, field_data, field_times)
    println("\n  Processing $field_name...")

    # Period 1: Sep-Dec immediately preceding the year
    println("    Period 1: Sep-Dec preceding")
    preceding_years = [yr - 1 for yr in normal_years]
    normal_p1 = calculate_temporal_average_standardized(field, field_time, 9:12, preceding_years)
    set!(normal_averages, "$(field_name)_period1", normal_p1)

    preceding_years_elevated = [yr - 1 for yr in elevated_years]
    elevated_p1 = calculate_temporal_average_standardized(field, field_time, 9:12, preceding_years_elevated)
    set!(elevated_averages, "$(field_name)_period1", elevated_p1)

    # Period 2: Jan-Apr of the year immediately following
    println("    Period 2: Jan-Apr of following year")
    following_years = [yr + 1 for yr in normal_years]
    normal_p2 = calculate_temporal_average_standardized(field, field_time, 1:4, following_years)
    set!(normal_averages, "$(field_name)_period2", normal_p2)

    following_years_elevated = [yr + 1 for yr in elevated_years]
    elevated_p2 = calculate_temporal_average_standardized(field, field_time, 1:4, following_years_elevated)
    set!(elevated_averages, "$(field_name)_period2", elevated_p2)

    # Period 3: Jan-Apr of the year after the year immediately following
    println("    Period 3: Jan-Apr of year+2")
    following_years2 = [yr + 2 for yr in normal_years]
    normal_p3 = calculate_temporal_average_standardized(field, field_time, 1:4, following_years2)
    set!(normal_averages, "$(field_name)_period3", normal_p3)

    following_years2_elevated = [yr + 2 for yr in elevated_years]
    elevated_p3 = calculate_temporal_average_standardized(field, field_time, 1:4, following_years2_elevated)
    set!(elevated_averages, "$(field_name)_period3", elevated_p3)
end

# ============================================================================
# CREATE 3x3 PLOT: ELEVATED - NORMAL
# ============================================================================

println("\nCreating 3x3 difference plots (Elevated - Normal)...")

# Create output directory
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/SAM/wang_replication"
mkpath(visdir)

# Set fixed colorbar limits
cbarlim = 1.0
println("  Setting colorbar limits to ... ±$cbarlim")
common_colornorm = colors.Normalize(vmin=-cbarlim, vmax=cbarlim)

# Setup figure
fig = plt.figure(figsize=(18, 12))

# Row labels for fields (now rows instead of columns)
field_labels = ["SST", "MSL", "Z500"]

# Column labels for periods (now columns instead of rows)
period_labels = [
    "Sep-Dec (Year-1)",
    "Jan-Apr (Year+1)",
    "Jan-Apr (Year+2)"
]

# Coordinate arrays for plotting
coords_list = [(sst_lon, sst_lat), (msl_lon, msl_lat), (z_lon, z_lat)]

# Plot each combination - FLIPPED: iterate fields in outer loop, periods in inner loop
plot_idx = 1
contour_plots = []
for (field_idx, (field_name, coords)) in enumerate(zip(gridded_fields, coords_list))
    for (period_idx, period) in enumerate(["period1", "period2", "period3"])
        println("  Plotting $field_name - $period")

        # Calculate difference: elevated - normal
        normal_key = "$(field_name)_$(period)"
        elevated_key = "$(field_name)_$(period)"

        difference = elevated_averages[elevated_key] .- normal_averages[normal_key]

        # Create subplot
        global plot_idx
        ax = fig.add_subplot(3, 3, plot_idx, projection=ccrs.Robinson(central_longitude=180))
        ax.set_global()
        ax.coastlines()

        # Plot difference with common colorbar limits
        lon_grid, lat_grid = coords
        c = ax.contourf(lon_grid, lat_grid, difference',
                       transform=ccrs.PlateCarree(),
                       cmap=cmr.prinsenvlag.reversed(),
                       levels=21,
                       norm=common_colornorm)

        # Store first contour plot for colorbar
        if plot_idx == 1
            push!(contour_plots, c)
        end

        # Set titles - column titles for periods
        if field_idx == 1
            ax.set_title(period_labels[period_idx], fontsize=14, fontweight="bold")
        end

        # Row labels for fields
        if period_idx == 1
            ax.text(-0.15, 0.5, field_labels[field_idx],
                   transform=ax.transAxes, fontsize=12, fontweight="bold",
                   rotation=90, va="center", ha="center")
        end

        global plot_idx += 1
    end
end

# Add overall title
fig.suptitle("Elevated SAM Years - Normal SAM Years\n(Gridded Field Differences)",
            fontsize=16, fontweight="bold", y=0.98)

# Add single colorbar for all subplots
cbar = fig.colorbar(contour_plots[1], ax=fig.get_axes(), orientation="horizontal",
                   pad=0.05, shrink=0.6, aspect=30)
cbar.set_label("Standardized Anomaly Difference", fontsize=12, fontweight="bold")

# Save figure
output_path = joinpath(visdir, "sam_elevated_minus_normal_3x3.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
println("\nSaved plot to: $output_path")
plt.close(fig)

println("\nAnalysis complete!")
