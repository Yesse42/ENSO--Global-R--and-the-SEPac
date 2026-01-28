"""
This script calculates point-by-point correlations between SAM and gridded fields
across different temporal averaging periods.

MODIFICATION: Instead of comparing high SAM vs normal SAM years, this version
calculates the direct correlation between SAM and each gridded field variable.
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
ANALYSIS_PERIOD = (Date(1980, 1, 1), Date(2024, 12, 31))

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
# HELPER FUNCTION FOR CALCULATING CORRELATIONS
# ============================================================================

"""
Calculate point-by-point correlation between gridded field and SAM values
for specified months and years with temporal offsets.

Arguments:
- field: 3D array (lon, lat, time)
- field_time: vector of dates for field
- sam_years: DataFrame with 'year' and 'sam_annual' columns
- months_range: range of months to consider (e.g., 1:4 for Jan-Apr)
- year_offset: offset from SAM year (e.g., +1 means field in year after SAM year)

Returns:
- 2D array (lon, lat) of correlation coefficients
"""
function calculate_sam_correlation(field, field_time, sam_years, months_range, year_offset)
    n_lon, n_lat, n_time = size(field)

    # Initialize correlation map
    correlation_map = fill(NaN, n_lon, n_lat)

    # Get field years and months
    field_years = year.(field_time)
    field_months = month.(field_time)

    # For each grid point, calculate correlation with SAM
    for i in 1:n_lon
        for j in 1:n_lat
            # Collect paired observations
            field_values = Float64[]
            sam_values = Float64[]

            for sam_row in eachrow(sam_years)
                sam_yr = sam_row.year
                sam_val = sam_row.sam_annual

                # Get field data for the offset year and specified months
                field_yr = sam_yr + year_offset

                # Find all time indices matching this year and months
                time_mask = (field_years .== field_yr) .& (in.(field_months, Ref(months_range)))

                if sum(time_mask) > 0
                    # Average field values over matching time points
                    field_slice = field[i, j, time_mask]

                    # Only include if we have valid data (not all NaN)
                    if !all(isnan.(field_slice))
                        avg_field = mean(filter(!isnan, field_slice))
                        push!(field_values, avg_field)
                        push!(sam_values, sam_val)
                    end
                end
            end

            # Calculate correlation if we have enough data points
            if length(field_values) >= 5  # Minimum threshold for meaningful correlation
                # Remove any NaN pairs
                valid_idx = .!isnan.(field_values) .& .!isnan.(sam_values)
                if sum(valid_idx) >= 5
                    correlation_map[i, j] = cor(field_values[valid_idx], sam_values[valid_idx])
                end
            end
        end
    end

    return correlation_map
end

# ============================================================================
# CALCULATE CORRELATIONS FOR EACH GRIDDED FIELD
# ============================================================================

println("\nCalculating SAM correlations for each gridded field...")

# Initialize dictionary to store results
correlation_maps = Dictionary{String, Array{Float64, 2}}()

gridded_fields = ["SST", "MSL", "Z500"]
field_data = [sst, msl, z500]
field_times = [sst_time, msl_time, z_time]

for (field_name, field, field_time) in zip(gridded_fields, field_data, field_times)
    println("\n  Processing $field_name...")

    # Period 1: Sep-Dec immediately preceding the SAM year (offset = -1, but month-based)
    println("    Period 1: Sep-Dec preceding SAM year")
    # For Sep-Dec of year before, we need special handling
    # We'll correlate SAM in year Y with field in Sep-Dec of year Y-1
    corr_p1 = calculate_sam_correlation(field, field_time, annual_sam, 9:12, -1)
    set!(correlation_maps, "$(field_name)_period1", corr_p1)

    # Period 2: Jan-Apr of the year immediately following SAM year
    println("    Period 2: Jan-Apr of year after SAM year")
    corr_p2 = calculate_sam_correlation(field, field_time, annual_sam, 1:4, +1)
    set!(correlation_maps, "$(field_name)_period2", corr_p2)

    # Period 3: Jan-Apr of year+2 after SAM year
    println("    Period 3: Jan-Apr of year+2 after SAM year")
    corr_p3 = calculate_sam_correlation(field, field_time, annual_sam, 1:4, +2)
    set!(correlation_maps, "$(field_name)_period3", corr_p3)
end

# ============================================================================
# CREATE 3x3 PLOT: SAM CORRELATIONS
# ============================================================================

println("\nCreating 3x3 correlation plots...")

# Create output directory
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/SAM/wang_replication"
mkpath(visdir)

# Set fixed colorbar limits for correlations
cbarlim = 0.6
println("  Setting colorbar limits to ... ±$cbarlim")
common_colornorm = colors.Normalize(vmin=-cbarlim, vmax=cbarlim)

# Setup figure
fig = plt.figure(figsize=(18, 12))

# Row labels for fields
field_labels = ["SST", "MSL", "Z500"]

# Column labels for periods
period_labels = [
    "Sep-Dec (Year-1)",
    "Jan-Apr (Year+1)",
    "Jan-Apr (Year+2)"
]

# Coordinate arrays for plotting
coords_list = [(sst_lon, sst_lat), (msl_lon, msl_lat), (z_lon, z_lat)]

# Plot each combination
plot_idx = 1
contour_plots = []
for (field_idx, (field_name, coords)) in enumerate(zip(gridded_fields, coords_list))
    for (period_idx, period) in enumerate(["period1", "period2", "period3"])
        println("  Plotting $field_name - $period")

        # Get correlation map
        corr_key = "$(field_name)_$(period)"
        correlation = correlation_maps[corr_key]

        # Create subplot
        global plot_idx
        ax = fig.add_subplot(3, 3, plot_idx, projection=ccrs.Robinson(central_longitude=180))
        ax.set_global()
        ax.coastlines()

        # Plot correlation with common colorbar limits
        lon_grid, lat_grid = coords
        c = ax.contourf(lon_grid, lat_grid, correlation',
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
fig.suptitle("Point-by-Point Correlation with SAM\n(Deseasonalized Data)",
            fontsize=16, fontweight="bold", y=0.98)

# Add single colorbar for all subplots
cbar = fig.colorbar(contour_plots[1], ax=fig.get_axes(), orientation="horizontal",
                   pad=0.05, shrink=0.6, aspect=30)
cbar.set_label("Correlation Coefficient", fontsize=12, fontweight="bold")

# Save figure
output_path = joinpath(visdir, "sam_correlation_3x3.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
println("\nSaved plot to: $output_path")
plt.close(fig)

println("\nAnalysis complete!")
