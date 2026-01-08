"""
This script calculates and plots the gridded effects (regression) of the SAM index
on various meteorological and radiative variables with different lead/lag periods.

The SAM timeseries is shifted to lead and lag the gridded fields by:
0, 1, 2, 3, 6, 12, and 24 months

Positive lag: SAM leads (e.g., lag=1 means SAM at time t predicts gridded field at t+1)
Negative lag: SAM lags (e.g., lag=-1 means gridded field at time t predicts SAM at t+1)
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")

using CSV, DataFrames, Dates, Dictionaries, PythonCall, Statistics, SplitApplyCombine
@py import matplotlib.pyplot as plt, cartopy.crs as ccrs, matplotlib.colors as colors, cmasher as cmr

# Create output directory
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/SAM/gridded_effects_lagged"
mkpath(visdir)

# Load the calculated SAM index
datadir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/SAM"
sam_df = CSV.read(joinpath(datadir, "calculated_sam.csv"), DataFrame)
sam_df[!, :date] = Date.(sam_df.year, sam_df.month, sam_df.day)

println("SAM data info:")
println("  Date range: $(minimum(sam_df.date)) to $(maximum(sam_df.date))")
println("  Number of records: $(nrow(sam_df))")
println("  First few dates: $(sam_df.date[1:min(5, nrow(sam_df))])")
println("  Sample SAM values: $(sam_df.sam_calculated[1:min(5, nrow(sam_df))])")

# Define time periods
era5_period = (Date(1980, 1, 1), Date(2024, 12, 31))
ceres_period = (Date(2003, 3, 1), Date(2025, 2, 28))

println("Loading ERA5 data for period: $era5_period")

# Load ERA5 single level variables: u10, v10, t2m, sst
era5_single_vars = ["u10", "v10", "t2m", "sst"]
era5_data, era5_coords = load_era5_data(era5_single_vars, era5_period)

era5_lat = era5_coords["latitude"]
era5_lon = era5_coords["longitude"]
era5_time = round.(era5_coords["time"], Month(1), RoundDown)
era5_time_valid = in_time_period.(era5_time, Ref(era5_period))
era5_time = era5_time[era5_time_valid]
era5_float_time = calc_float_time.(era5_time)
era5_precalculated_month_groups = groupfind(month, era5_time)

println("ERA5 time info:")
println("  Date range: $(minimum(era5_time)) to $(maximum(era5_time))")
println("  Number of time points: $(length(era5_time))")
println("  First few dates: $(era5_time[1:min(5, length(era5_time))])")

# Restrict ERA5 data to valid time period
for var in era5_single_vars
    era5_data[var] = era5_data[var][:, :, era5_time_valid]
end

# Replace missings with NaNs in SST data
println("Processing SST data: replacing missings with NaNs...")
era5_data["sst"] = replace(era5_data["sst"], missing => NaN)

# Calculate wind speed from u10 and v10 BEFORE deseasonalizing
println("Calculating wind speed...")
set!(era5_data, "wind_speed", sqrt.(era5_data["u10"].^2 .+ era5_data["v10"].^2))

# Now deseasonalize all variables including wind speed (no detrending)
for var in vcat(era5_single_vars, ["wind_speed"])
    for slice in eachslice(era5_data[var]; dims = (1,2))
        deseasonalize_precalculated_groups(slice, era5_precalculated_month_groups)
    end
end

println("Loading CERES data for period: $ceres_period")

# Load CERES data
ceres_varnames = ["toa_net_all_mon"]
ceres_data, ceres_coords = load_new_ceres_data(ceres_varnames, ceres_period)

ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]
ceres_time = round.(ceres_coords["time"], Month(1), RoundDown)
ceres_time_valid = in_time_period.(ceres_time, Ref(ceres_period))
ceres_time = ceres_time[ceres_time_valid]
ceres_float_time = calc_float_time.(ceres_time)
ceres_precalculated_month_groups = groupfind(month, ceres_time)

# Deseasonalize CERES data (no detrending)
for var in ceres_varnames
    ceres_data[var] = ceres_data[var][:, :, ceres_time_valid]
    for slice in eachslice(ceres_data[var]; dims = (1,2))
        deseasonalize_precalculated_groups(slice, ceres_precalculated_month_groups)
    end
end

# Function to calculate correlation coefficient
function calculate_correlation_grid(grid_data, time_series)
    """
    Calculate correlation coefficient for each grid point.
    Correlation = Pearson correlation between grid data and time series.
    """
    n_lon, n_lat, n_time = size(grid_data)
    correlation_map = fill(NaN, n_lon, n_lat)

    # Filter out NaNs from time series
    valid_ts = .!isnan.(time_series)
    if sum(valid_ts) < 10
        @warn "Not enough valid time series points for correlation"
        return correlation_map
    end

    for i in 1:n_lon, j in 1:n_lat
        grid_point = grid_data[i, j, :]

        # Find valid indices (both grid and time series must be valid)
        valid_idx = .!isnan.(grid_point) .& .!isnan.(time_series)

        if sum(valid_idx) > 10  # Need at least 10 points
            y = grid_point[valid_idx]
            x = time_series[valid_idx]

            # Calculate Pearson correlation coefficient
            # cor(x,y) = cov(x,y) / (std(x) * std(y))
            correlation_map[i, j] = cor(x, y)
        end
    end

    return correlation_map
end

# Function to create lagged SAM timeseries
function create_lagged_sam_timeseries(time_vec, sam_data_df, lag_months)
    """
    Create lagged SAM timeseries where SAM leads (positive lag) or lags (negative lag) the grid data.
    Uses padding with NaNs to keep the full time period.

    Parameters:
    - time_vec: Vector of dates corresponding to grid_data time dimension
    - sam_data_df: DataFrame with 'date' and 'sam_calculated' columns
    - lag_months: Integer, positive means SAM leads, negative means SAM lags

    Returns:
    - lagged_sam_ts: Vector of SAM values with same length as time_vec, padded with NaNs
    """

    # Initialize with NaNs
    lagged_sam_ts = fill(NaN, length(time_vec))

    # Create a dictionary for quick lookup from all SAM data
    # Round SAM dates to first of month for matching
    sam_dict = Dict()
    for row in eachrow(sam_data_df)
        date_key = Date(year(row.date), month(row.date), 1)
        sam_dict[date_key] = row.sam_calculated
    end

    # Fill in the lagged values
    for (i, t) in enumerate(time_vec)
        # Round grid time to first of month
        t_rounded = Date(year(t), month(t), 1)

        # For positive lag: SAM at time t-lag predicts grid at time t
        # For negative lag: SAM at time t+|lag| predicts grid at time t
        sam_date = t_rounded - Month(lag_months)

        if haskey(sam_dict, sam_date)
            lagged_sam_ts[i] = sam_dict[sam_date]
        end
        # Otherwise remains NaN (padding)
    end

    return lagged_sam_ts
end

# Define lag periods (positive = SAM leads, negative = SAM lags)
lag_months = [0, 1, 2, 3, 6, 12, 24, -1, -2, -3, -6, -12, -24]

# Dictionary to store all effects
era5_vars_to_plot = ["u10", "v10", "wind_speed", "t2m", "sst"]

# Calculate effects for each lag period
println("\nCalculating lagged effects...")

# Store all effects for multi-panel plotting
era5_all_effects = Dictionary{String, Dictionary{Int, Matrix{Float64}}}()
ceres_all_effects = Dictionary{String, Dictionary{Int, Matrix{Float64}}}()

for var in era5_vars_to_plot
    set!(era5_all_effects, var, Dictionary{Int, Matrix{Float64}}())
end
for var in ceres_varnames
    set!(ceres_all_effects, var, Dictionary{Int, Matrix{Float64}}())
end

for lag in lag_months
    lag_str = lag >= 0 ? "lead_$(lag)" : "lag_$(abs(lag))"
    println("\n" * "="^60)
    println("Processing lag = $lag months (SAM $(lag >= 0 ? "leads" : "lags") by $(abs(lag)) months)")
    println("="^60)

    # ERA5 analysis
    println("\nERA5 Variables:")

    # Create lagged SAM timeseries for ERA5
    lagged_sam_era5 = create_lagged_sam_timeseries(era5_time, sam_df, lag)
    n_valid_era5 = sum(.!isnan.(lagged_sam_era5))
    println("  ERA5 SAM timeseries: $(n_valid_era5) valid points out of $(length(lagged_sam_era5))")

    if n_valid_era5 > 0
        valid_sam = lagged_sam_era5[.!isnan.(lagged_sam_era5)]
        println("    Sample valid SAM values: $(valid_sam[1:min(5, n_valid_era5)])")
        println("    SAM mean: $(mean(valid_sam)), std: $(std(valid_sam))")
    end

    for var in era5_vars_to_plot
        println("  Processing $var...")

        # Calculate correlation using full grid data and lagged SAM
        correlation = calculate_correlation_grid(era5_data[var], lagged_sam_era5)
        set!(era5_all_effects[var], lag, correlation)
    end

    # CERES analysis
    println("\nCERES Variables:")

    # Create lagged SAM timeseries for CERES
    lagged_sam_ceres = create_lagged_sam_timeseries(ceres_time, sam_df, lag)
    n_valid_ceres = sum(.!isnan.(lagged_sam_ceres))
    println("  CERES SAM timeseries: $(n_valid_ceres) valid points out of $(length(lagged_sam_ceres))")

    for var in ceres_varnames
        println("  Processing $var...")

        # Calculate correlation using full grid data and lagged SAM
        correlation = calculate_correlation_grid(ceres_data[var], lagged_sam_ceres)
        set!(ceres_all_effects[var], lag, correlation)
    end
end

# Create multi-panel plots for ERA5 variables
println("\n" * "="^60)
println("Creating multi-panel plots...")
println("="^60)

for var in era5_vars_to_plot
    println("\nPlotting $var with all lags...")

    # Variable labels
    var_labels = Dict(
        "u10" => ("10m Zonal Wind (U10)", "Correlation with SAM"),
        "v10" => ("10m Meridional Wind (V10)", "Correlation with SAM"),
        "wind_speed" => ("10m Wind Speed", "Correlation with SAM"),
        "t2m" => ("2m Temperature (T2M)", "Correlation with SAM"),
        "sst" => ("Sea Surface Temperature (SST)", "Correlation with SAM")
    )

    title_base, cbar_label = var_labels[var]

    # Use fixed color limits for correlation (-1 to 1)
    global_colornorm = colors.Normalize(vmin=-1, vmax=1)

    # Create figure with subplots (5 rows x 3 columns for 13 lags + 2 empty)
    fig = plt.figure(figsize=(18, 24))

    contour_handle = nothing
    for (idx, lag) in enumerate(lag_months)
        # Create subplot with projection
        ax = fig.add_subplot(5, 3, idx, projection=ccrs.Robinson(central_longitude=180))

        ax.set_global()
        ax.coastlines(linewidth=0.5)

        effect_map = era5_all_effects[var][lag]

        # Plot
        c = ax.contourf(era5_lon, era5_lat, effect_map',
                       transform=ccrs.PlateCarree(),
                       cmap=cmr.prinsenvlag.reversed(),
                       levels=21,
                       norm=global_colornorm)

        # Store the contour handle for colorbar (use last one created)
        contour_handle = c

        # Create subplot title
        if lag == 0
            lag_title = "Lag = 0 (Simultaneous)"
        elseif lag > 0
            lag_title = "SAM leads by $lag mo"
        else
            lag_title = "SAM lags by $(abs(lag)) mo"
        end

        ax.set_title(lag_title, fontsize=11, fontweight="bold")
    end

    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92, hspace=0.15, wspace=0.1)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour_handle, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=13)

    # Add overall title
    fig.suptitle("SAM Correlation with $title_base\n(ERA5, 1980-2024)",
                fontsize=16, fontweight="bold", y=0.98)

    # Save
    output_path = joinpath(visdir, "sam_effect_$(var)_all_lags.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    println("  Saved: $output_path")
    plt.close(fig)
end

# Create multi-panel plots for CERES variables
for var in ceres_varnames
    println("\nPlotting $var with all lags...")

    # Use fixed color limits for correlation (-1 to 1)
    global_colornorm = colors.Normalize(vmin=-1, vmax=1)

    # Create figure with subplots (5 rows x 3 columns for 13 lags + 2 empty)
    fig = plt.figure(figsize=(18, 24))

    contour_handle = nothing
    for (idx, lag) in enumerate(lag_months)
        # Create subplot with projection
        ax = fig.add_subplot(5, 3, idx, projection=ccrs.Robinson(central_longitude=180))

        ax.set_global()
        ax.coastlines(linewidth=0.5)

        effect_map = ceres_all_effects[var][lag]

        # Plot
        c = ax.contourf(ceres_lon, ceres_lat, effect_map',
                       transform=ccrs.PlateCarree(),
                       cmap=cmr.prinsenvlag.reversed(),
                       levels=21,
                       norm=global_colornorm)

        # Store the contour handle for colorbar (use last one created)
        contour_handle = c

        # Create subplot title
        if lag == 0
            lag_title = "Lag = 0 (Simultaneous)"
        elseif lag > 0
            lag_title = "SAM leads by $lag mo"
        else
            lag_title = "SAM lags by $(abs(lag)) mo"
        end

        ax.set_title(lag_title, fontsize=11, fontweight="bold")
    end

    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92, hspace=0.15, wspace=0.1)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour_handle, cax=cbar_ax)
    cbar.set_label("Correlation with SAM", fontsize=13)

    # Add overall title
    fig.suptitle("SAM Correlation with TOA Net Radiation\n(CERES, March 2003 - February 2025)",
                fontsize=16, fontweight="bold", y=0.98)

    # Save
    output_path = joinpath(visdir, "sam_effect_$(var)_all_lags.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    println("  Saved: $output_path")
    plt.close(fig)
end

println("\n" * "="^60)
println("Analysis complete!")
println("All plots saved to: $visdir")
println("="^60)
