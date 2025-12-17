"""
This script calculates and plots the gridded effects (regression) of the SAM index
on various meteorological and radiative variables.
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")

using CSV, DataFrames, Dates, Dictionaries, PythonCall, Statistics, SplitApplyCombine
@py import matplotlib.pyplot as plt, cartopy.crs as ccrs, matplotlib.colors as colors, cmasher as cmr

# Create output directory
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/SAM/gridded_effects"
mkpath(visdir)

# Load the calculated SAM index
datadir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/SAM"
sam_df = CSV.read(joinpath(datadir, "calculated_sam.csv"), DataFrame)
sam_df[!, :date] = Date.(sam_df.year, sam_df.month, sam_df.day)

# Define time periods
era5_period = (Date(1980, 1, 1), Date(2024, 12, 31))
ceres_period = (Date(2003, 3, 1), Date(2025, 2, 28))

println("Loading ERA5 data for period: $era5_period")

# Load ERA5 single level variables: u10, v10, t2m
era5_single_vars = ["u10", "v10", "t2m"]
era5_data, era5_coords = load_era5_data(era5_single_vars, era5_period)

era5_lat = era5_coords["latitude"]
era5_lon = era5_coords["longitude"]
era5_time = round.(era5_coords["time"], Month(1), RoundDown)
era5_time_valid = in_time_period.(era5_time, Ref(era5_period))
era5_time = era5_time[era5_time_valid]
era5_float_time = calc_float_time.(era5_time)
era5_precalculated_month_groups = groupfind(month, era5_time)

# Restrict ERA5 data to valid time period
for var in era5_single_vars
    era5_data[var] = era5_data[var][:, :, era5_time_valid]
end

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

# Function to calculate regression coefficient (effect)
function calculate_regression_grid(grid_data, time_series)
    """
    Calculate regression coefficient for each grid point.
    Effect = regression coefficient when regressing grid data onto standardized time series.
    """
    n_lon, n_lat, n_time = size(grid_data)
    regression_map = fill(NaN, n_lon, n_lat)
    
    # Standardize the time series
    ts_standardized = (time_series .- mean(skipmissing(time_series))) ./ std(skipmissing(time_series))
    
    for i in 1:n_lon, j in 1:n_lat
        grid_point = grid_data[i, j, :]
        
        # Skip if all missing
        if all(ismissing.(grid_point))
            continue
        end
        
        # Find valid indices
        valid_idx = .!ismissing.(grid_point) .& .!ismissing.(ts_standardized)
        
        if sum(valid_idx) > 10  # Need at least 10 points
            y = grid_point[valid_idx]
            x = ts_standardized[valid_idx]
            
            # Calculate regression coefficient: y = β*x + α
            # β = cov(x,y) / var(x)
            regression_map[i, j] = cov(x, y) / var(x)
        end
    end
    
    return regression_map
end

println("Calculating SAM effects on ERA5 variables...")

# Match SAM data to ERA5 time period
sam_era5 = filter(row -> row.date in era5_time, sam_df)
sort!(sam_era5, :date)

# Ensure alignment
@assert sam_era5.date == era5_time "Time mismatch between SAM and ERA5 data"

sam_ts_era5 = sam_era5.sam_calculated

# Calculate effects for ERA5 variables
era5_vars_to_plot = ["u10", "v10", "wind_speed", "t2m"]
era5_effects = Dictionary{String, Matrix{Float64}}()

for var in era5_vars_to_plot
    println("  Processing $var...")
    effect = calculate_regression_grid(era5_data[var], sam_ts_era5)
    set!(era5_effects, var, effect)
end

println("Calculating SAM effects on CERES variables...")

# Match SAM data to CERES time period
sam_ceres = filter(row -> row.date in ceres_time, sam_df)
sort!(sam_ceres, :date)

# Ensure alignment
@assert sam_ceres.date == ceres_time "Time mismatch between SAM and CERES data"

sam_ts_ceres = sam_ceres.sam_calculated

# Calculate effects for CERES variables
ceres_effects = Dictionary{String, Matrix{Float64}}()

for var in ceres_varnames
    println("  Processing $var...")
    effect = calculate_regression_grid(ceres_data[var], sam_ts_ceres)
    set!(ceres_effects, var, effect)
end

println("Calculating seasonal effects...")

# Create seasonal masks for April-September and Oct-March
era5_months = month.(era5_time)
era5_apr_sep_mask = (era5_months .>= 4) .& (era5_months .<= 9)
era5_oct_mar_mask = (era5_months .<= 3) .| (era5_months .>= 10)

ceres_months = month.(ceres_time)
ceres_apr_sep_mask = (ceres_months .>= 4) .& (ceres_months .<= 9)
ceres_oct_mar_mask = (ceres_months .<= 3) .| (ceres_months .>= 10)

# Calculate seasonal effects for ERA5 variables
era5_effects_apr_sep = Dictionary{String, Matrix{Float64}}()
era5_effects_oct_mar = Dictionary{String, Matrix{Float64}}()

for var in era5_vars_to_plot
    println("  Processing $var for April-September...")
    # Subset data for April-September
    data_apr_sep = era5_data[var][:, :, era5_apr_sep_mask]
    sam_apr_sep = sam_ts_era5[era5_apr_sep_mask]
    effect = calculate_regression_grid(data_apr_sep, sam_apr_sep)
    set!(era5_effects_apr_sep, var, effect)
    
    println("  Processing $var for October-March...")
    # Subset data for October-March
    data_oct_mar = era5_data[var][:, :, era5_oct_mar_mask]
    sam_oct_mar = sam_ts_era5[era5_oct_mar_mask]
    effect = calculate_regression_grid(data_oct_mar, sam_oct_mar)
    set!(era5_effects_oct_mar, var, effect)
end

# Calculate seasonal effects for CERES variables
ceres_effects_apr_sep = Dictionary{String, Matrix{Float64}}()
ceres_effects_oct_mar = Dictionary{String, Matrix{Float64}}()

for var in ceres_varnames
    println("  Processing $var for April-September...")
    # Subset data for April-September
    data_apr_sep = ceres_data[var][:, :, ceres_apr_sep_mask]
    sam_apr_sep = sam_ts_ceres[ceres_apr_sep_mask]
    effect = calculate_regression_grid(data_apr_sep, sam_apr_sep)
    set!(ceres_effects_apr_sep, var, effect)
    
    println("  Processing $var for October-March...")
    # Subset data for October-March
    data_oct_mar = ceres_data[var][:, :, ceres_oct_mar_mask]
    sam_oct_mar = sam_ts_ceres[ceres_oct_mar_mask]
    effect = calculate_regression_grid(data_oct_mar, sam_oct_mar)
    set!(ceres_effects_oct_mar, var, effect)
end

println("Creating plots...")

# Plot ERA5 effects
for var in era5_vars_to_plot
    println("  Plotting $var...")
    
    effect_map = era5_effects[var]
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))
    
    ax.set_global()
    ax.coastlines()
    
    # Calculate symmetric color limits
    absmax = pyimport("numpy").nanmax(pyimport("numpy").abs(effect_map))
    colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    
    # Plot
    c = ax.contourf(era5_lon, era5_lat, effect_map', 
                   transform=ccrs.PlateCarree(), 
                   cmap=cmr.prinsenvlag.reversed(), 
                   levels=21, 
                   norm=colornorm)
    
    # Add colorbar
    cbar = plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    
    # Set title and labels based on variable
    var_labels = Dict(
        "u10" => ("10m Zonal Wind (U10)", "m/s per σ_SAM"),
        "v10" => ("10m Meridional Wind (V10)", "m/s per σ_SAM"),
        "wind_speed" => ("10m Wind Speed", "m/s per σ_SAM"),
        "t2m" => ("2m Temperature (T2M)", "K per σ_SAM")
    )
    
    title, cbar_label = var_labels[var]
    ax.set_title("SAM Effect on $title\n(1980-2024)", fontsize=14, fontweight="bold")
    cbar.set_label(cbar_label, fontsize=11)
    
    # Save
    output_path = joinpath(visdir, "sam_effect_$(var).png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    println("    Saved: $output_path")
    plt.close(fig)
end

# Plot seasonal ERA5 effects
for season_label in ["apr_sep", "oct_mar"]
    effects_dict = season_label == "apr_sep" ? era5_effects_apr_sep : era5_effects_oct_mar
    season_name = season_label == "apr_sep" ? "April-September" : "October-March"
    
    for var in era5_vars_to_plot
        println("  Plotting $var ($season_name)...")
        
        effect_map = effects_dict[var]
        
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))
        
        ax.set_global()
        ax.coastlines()
        
        # Calculate symmetric color limits
        absmax = pyimport("numpy").nanmax(pyimport("numpy").abs(effect_map))
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
        
        # Plot
        c = ax.contourf(era5_lon, era5_lat, effect_map', 
                       transform=ccrs.PlateCarree(), 
                       cmap=cmr.prinsenvlag.reversed(), 
                       levels=21, 
                       norm=colornorm)
        
        # Add colorbar
        cbar = plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        
        # Set title and labels based on variable
        var_labels = Dict(
            "u10" => ("10m Zonal Wind (U10)", "m/s per σ_SAM"),
            "v10" => ("10m Meridional Wind (V10)", "m/s per σ_SAM"),
            "wind_speed" => ("10m Wind Speed", "m/s per σ_SAM"),
            "t2m" => ("2m Temperature (T2M)", "K per σ_SAM")
        )
        
        title, cbar_label = var_labels[var]
        ax.set_title("SAM Effect on $title ($season_name)\n(1980-2024)", fontsize=14, fontweight="bold")
        cbar.set_label(cbar_label, fontsize=11)
        
        # Save
        output_path = joinpath(visdir, "sam_effect_$(var)_$(season_label).png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        println("    Saved: $output_path")
        plt.close(fig)
    end
end

# Plot CERES effects
for var in ceres_varnames
    println("  Plotting $var...")
    
    effect_map = ceres_effects[var]
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))
    
    ax.set_global()
    ax.coastlines()
    
    # Calculate symmetric color limits
    absmax = pyimport("numpy").nanmax(pyimport("numpy").abs(effect_map))
    colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
    
    # Plot
    c = ax.contourf(ceres_lon, ceres_lat, effect_map', 
                   transform=ccrs.PlateCarree(), 
                   cmap=cmr.prinsenvlag.reversed(), 
                   levels=21, 
                   norm=colornorm)
    
    # Add colorbar
    cbar = plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    
    ax.set_title("SAM Effect on TOA Net Radiation\n(March 2003 - February 2025)", 
                fontsize=14, fontweight="bold")
    cbar.set_label("W/m² per σ_SAM", fontsize=11)
    
    # Save
    output_path = joinpath(visdir, "sam_effect_$(var).png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    println("    Saved: $output_path")
    plt.close(fig)
end

# Plot seasonal CERES effects
for season_label in ["apr_sep", "oct_mar"]
    effects_dict = season_label == "apr_sep" ? ceres_effects_apr_sep : ceres_effects_oct_mar
    season_name = season_label == "apr_sep" ? "April-September" : "October-March"
    
    for var in ceres_varnames
        println("  Plotting $var ($season_name)...")
        
        effect_map = effects_dict[var]
        
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))
        
        ax.set_global()
        ax.coastlines()
        
        # Calculate symmetric color limits
        absmax = pyimport("numpy").nanmax(pyimport("numpy").abs(effect_map))
        colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)
        
        # Plot
        c = ax.contourf(ceres_lon, ceres_lat, effect_map', 
                       transform=ccrs.PlateCarree(), 
                       cmap=cmr.prinsenvlag.reversed(), 
                       levels=21, 
                       norm=colornorm)
        
        # Add colorbar
        cbar = plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        
        ax.set_title("SAM Effect on TOA Net Radiation ($season_name)\n(March 2003 - February 2025)", 
                    fontsize=14, fontweight="bold")
        cbar.set_label("W/m² per σ_SAM", fontsize=11)
        
        # Save
        output_path = joinpath(visdir, "sam_effect_$(var)_$(season_label).png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        println("    Saved: $output_path")
        plt.close(fig)
    end
end

println("\nAnalysis complete! Plots saved to: $visdir")
