using JLD2, Statistics, StatsBase, Dates, SplitApplyCombine, CSV, DataFrames, PythonCall
using NCDatasets, Dictionaries

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../area_averager/area_averager.jl")
include("../utils/plot_global.jl")
include("regions.jl")

@py import matplotlib.pyplot as plt
@py import matplotlib.patches as patches

"""
This script averages ERA5 surface temperature ("t2m") and CERES local radiation 
over the lat/lon boxes prescribed in regions.jl using the area_averager module.
"""

# Set up output directories
visdir = "../../vis/radiative_corrs_different_regions"
if !isdir(visdir)
    mkpath(visdir)
end

savedir = "../../data/examine_radiative_corrs_for_different_regions"
if !isdir(savedir)
    mkpath(savedir)
end

# Create region time series plot directories
region_plots_dir = joinpath(visdir, "region_time_series")
if !isdir(region_plots_dir)
    mkpath(region_plots_dir)
end

vis = false

println("Loading ERA5 surface temperature data...")
# Load ERA5 surface temperature
era5_variables = ["t2m"]
era5_data, era5_coords = load_era5_data(era5_variables, (Date(1981, 10), Date(2024, 11)))

# Extract ERA5 data and coordinates
t2m_data = era5_data["t2m"]  # (lon, lat, time)
era5_lat = era5_coords["latitude"]
era5_lon = era5_coords["longitude"] 
era5_time = era5_coords["time"]

println("ERA5 surface temperature loaded:")
println("  Grid: $(length(era5_lon)) × $(length(era5_lat))")
println("  Time points: $(length(era5_time))")
println("  Data shape: $(size(t2m_data))")

println("Loading CERES radiation data...")
# Load CERES radiation data
ceres_variables = ["toa_net_all_mon", "gridded_net_sw", "toa_lw_all_mon"]
ceres_data, ceres_coords = load_ceres_data(ceres_variables, time_period)

# Extract CERES data and coordinates
toa_net = ceres_data["toa_net_all_mon"]  # (lon, lat, time)
net_sw = ceres_data["gridded_net_sw"]    # (lon, lat, time)
toa_lw = ceres_data["toa_lw_all_mon"]    # (lon, lat, time)
ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]
ceres_time = ceres_coords["time"]

println("CERES radiation data loaded:")
println("  Grid: $(length(ceres_lon)) × $(length(ceres_lat))")
println("  Time points: $(length(ceres_time))")
println("  TOA Net shape: $(size(toa_net))")
println("  Net SW shape: $(size(net_sw))")
println("  TOA LW shape: $(size(toa_lw))")

# Define the regions from regions.jl
regions = Dict(
    "SEPac" => (sepac_lon, sepac_lat),
    "NEPac" => (nepac_lon, nepac_lat),
    "WEqPac" => (weqpac_lon, weqpac_lat),
    "Kuroshio" => (kuroshio_lon, kuroshio_lat),
    "Rockies" => (rockies_lon, rockies_lat)
)

println("Processing regions: $(join(keys(regions), ", "))")

# Create masks for each region and average the data
results = Dictionary()

for (region_name, (lon_bounds, lat_bounds)) in regions
    println("\nProcessing region: $region_name")
    println("  Longitude bounds: $(lon_bounds)")
    println("  Latitude bounds: $(lat_bounds)")
    
    # Create box mask for this region on ERA5 grid
    era5_region_mask = create_box_mask(era5_lat, era5_lon, 
                                      lat_bounds=lat_bounds, 
                                      lon_bounds=lon_bounds)
    
    # Create box mask for this region on CERES grid (separate mask, not interpolated)
    ceres_region_mask = create_box_mask(ceres_lat, ceres_lon,
                                       lat_bounds=lat_bounds,
                                       lon_bounds=lon_bounds)
    
    n_points_era5 = sum(era5_region_mask)
    n_points_ceres = sum(ceres_region_mask)
    println("  Region covers $n_points_era5 ERA5 grid points and $n_points_ceres CERES grid points")
    
    if n_points_era5 == 0 || n_points_ceres == 0
        @warn "No grid points found in region $region_name on one or both grids - skipping"
        continue
    end
    
    # Average ERA5 surface temperature over the region
    t2m_avg, t2m_mask_interp, t2m_weights = average_over_mask(
        t2m_data, era5_lat, era5_lon,           # Data and its grid
        era5_region_mask, era5_lat, era5_lon,   # Mask and its grid (same as data)
        cosine_weights=true
    )
    
    # Average CERES data over the region using native CERES mask
    toa_net_avg, net_mask_interp, net_weights = average_over_mask(
        toa_net, ceres_lat, ceres_lon,          # Data and its grid
        ceres_region_mask, ceres_lat, ceres_lon, # Mask and its grid (same as data)
        cosine_weights=true
    )
    
    net_sw_avg, sw_mask_interp, sw_weights = average_over_mask(
        net_sw, ceres_lat, ceres_lon,
        ceres_region_mask, ceres_lat, ceres_lon,
        cosine_weights=true
    )
    
    toa_lw_avg, lw_mask_interp, lw_weights = average_over_mask(
        toa_lw, ceres_lat, ceres_lon,
        ceres_region_mask, ceres_lat, ceres_lon,
        cosine_weights=true
    )
    
    # Create detrended and deseasonalized versions
    # ERA5 T2M detrended and deseasonalized
    t2m_detrend_deseas = copy(t2m_avg)
    era5_float_times = calc_float_time.(era5_time)
    era5_months = month.(era5_time)
    t2m_fit = detrend_and_deseasonalize!(t2m_detrend_deseas, era5_float_times, era5_months; aggfunc = median, trendfunc = theil_sen_fit)
    
    # CERES data detrended and deseasonalized
    toa_net_detrend_deseas = copy(toa_net_avg)
    ceres_float_times = calc_float_time.(ceres_time)
    ceres_months = month.(ceres_time)
    toa_net_fit = detrend_and_deseasonalize!(toa_net_detrend_deseas, ceres_float_times, ceres_months; aggfunc = median, trendfunc = theil_sen_fit)

    net_sw_detrend_deseas = copy(net_sw_avg)
    net_sw_fit = detrend_and_deseasonalize!(net_sw_detrend_deseas, ceres_float_times, ceres_months; aggfunc = median, trendfunc = theil_sen_fit)

    toa_lw_detrend_deseas = copy(toa_lw_avg)
    toa_lw_fit = detrend_and_deseasonalize!(toa_lw_detrend_deseas, ceres_float_times, ceres_months; aggfunc = median, trendfunc = theil_sen_fit)

    # Store results
    region_results = Dictionary()
    set!(region_results, "t2m_avg", t2m_avg)
    set!(region_results, "toa_net_avg", toa_net_avg)
    set!(region_results, "net_sw_avg", net_sw_avg)
    set!(region_results, "toa_lw_avg", toa_lw_avg)
    # Store detrended and deseasonalized versions
    set!(region_results, "t2m_detrend_deseas", t2m_detrend_deseas)
    set!(region_results, "toa_net_detrend_deseas", toa_net_detrend_deseas)
    set!(region_results, "net_sw_detrend_deseas", net_sw_detrend_deseas)
    set!(region_results, "toa_lw_detrend_deseas", toa_lw_detrend_deseas)
    # Store trend fits for reference
    set!(region_results, "t2m_fit", t2m_fit)
    set!(region_results, "toa_net_fit", toa_net_fit)
    set!(region_results, "net_sw_fit", net_sw_fit)
    set!(region_results, "toa_lw_fit", toa_lw_fit)
    set!(region_results, "era5_mask", era5_region_mask)
    set!(region_results, "ceres_mask", ceres_region_mask)
    set!(region_results, "lon_bounds", lon_bounds)
    set!(region_results, "lat_bounds", lat_bounds)
    
    set!(results, region_name, region_results)
    
    println("  Average values:")
    println("    T2M: $(round(mean(t2m_avg), digits=2)) ± $(round(std(t2m_avg), digits=2)) K")
    println("    TOA Net: $(round(mean(toa_net_avg), digits=2)) ± $(round(std(toa_net_avg), digits=2)) W/m²")
    println("    Net SW: $(round(mean(net_sw_avg), digits=2)) ± $(round(std(net_sw_avg), digits=2)) W/m²")
    println("    TOA LW: $(round(mean(toa_lw_avg), digits=2)) ± $(round(std(toa_lw_avg), digits=2)) W/m²")
    println("  Detrended & deseasonalized std deviations:")
    println("    T2M: $(round(std(t2m_detrend_deseas), digits=2)) K")
    println("    TOA Net: $(round(std(toa_net_detrend_deseas), digits=2)) W/m²")
    println("    Net SW: $(round(std(net_sw_detrend_deseas), digits=2)) W/m²")
    println("    TOA LW: $(round(std(toa_lw_detrend_deseas), digits=2)) W/m²")
end

# Save individual CSV files for each region (separate ERA5 and CERES files)
println("\nSaving individual region files (no combined datasets)...")

for (region_name, region_data) in pairs(results)
    # ERA5 file for this region
    era5_region_df = DataFrame(
        Date = era5_time,
        T2M = region_data["t2m_avg"],
        T2M_Detrend_Deseas = region_data["t2m_detrend_deseas"]
    )
    CSV.write(joinpath(savedir, "$(lowercase(region_name))_era5_t2m.csv"), era5_region_df)
    println("Saved ERA5 T2M data for $region_name")
    
    # CERES file for this region
    ceres_region_df = DataFrame(
        Date = ceres_time,
        TOA_Net = region_data["toa_net_avg"],
        Net_SW = region_data["net_sw_avg"],
        TOA_LW = region_data["toa_lw_avg"],
        TOA_Net_Detrend_Deseas = region_data["toa_net_detrend_deseas"],
        Net_SW_Detrend_Deseas = region_data["net_sw_detrend_deseas"],
        TOA_LW_Detrend_Deseas = region_data["toa_lw_detrend_deseas"]
    )
    CSV.write(joinpath(savedir, "$(lowercase(region_name))_ceres_radiation.csv"), ceres_region_df)
    println("Saved CERES radiation data for $region_name")
end

println("Saved all individual region files to $savedir")


# Create time series plots for individual regions with subdirectories
println("\nCreating individual time series plots for each region...")

variables_era5 = ["T2M"]
variables_ceres = ["TOA_Net", "Net_SW", "TOA_LW"]
var_labels = Dict(
    "T2M" => "Surface Temperature (K)",
    "TOA_Net" => "TOA Net Radiation (W/m²)",
    "Net_SW" => "Net SW Radiation (W/m²)", 
    "TOA_LW" => "TOA LW Radiation (W/m²)"
)

# Create plots for each region individually
for (region_name, region_data) in pairs(results)
    # Create subdirectory for this region
    region_subdir = joinpath(region_plots_dir, lowercase(region_name))
    if !isdir(region_subdir)
        mkpath(region_subdir)
    end
    
    # Plot ERA5 variables for this region (original and detrended/deseasonalized)
    for var in variables_era5
        # Original time series
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        data_values = region_data["t2m_avg"]
        ax.plot(era5_time, data_values, linewidth=1.5, color="blue")
        
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(var_labels[var], fontsize=12)
        ax.set_title("$region_name: $(var_labels[var]) (ERA5) - Original", fontsize=14)
        ax.grid(true, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(joinpath(region_subdir, "$(lowercase(var))_original.png"), dpi=300, bbox_inches="tight")
        if vis
            plt.show()
        end
        plt.close()
        
        # Detrended and deseasonalized time series
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        data_values_detrend = region_data["t2m_detrend_deseas"]
        ax.plot(era5_time, data_values_detrend, linewidth=1.5, color="darkblue")
        
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(var_labels[var], fontsize=12)
        ax.set_title("$region_name: $(var_labels[var]) (ERA5) - Detrended & Deseasonalized", fontsize=14)
        ax.grid(true, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(joinpath(region_subdir, "$(lowercase(var))_detrend_deseas.png"), dpi=300, bbox_inches="tight")
        if vis
            plt.show()
        end
        plt.close()
    end
    
    # Plot CERES variables for this region (original and detrended/deseasonalized)
    for var in variables_ceres
        # Get data for this region
        if var == "TOA_Net"
            data_values = region_data["toa_net_avg"]
            data_values_detrend = region_data["toa_net_detrend_deseas"]
        elseif var == "Net_SW"
            data_values = region_data["net_sw_avg"]
            data_values_detrend = region_data["net_sw_detrend_deseas"]
        elseif var == "TOA_LW"
            data_values = region_data["toa_lw_avg"]
            data_values_detrend = region_data["toa_lw_detrend_deseas"]
        end
        
        # Original time series
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(ceres_time, data_values, linewidth=1.5, color="red")
        
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(var_labels[var], fontsize=12)
        ax.set_title("$region_name: $(var_labels[var]) (CERES) - Original", fontsize=14)
        ax.grid(true, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(joinpath(region_subdir, "$(lowercase(replace(var, "_" => "")))_original.png"), dpi=300, bbox_inches="tight")
        if vis
            plt.show()
        end
        plt.close()
        
        # Detrended and deseasonalized time series
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(ceres_time, data_values_detrend, linewidth=1.5, color="darkred")
        
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(var_labels[var], fontsize=12)
        ax.set_title("$region_name: $(var_labels[var]) (CERES) - Detrended & Deseasonalized", fontsize=14)
        ax.grid(true, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(joinpath(region_subdir, "$(lowercase(replace(var, "_" => "")))_detrend_deseas.png"), dpi=300, bbox_inches="tight")
        if vis
            plt.show()
        end
        plt.close()
    end
    
    println("Saved plots for $region_name in $(region_subdir)")
end

# Create summary statistics table
println("\nCreating summary statistics...")

summary_stats = []
for (region_name, region_data) in pairs(results)
    stats = Dict(
        "Region" => region_name,
        "Grid_Points_ERA5" => sum(region_data["era5_mask"]),
        "Grid_Points_CERES" => sum(region_data["ceres_mask"]),
        "T2M_Mean" => round(mean(region_data["t2m_avg"]), digits=2),
        "T2M_Std" => round(std(region_data["t2m_avg"]), digits=2),
        "T2M_Detrend_Deseas_Std" => round(std(region_data["t2m_detrend_deseas"]), digits=2),
        "TOA_Net_Mean" => round(mean(region_data["toa_net_avg"]), digits=2),
        "TOA_Net_Std" => round(std(region_data["toa_net_avg"]), digits=2),
        "TOA_Net_Detrend_Deseas_Std" => round(std(region_data["toa_net_detrend_deseas"]), digits=2),
        "Net_SW_Mean" => round(mean(region_data["net_sw_avg"]), digits=2),
        "Net_SW_Std" => round(std(region_data["net_sw_avg"]), digits=2),
        "Net_SW_Detrend_Deseas_Std" => round(std(region_data["net_sw_detrend_deseas"]), digits=2),
        "TOA_LW_Mean" => round(mean(region_data["toa_lw_avg"]), digits=2),
        "TOA_LW_Std" => round(std(region_data["toa_lw_avg"]), digits=2),
        "TOA_LW_Detrend_Deseas_Std" => round(std(region_data["toa_lw_detrend_deseas"]), digits=2),
        "Lon_Min" => region_data["lon_bounds"][1],
        "Lon_Max" => region_data["lon_bounds"][2],
        "Lat_Min" => region_data["lat_bounds"][1],
        "Lat_Max" => region_data["lat_bounds"][2]
    )
    push!(summary_stats, stats)
end

summary_df = DataFrame(summary_stats)
CSV.write(joinpath(savedir, "regional_summary_statistics.csv"), summary_df)

println("Summary Statistics:")
println(summary_df)

# Save masks and metadata to JLD2 file for future use
masks_dict = Dictionary()
for (region_name, region_data) in pairs(results)
    region_dict = Dictionary()
    set!(region_dict, "era5_mask", Float32.(region_data["era5_mask"]))
    set!(region_dict, "ceres_mask", Float32.(region_data["ceres_mask"]))
    set!(region_dict, "lon_bounds", region_data["lon_bounds"])
    set!(region_dict, "lat_bounds", region_data["lat_bounds"])
    set!(masks_dict, region_name, region_dict)
end

jldsave(joinpath(savedir, "regional_masks.jld2");
    masks = masks_dict,
    era5_latitude = Float32.(era5_lat),
    era5_longitude = Float32.(era5_lon),
    ceres_latitude = Float32.(ceres_lat),
    ceres_longitude = Float32.(ceres_lon),
    era5_time = era5_time,
    ceres_time = ceres_time
)

println("\nScript completed successfully!")
println("Output files created:")
println("- Regional boundaries map: $(joinpath(visdir, "regional_boundaries_map.png"))")
println("- Individual region time series plots: $(region_plots_dir)/<region_name>/*_original.png")
println("- Individual region detrended plots: $(region_plots_dir)/<region_name>/*_detrend_deseas.png")  
println("- Individual ERA5 region files: $(savedir)/*_era5_t2m.csv")
println("- Individual CERES region files: $(savedir)/*_ceres_radiation.csv")
println("- Summary statistics: $(joinpath(savedir, "regional_summary_statistics.csv"))")
println("- Masks and metadata: $(joinpath(savedir, "regional_masks.jld2"))")
println("\nData files maintain original time series lengths:")
println("- ERA5 T2M: $(length(era5_time)) time points")
println("- CERES radiation: $(length(ceres_time)) time points")
println("- No combined datasets created - only individual region files")
println("\nEach CSV file now includes both original and detrended/deseasonalized time series:")
println("- ERA5 files: T2M and T2M_Detrend_Deseas columns")
println("- CERES files: Original variables and *_Detrend_Deseas columns")
println("\nPlot organization:")
println("- Each region has its own subdirectory in $(region_plots_dir)/")
println("- Original time series: *_original.png")
println("- Detrended & deseasonalized: *_detrend_deseas.png")
