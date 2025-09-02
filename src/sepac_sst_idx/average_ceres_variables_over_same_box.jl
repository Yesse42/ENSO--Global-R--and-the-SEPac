using JLD2, Statistics, StatsBase, Dates, SplitApplyCombine, CSV, DataFrames, PythonCall
using NCDatasets, Dictionaries

# Include necessary modules
cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")
include("../pls_regressor/pls_functions.jl")
include("../pls_regressor/pls_structs.jl")
include("../utils/plot_global.jl")

@py import matplotlib.pyplot as plt

"""
This script calculates cosine weighted averages of CERES data (net SW, -1*LW, net radiation) 
over the SEPac region using the same filters as in create_sepac_sst_idx.jl
"""

# Set up output directory
visdir = "../../vis/sepac_radiation_effects/ceres_averages"
if !isdir(visdir)
    mkpath(visdir)
end

savedir = "../../data/SEPac_SST/"
vis = false

# Define time period for CERES data
time_period = (Date(2001, 1, 1), Date(2024, 1, 1))

# Load the existing SEPac SST mask and weights from ERA5 grid
mask_file = "../../data/SEPac_SST/sepac_sst_mask_and_weights.jld2"
mask_data = load(mask_file)
era5_mask = mask_data["final_mask"]
era5_weights = mask_data["area_weights"]
era5_lon = mask_data["longitude"]
era5_lat = mask_data["latitude"]

println("Loaded ERA5 mask with dimensions: ", size(era5_mask))
println("Number of valid points in ERA5 mask: ", sum(era5_mask))

# Load CERES data using the load function
ceres_variables = ["toa_net_all_mon", "gridded_net_sw", "toa_lw_all_mon"]
ceres_data, ceres_coords = load_ceres_data(ceres_variables, time_period)

# Extract coordinates and data
ceres_lat = ceres_coords["latitude"]
ceres_lon = ceres_coords["longitude"]
ceres_time = ceres_coords["time"]

println("CERES grid dimensions: ", length(ceres_lon), " x ", length(ceres_lat))
println("CERES time points: ", length(ceres_time))

# Extract CERES radiation variables
toa_net_all_mon = ceres_data["toa_net_all_mon"]
net_sw_ceres = ceres_data["gridded_net_sw"]
toa_lw_all_mon = ceres_data["toa_lw_all_mon"]

println("CERES data loaded successfully")
println("Net SW shape: ", size(net_sw_ceres))
println("LW shape: ", size(toa_lw_all_mon))
println("Net radiation shape: ", size(toa_net_all_mon))

# Downscale the ERA5 mask to CERES grid using nearest neighbor interpolation
println("Downscaling ERA5 mask to CERES grid...")

# Create the downscaled mask for CERES grid
ceres_mask = zeros(Float64, length(ceres_lon), length(ceres_lat))

# For each CERES grid point, find the nearest ERA5 grid point
for (i, clon) in enumerate(ceres_lon)
    for (j, clat) in enumerate(ceres_lat)
        # Find nearest ERA5 grid point
        lon_idx = argmin(abs.(era5_lon .- clon))
        lat_idx = argmin(abs.(era5_lat .- clat))
        
        # Copy the mask value
        ceres_mask[i, j] = era5_mask[lon_idx, lat_idx]
    end
end

# Use CERES coordinates for our calculations
lat = ceres_lat
lon = ceres_lon
final_mask = ceres_mask
bool_mask = final_mask .== 1

# Calculate cosine latitude weights for CERES grid
coslat = cosd.(lat)'
area_weights = coslat .* bool_mask
total_weights = sum(area_weights)

println("Final mask covers $(sum(bool_mask)) grid points")
println("Total cosine weights: $(total_weights)")

# Calculate cosine weighted averages over the SEPac region
println("Calculating cosine weighted averages...")

# Net SW radiation (positive values represent energy input to Earth)
sepac_net_sw = vec(sum(net_sw_ceres .* area_weights, dims = (1,2)) ./ total_weights)

# LW radiation (multiply by -1 as requested - representing energy loss from Earth)
sepac_minus_lw = vec(sum((-1 * toa_lw_all_mon) .* area_weights, dims = (1,2)) ./ total_weights)

# Net radiation (SW - LW, energy balance)
sepac_net_rad = vec(sum(toa_net_all_mon .* area_weights, dims = (1,2)) ./ total_weights)

# Create DataFrame with the results
df = DataFrame(
    Date = ceres_time,
    SEPac_Net_SW = sepac_net_sw,
    SEPac_Minus_LW = sepac_minus_lw,
    SEPac_Net_Radiation = sepac_net_rad
)

println("Data summary:")
println("Net SW: mean = $(mean(sepac_net_sw)), std = $(std(sepac_net_sw))")
println("Minus LW: mean = $(mean(sepac_minus_lw)), std = $(std(sepac_minus_lw))")
println("Net Radiation: mean = $(mean(sepac_net_rad)), std = $(std(sepac_net_rad))")

# Create detrended and deseasonalized versions
println("Creating detrended and deseasonalized versions...")

# Convert times to float for trend calculation
float_times = [calc_float_time(t) for t in ceres_time]
months = [month(t) for t in ceres_time]

# Detrend and deseasonalize each variable
sepac_net_sw_processed = copy(sepac_net_sw)
sepac_minus_lw_processed = copy(sepac_minus_lw)
sepac_net_rad_processed = copy(sepac_net_rad)

fit_net_sw = detrend_and_deseasonalize!(sepac_net_sw_processed, float_times, months)
fit_minus_lw = detrend_and_deseasonalize!(sepac_minus_lw_processed, float_times, months)
fit_net_rad = detrend_and_deseasonalize!(sepac_net_rad_processed, float_times, months)

println("Processed data summary:")
println("Net SW (detrended/deseasonalized): mean = $(mean(sepac_net_sw_processed)), std = $(std(sepac_net_sw_processed))")
println("Minus LW (detrended/deseasonalized): mean = $(mean(sepac_minus_lw_processed)), std = $(std(sepac_minus_lw_processed))")
println("Net Radiation (detrended/deseasonalized): mean = $(mean(sepac_net_rad_processed)), std = $(std(sepac_net_rad_processed))")

# Add processed data to DataFrame
df.SEPac_Net_SW_Processed = sepac_net_sw_processed
df.SEPac_Minus_LW_Processed = sepac_minus_lw_processed
df.SEPac_Net_Radiation_Processed = sepac_net_rad_processed

# Create time series plots
println("Creating plots...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=true)

# Plot Net SW
axes[0].plot(df.Date, df.SEPac_Net_SW, "b-", linewidth=1.5, label="SEPac Net SW")
axes[0].set_ylabel("Net SW (W/m²)", fontsize=12)
axes[0].set_title("SEPac Region CERES Radiation Time Series", fontsize=14)
axes[0].grid(true, alpha=0.3)
axes[0].legend()

# Plot -1*LW
axes[1].plot(df.Date, df.SEPac_Minus_LW, "r-", linewidth=1.5, label="SEPac -1×LW")
axes[1].set_ylabel("-1×LW (W/m²)", fontsize=12)
axes[1].grid(true, alpha=0.3)
axes[1].legend()

# Plot Net Radiation
axes[2].plot(df.Date, df.SEPac_Net_Radiation, "g-", linewidth=1.5, label="SEPac Net Radiation")
axes[2].set_ylabel("Net Radiation (W/m²)", fontsize=12)
axes[2].set_xlabel("Date", fontsize=12)
axes[2].grid(true, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.savefig(joinpath(visdir, "sepac_ceres_radiation_timeseries.png"), dpi=300, bbox_inches="tight")
if vis
    plt.show()
end
plt.close()

# Create time series plots for detrended and deseasonalized data
println("Creating detrended and deseasonalized plots...")

fig_processed, axes_processed = plt.subplots(3, 1, figsize=(12, 10), sharex=true)
# Calculate common y-axis limits for processed data
processed_ylim = (minimum([minimum(df.SEPac_Net_SW_Processed), minimum(df.SEPac_Minus_LW_Processed), minimum(df.SEPac_Net_Radiation_Processed)]) * 1.1, 
                  maximum([maximum(df.SEPac_Net_SW_Processed), maximum(df.SEPac_Minus_LW_Processed), maximum(df.SEPac_Net_Radiation_Processed)]) * 1.1)

# Set common y-axis limits for all processed data plots
for ax in axes_processed
    ax.set_ylim(processed_ylim)
end

# Plot Net SW (processed)
axes_processed[0].plot(df.Date, df.SEPac_Net_SW_Processed, "b-", linewidth=1.5, label="SEPac Net SW (detrended/deseasonalized)")
axes_processed[0].set_ylabel("Net SW (W/m²)", fontsize=12)
axes_processed[0].set_title("SEPac Region CERES Radiation Time Series (Detrended & Deseasonalized)", fontsize=14)
axes_processed[0].grid(true, alpha=0.3)
axes_processed[0].legend()

# Plot -1*LW (processed)
axes_processed[1].plot(df.Date, df.SEPac_Minus_LW_Processed, "r-", linewidth=1.5, label="SEPac -1×LW (detrended/deseasonalized)")
axes_processed[1].set_ylabel("-1×LW (W/m²)", fontsize=12)
axes_processed[1].grid(true, alpha=0.3)
axes_processed[1].legend()

# Plot Net Radiation (processed)
axes_processed[2].plot(df.Date, df.SEPac_Net_Radiation_Processed, "g-", linewidth=1.5, label="SEPac Net Radiation (detrended/deseasonalized)")
axes_processed[2].set_ylabel("Net Radiation (W/m²)", fontsize=12)
axes_processed[2].set_xlabel("Date", fontsize=12)
axes_processed[2].grid(true, alpha=0.3)
axes_processed[2].legend()

plt.tight_layout()
plt.savefig(joinpath(visdir, "sepac_ceres_radiation_timeseries_processed.png"), dpi=300, bbox_inches="tight")
if vis
    plt.show()
end
plt.close()

# Create comparison plots showing original vs processed data
fig_comp, axes_comp = plt.subplots(3, 2, figsize=(16, 10), sharex=true)

# Calculate common y-axis limits for processed data
processed_data_all = vcat(df.SEPac_Net_SW_Processed, df.SEPac_Minus_LW_Processed, df.SEPac_Net_Radiation_Processed)
processed_ylim = (minimum(processed_data_all) * 1.05, maximum(processed_data_all) * 1.05)

# Net SW comparison
axes_comp[0, 0].plot(df.Date, df.SEPac_Net_SW, "b-", linewidth=1.5, label="Original")
axes_comp[0, 0].set_ylabel("Net SW (W/m²)", fontsize=12)
axes_comp[0, 0].set_title("Original Net SW", fontsize=12)
axes_comp[0, 0].grid(true, alpha=0.3)
axes_comp[0, 0].legend()

axes_comp[0, 1].plot(df.Date, df.SEPac_Net_SW_Processed, "b-", linewidth=1.5, label="Detrended/Deseasonalized")
axes_comp[0, 1].set_ylabel("Net SW (W/m²)", fontsize=12)
axes_comp[0, 1].set_title("Processed Net SW", fontsize=12)
axes_comp[0, 1].set_ylim(processed_ylim)
axes_comp[0, 1].grid(true, alpha=0.3)
axes_comp[0, 1].legend()

# -1*LW comparison
axes_comp[1, 0].plot(df.Date, df.SEPac_Minus_LW, "r-", linewidth=1.5, label="Original")
axes_comp[1, 0].set_ylabel("-1×LW (W/m²)", fontsize=12)
axes_comp[1, 0].set_title("Original -1×LW", fontsize=12)
axes_comp[1, 0].grid(true, alpha=0.3)
axes_comp[1, 0].legend()

axes_comp[1, 1].plot(df.Date, df.SEPac_Minus_LW_Processed, "r-", linewidth=1.5, label="Detrended/Deseasonalized")
axes_comp[1, 1].set_ylabel("-1×LW (W/m²)", fontsize=12)
axes_comp[1, 1].set_title("Processed -1×LW", fontsize=12)
axes_comp[1, 1].set_ylim(processed_ylim)
axes_comp[1, 1].grid(true, alpha=0.3)
axes_comp[1, 1].legend()

# Net Radiation comparison
axes_comp[2, 0].plot(df.Date, df.SEPac_Net_Radiation, "g-", linewidth=1.5, label="Original")
axes_comp[2, 0].set_ylabel("Net Radiation (W/m²)", fontsize=12)
axes_comp[2, 0].set_title("Original Net Radiation", fontsize=12)
axes_comp[2, 0].set_xlabel("Date", fontsize=12)
axes_comp[2, 0].grid(true, alpha=0.3)
axes_comp[2, 0].legend()

axes_comp[2, 1].plot(df.Date, df.SEPac_Net_Radiation_Processed, "g-", linewidth=1.5, label="Detrended/Deseasonalized")
axes_comp[2, 1].set_ylabel("Net Radiation (W/m²)", fontsize=12)
axes_comp[2, 1].set_title("Processed Net Radiation", fontsize=12)
axes_comp[2, 1].set_xlabel("Date", fontsize=12)
axes_comp[2, 1].set_ylim(processed_ylim)
axes_comp[2, 1].grid(true, alpha=0.3)
axes_comp[2, 1].legend()

plt.tight_layout()
plt.savefig(joinpath(visdir, "sepac_ceres_radiation_comparison.png"), dpi=300, bbox_inches="tight")
if vis
    plt.show()
end
plt.close()

# Create a summary plot showing all three on the same axes for comparison
fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(df.Date, df.SEPac_Net_SW, "b-", linewidth=1.5, label="Net SW", alpha=0.8)
ax.plot(df.Date, df.SEPac_Minus_LW, "r-", linewidth=1.5, label="-1×LW", alpha=0.8)
ax.plot(df.Date, df.SEPac_Net_Radiation, "g-", linewidth=1.5, label="Net Radiation", alpha=0.8)

ax.set_ylabel("Radiation (W/m²)", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.set_title("SEPac Region CERES Radiation Components", fontsize=14)
ax.grid(true, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig(joinpath(visdir, "sepac_ceres_radiation_combined.png"), dpi=300, bbox_inches="tight")
if vis
    plt.show()
end
plt.close()

# Create a summary plot showing all three processed variables on the same axes
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))

ax3.plot(df.Date, df.SEPac_Net_SW_Processed, "b-", linewidth=1.5, label="Net SW (processed)", alpha=0.8)
ax3.plot(df.Date, df.SEPac_Minus_LW_Processed, "r-", linewidth=1.5, label="-1×LW (processed)", alpha=0.8)
ax3.plot(df.Date, df.SEPac_Net_Radiation_Processed, "g-", linewidth=1.5, label="Net Radiation (processed)", alpha=0.8)

ax3.set_ylabel("Radiation (W/m²)", fontsize=12)
ax3.set_xlabel("Date", fontsize=12)
ax3.set_title("SEPac Region CERES Radiation Components (Detrended & Deseasonalized)", fontsize=14)
ax3.grid(true, alpha=0.3)
ax3.legend(fontsize=11)

plt.tight_layout()
plt.savefig(joinpath(visdir, "sepac_ceres_radiation_combined_processed.png"), dpi=300, bbox_inches="tight")
if vis
    plt.show()
end
plt.close()

# Show spatial pattern of the mask for verification
fig_mask = plot_global_heatmap(lat, lon, final_mask; 
                              title = "SEPac CERES Averaging Mask", 
                              colorbar_label = "Mask")
fig_mask.savefig(joinpath(visdir, "sepac_ceres_mask.png"))
if vis
    display(fig_mask)
end
plt.close(fig_mask)

# Save the results
println("Saving results...")
CSV.write(joinpath(savedir, "sepac_ceres_flux_time_series.csv"), df)

println("Results saved to:")
println("- CSV data: $(joinpath(savedir, "sepac_ceres_flux_time_series.csv"))")
println("- Original time series plots: $(joinpath(visdir, "sepac_ceres_radiation_timeseries.png"))")
println("- Processed time series plots: $(joinpath(visdir, "sepac_ceres_radiation_timeseries_processed.png"))")
println("- Comparison plots (original vs processed): $(joinpath(visdir, "sepac_ceres_radiation_comparison.png"))")
println("- Combined original plots: $(joinpath(visdir, "sepac_ceres_radiation_combined.png"))")
println("- Combined processed plots: $(joinpath(visdir, "sepac_ceres_radiation_combined_processed.png"))")
println("- Mask verification: $(joinpath(visdir, "sepac_ceres_mask.png"))")

println("Script completed successfully!")