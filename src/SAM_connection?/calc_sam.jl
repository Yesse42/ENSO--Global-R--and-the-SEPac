cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/gridded_effects_utils.jl")

using CSV, DataFrames, Dates, Dictionaries, PythonCall, SplitApplyCombine, Statistics, LinearAlgebra
@py import matplotlib.pyplot as plt, cartopy.crs as ccrs, matplotlib.colors as colors, cmasher as cmr, matplotlib.path as mpath, numpy as np

datadir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/SAM"
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/SAM"
mkpath(visdir)

time_period = (Date("1980-01-01"), Date("2025-03-31"))

# Load z data
z_data, z_coords = load_era5_data(["z"], time_period; pressure_level_file = "new_pressure_levels.nc")

times = z_coords["pressure_time"]
float_times = calc_float_time.(times)
precalculated_groups = groupfind(month, times)
z = z_data["z"]
hpa_700_idx = findfirst(x -> x == 700, z_coords["pressure_level"])
z_700 = z[:,:,hpa_700_idx,:]

# Deseasonalize only (no detrending)
for slice in eachslice(z_700, dims = (1,2))
    deseasonalize_precalculated_groups(slice, precalculated_groups)
end

# Get latitude and longitude
lats = z_coords["latitude"]
lons = z_coords["longitude"]

# Restrict to Southern Hemisphere (20S to 90S)
sh_mask = lats .<= -20
sh_lats = lats[sh_mask]
z_700_sh = z_700[:, sh_mask, :]

println("Southern Hemisphere data shape: ", size(z_700_sh))
println("Latitude range: $(minimum(sh_lats)) to $(maximum(sh_lats))")

# Apply sqrt(cos(lat)) weighting for EOF analysis
# Create weight matrix for each grid point
lat_weights = sqrt.(cosd.(sh_lats))  # 1D array for latitudes
weight_matrix = ones(length(lons)) * lat_weights'  # lon x lat matrix

# Apply weights to data
z_700_sh_weighted = copy(z_700_sh)
for t in 1:size(z_700_sh, 3)
    z_700_sh_weighted[:, :, t] .*= weight_matrix
end

# Reshape for EOF: (space x time)
reshaped_zs = reshape(z_700_sh_weighted, :, size(z_700_sh_weighted, 3))

# Remove missing values - create mask for valid grid points
valid_mask = vec(.!any(ismissing.(reshaped_zs), dims=2))
reshaped_zs_valid = reshaped_zs[valid_mask, :]

println("Valid grid points: $(sum(valid_mask)) out of $(length(valid_mask))")

# Convert to Float64 and handle any remaining issues
reshaped_zs_valid = Float64.(reshaped_zs_valid)

# Calculate EOF using SVD
println("Calculating EOF via SVD...")
U, S, Vt = svd(reshaped_zs_valid')  # Transpose so time is rows

# First PC time series (already standardized from SVD)
pc1_timeseries = U[:, 1] .* S[1]

# First EOF spatial pattern (needs to be mapped back to grid)
eof1_pattern_valid = Vt[:, 1]

# Reconstruct full spatial pattern with missing values
eof1_pattern_full = fill(NaN, length(valid_mask))
eof1_pattern_full[valid_mask] = eof1_pattern_valid

# Reshape to 2D grid
eof1_2d = reshape(eof1_pattern_full, length(lons), length(sh_lats))

# Remove the weighting for visualization
eof1_2d_unweighted = eof1_2d ./ weight_matrix

# Calculate variance explained
total_var = sum(S.^2)
var_explained = (S[1]^2 / total_var) * 100
println("Variance explained by PC1: $(round(var_explained, digits=2))%")

# Standardize PC1 for comparison
pc1_standardized = (pc1_timeseries .- mean(pc1_timeseries)) ./ std(pc1_timeseries)

# Load original SAM data
sam_df = CSV.read(joinpath(datadir, "original_sam.csv"), DataFrame)
sam_df[!, :date] = Date.(sam_df.year, sam_df.month, sam_df.day)

# Create monthly averages for both time series
# First, create monthly PC1 dataframe
pc1_df = DataFrame(date = times, pc1 = pc1_standardized)
pc1_df[!, :year] = year.(pc1_df.date)
pc1_df[!, :month] = month.(pc1_df.date)
pc1_monthly = combine(groupby(pc1_df, [:year, :month]), :pc1 => mean => :pc1_mean)
pc1_monthly[!, :date] = Date.(pc1_monthly.year, pc1_monthly.month, 1)

# Create monthly SAM averages
sam_df[!, :year] = year.(sam_df.date)
sam_df[!, :month] = month.(sam_df.date)
sam_monthly = combine(groupby(sam_df, [:year, :month]), :aao_index_cdas => mean => :sam_mean)
sam_monthly[!, :date] = Date.(sam_monthly.year, sam_monthly.month, 1)

# Merge the two time series
comparison_df = DataFrames.innerjoin(pc1_monthly, sam_monthly, on=:date, makeunique = true)

# Calculate correlation
function skipmissing_corr(x,y)
    valid_x = .!ismissing.(x)
    valid_y = .!ismissing.(y)
    valid_indices = valid_x .& valid_y
    return cor(x[valid_indices], y[valid_indices])
end
correlation = skipmissing_corr(comparison_df.pc1_mean, comparison_df.sam_mean)
println("\nCorrelation between calculated PC1 and original SAM: $(round(correlation, digits=4))")

# Flip sign if necessary (EOF sign is arbitrary)
if correlation < 0
    println("Flipping sign of PC1 to match SAM convention...")
    pc1_standardized .*= -1
    eof1_2d_unweighted .*= -1
    comparison_df.pc1_mean .*= -1
    correlation = -correlation
end

println("Correlation after sign adjustment: $(round(correlation, digits=4))")

# Save the calculated SAM index
output_df = DataFrame(
    year = year.(times),
    month = month.(times),
    day = 1,
    sam_calculated = pc1_standardized
)
output_path = joinpath(datadir, "calculated_sam.csv")
CSV.write(output_path, output_df)
println("Saved calculated SAM index to: $output_path")

#===========================================
    PLOTTING
===========================================#

using Plots
Plots.gr()

# 1. Plot comparison of PC1 vs original SAM using Plots.jl
p1 = plot(comparison_df.date, comparison_df.pc1_mean, 
          label="Calculated PC1", linewidth=1.5, alpha=0.8,
          xlabel="Date", ylabel="Index Value",
          title="Comparison of Calculated PC1 and Original SAM Index (Monthly Mean)",
          legend=:topleft, grid=true, gridalpha=0.3)
plot!(p1, comparison_df.date, comparison_df.sam_mean, 
      label="Original SAM", linewidth=1.5, alpha=0.8)

# Scatter plot
p2 = scatter(comparison_df.sam_mean, comparison_df.pc1_mean, 
             alpha=0.5, markersize=3, label="",
             xlabel="Original SAM Index", ylabel="Calculated PC1",
             title="Scatter Plot: PC1 vs SAM (r = $(round(correlation, digits=4)))",
             grid=true, gridalpha=0.3, aspect_ratio=:equal)
plot!(p2, [-4, 4], [-4, 4], linestyle=:dash, linewidth=1, 
      color=:red, label="1:1 line")

# Combine plots
combined_plot = plot(p1, p2, layout=(2,1), size=(1000, 800))
savefig(combined_plot, joinpath(visdir, "pc1_vs_original_sam_comparison.png"))
println("Saved comparison plot to: $(joinpath(visdir, "pc1_vs_original_sam_comparison.png"))")

# 2. Plot spatial pattern of EOF1 on Southern Hemisphere polar projection
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())

# Set extent to Southern Hemisphere
ax.set_extent([-180, 180, -90, -20], ccrs.PlateCarree())

# Add coastlines and gridlines
ax.coastlines()
gl = ax.gridlines(draw_labels=false, linewidth=0.5, alpha=0.5, linestyle="--")

# Create circular boundary for polar plot
theta = pyimport("numpy").linspace(0, 2*pyimport("numpy").pi, 100)
center, radius = np.array([0.5, 0.5]), 0.5
verts = pyimport("numpy").vstack([pyimport("numpy").sin(theta), pyimport("numpy").cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

# Calculate color normalization
absmax = pyimport("numpy").nanmax(pyimport("numpy").abs(eof1_2d_unweighted))
colornorm = colors.Normalize(vmin=-absmax, vmax=absmax)

# Plot the EOF pattern
c = ax.contourf(lons, sh_lats, eof1_2d_unweighted', 
               transform=ccrs.PlateCarree(), 
               cmap=cmr.prinsenvlag.reversed(), 
               levels=21, 
               norm=colornorm)

# Add colorbar
cbar = plt.colorbar(c, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
cbar.set_label("EOF1 Pattern (Z700 anomaly)", fontsize=12)

ax.set_title("Leading EOF of Z700 (20°S-90°S)\nVariance Explained: $(round(var_explained, digits=1))%", 
            fontsize=14, fontweight="bold", pad=20)

fig.savefig(joinpath(visdir, "eof1_spatial_pattern_polar.png"), dpi=300, bbox_inches="tight")
println("Saved EOF spatial pattern to: $(joinpath(visdir, "eof1_spatial_pattern_polar.png"))")
plt.close(fig)

println("\nAnalysis complete!")