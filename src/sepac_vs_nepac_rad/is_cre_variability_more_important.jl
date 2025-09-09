"""
This script plots the standard deviation of cloud radiative effects over time using the CERES data, 
and displays the same latlon bounding boxes as in determine_cre_regions.jl to visualize 
where the SEPac, NEPac, and SEAtl stratocumulus regions are located relative to CRE variability.
"""

cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/compare_stratocumulus_regions"

# Create directories if they don't exist
mkpath(visdir)

net_cre_var = "toa_cre_net_mon"
time_period = (Date(2000, 3), Date(2025, 4)) # in time period does not include the right endpoint

ceres_data, ceres_coords = load_new_ceres_data([net_cre_var], time_period)
ceres_time = ceres_coords["time"]
ceres_float_time = calc_float_time.(ceres_time)
ceres_month_groups = groupfind(month.(ceres_time))

net_rad = ceres_data[net_cre_var]

detrend_and_deseasonalize_precalculated_groups!.(eachslice(net_rad; dims = (1,2)), Ref(ceres_float_time), Ref(ceres_month_groups))

# Calculate standard deviation over time dimension instead of mean
std_cre = mapslices(std, net_rad, dims=(3,))[:,:,1]  # std over time dimension
lat = ceres_coords["latitude"]
lon = ceres_coords["longitude"]

fig = plot_global_heatmap(lat, lon, std_cre; 
                          title="Standard Deviation of Cloud Radiative Effect (2000-2025)", 
                          colorbar_label="W/m²", central_longitude = 160)

ax = fig.axes[0]

# Define the same bounds as in determine_cre_regions.jl
bounds = Dict{String, Dict{String, Float64}}()

# SEPac bounds: sepac_lon, sepac_lat = ((-110, -69.3) .+ 360, (-40, 0))
bounds["SEPac"] = Dict("lat_min"=>-34, "lat_max"=>0, "lon_min"=>250, "lon_max"=>290.7)
bounds["NEPac"] = Dict("lat_min"=>15, "lat_max"=>38, "lon_min"=>210, "lon_max"=>260)
bounds["SEAtl"] = Dict("lat_min"=>-30, "lat_max"=>-7, "lon_min"=>-15, "lon_max"=>15)

# Plot the bounds as rectangles on the map
@py import matplotlib.patches as patches

# Define colors for each region (same as determine_cre_regions.jl)
colorsdict = Dict("SEPac" => "red", "NEPac" => "blue", "SEAtl" => "green")

for (region_name, region_bounds) in bounds
    # Create rectangle coordinates
    lon_min = region_bounds["lon_min"]
    lon_max = region_bounds["lon_max"]
    lat_min = region_bounds["lat_min"]
    lat_max = region_bounds["lat_max"]
    
    # Create rectangle patch
    rect = patches.Rectangle((lon_min, lat_min), 
                           lon_max - lon_min, 
                           lat_max - lat_min,
                           linewidth=2, 
                           edgecolor=colorsdict[region_name], 
                           facecolor="none",
                           transform=ccrs.PlateCarree(),
                           label=region_name)
    
    ax.add_patch(rect)
end

# Add legend
ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

# Save the figure
plt.savefig(joinpath(visdir, "cre_std_with_bounding_boxes.png"), dpi=300, bbox_inches="tight")
println("Saved figure: cre_std_with_bounding_boxes.png")

plt.close(fig)