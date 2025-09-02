"""
Stepwise regression of global TOA net CERES radiation from cosine weighted gridded net CERES radiation.

This script performs stepwise regression using all grid points and plots:
1. The order in which grid points are added to the model
2. The skill of the model as a function of the number of points added
"""

using Dates, Statistics, NCDatasets, Plots, PythonCall, Dictionaries
@py import matplotlib.pyplot as plt

# Set visualization directory
visdir = "../../vis/ceres_stepwise_regress"

# Include required modules
cd(@__DIR__)  # Change to script directory for relative paths
include("../stepwise_regressor/stepwise_regressor.jl")
include("../utils/load_funcs.jl")
include("../utils/plot_global.jl")
include("../utils/constants.jl")
include("../utils/utilfuncs.jl")

println("Loading CERES data...")
# Load CERES data
ceres_variables = ["gtoa_net_all_mon", "toa_net_all_mon"]
ceres_data, ceres_coords = load_ceres_data(ceres_variables, time_period)

# Extract data
global_toa_net = ceres_data["gtoa_net_all_mon"]  # Global mean TOA net radiation (predictand)
gridded_toa_net = ceres_data["toa_net_all_mon"]  # Gridded TOA net radiation (predictors)
lat = ceres_coords["latitude"]
lon = ceres_coords["longitude"]
time = ceres_coords["time"]

# Convert time to float years for detrending
float_times = calc_float_time.(time)
months = month.(time)

# Detrend and deseasonalize the global data
println("Detrending and deseasonalizing global TOA net radiation...")
global_toa_net_detrended = copy(global_toa_net)
detrend_and_deseasonalize!(global_toa_net_detrended, float_times, months)

# Detrend and deseasonalize the gridded data
println("Detrending and deseasonalizing gridded TOA net radiation...")
gridded_toa_net_detrended = copy(gridded_toa_net)
n_time, n_lat, n_lon = size(gridded_toa_net)
for i in 1:n_lat, j in 1:n_lon
    slice = view(gridded_toa_net_detrended, j, i, :)
    if !any(isnan.(slice))  # Only process if no NaN values
        detrend_and_deseasonalize!(slice, float_times, months)
    end
end

# Update the variables to use detrended data
global_toa_net = global_toa_net_detrended
gridded_toa_net = gridded_toa_net_detrended



println("Data loaded successfully!")
println("Global TOA net size: ", size(global_toa_net))
println("Gridded TOA net size: ", size(gridded_toa_net))
println("Time period: ", length(time), " months")

# Calculate cosine weights for area weighting
lat_rad = deg2rad.(lat)
cos_weights = cos.(lat_rad)

# Create cosine weight matrix (broadcast across longitude)
weight_matrix = cos_weights' .* ones(length(lon), 1)  # (lon, lat)

# Apply cosine weights to gridded data
weighted_gridded_data = gridded_toa_net .* weight_matrix

# Reshape gridded data: from (lon, lat, time) to (time, lon*lat)
n_time, n_lon, n_lat = length(time), length(lon), length(lat)
n_grid_points = n_lon * n_lat

# Reshape to (time, grid_points)
predictors = reshape(weighted_gridded_data, n_grid_points, n_time)'

println("Predictors matrix size: ", size(predictors))
println("Predictand vector size: ", size(global_toa_net))

# Remove any NaN values
valid_time_indices = .!isnan.(global_toa_net)
predictors_clean = predictors[valid_time_indices, :]
predictand_clean = global_toa_net[valid_time_indices]

# Also remove grid points that have any NaN values
valid_grid_indices = [!any(isnan.(predictors_clean[:, i])) for i in 1:size(predictors_clean, 2)]
predictors_final = predictors_clean[:, valid_grid_indices]

println("After removing NaN values:")
println("Valid time points: ", sum(valid_time_indices))
println("Valid grid points: ", sum(valid_grid_indices))
println("Final predictors size: ", size(predictors_final))

# Create index mapping from valid grid points back to (lon, lat) coordinates
valid_indices = findall(valid_grid_indices)
all_indices = 1:length(valid_indices)

# Define evaluation function for stepwise regression
function correlation_eval(current_indices, predictors, predictand)
    if isempty(current_indices)
        return 0.0
    end
    combined_predictor = sum(@view(predictors[:, current_indices]), dims=2)
    return cor(vec(combined_predictor), predictand)
end

println("Starting stepwise regression...")
# Perform stepwise regression
selected_indices, skill_values = stepwise_regress(
    predictors_final, 
    predictand_clean, 
    correlation_eval, 
    all_indices;
    max_num = 50,  # Limit to first 50 points for computational efficiency
    min_gain = 0.001  # Minimum improvement in correlation
)

println("Stepwise regression completed!")
println("Number of selected points: ", length(selected_indices))
println("Final correlation: ", skill_values[end])

# Convert selected indices back to spatial coordinates
function index_to_coords(idx, n_lon, n_lat)
    # Convert linear index to (lon_idx, lat_idx)
    linear_idx = valid_indices[idx]  # Map back to original grid
    lat_idx = div(linear_idx - 1, n_lon) + 1
    lon_idx = mod(linear_idx - 1, n_lon) + 1
    return lon_idx, lat_idx
end

# Create order matrix showing when each point was added
order_matrix = fill(0.0, n_lon, n_lat)
for (step, idx) in enumerate(selected_indices)
    lon_idx, lat_idx = index_to_coords(idx, n_lon, n_lat)
    order_matrix[lon_idx, lat_idx] = step
end

# Create visualization directory if it doesn't exist
if !isdir(visdir)
    mkpath(visdir)
    println("Created directory: $visdir")
end

# Plot 1: Order of selection
println("Creating plots...")
fig1 = plot_global_heatmap(
    lat, lon, order_matrix;
    title="Order of Grid Point Selection in Stepwise Regression",
    colorbar_label="Selection Step",
    cmap=plt.cm.viridis
)
plt.savefig("$visdir/stepwise_selection_order.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

# Plot 2: Model skill as function of number of points
plt.figure(figsize=(10, 6))
plt.plot(1:length(skill_values), skill_values, "o-", linewidth=2, markersize=4)
plt.xlabel("Number of Grid Points")
plt.ylabel("Correlation with Global TOA Net Radiation")
plt.title("Stepwise Regression Model Skill")
plt.grid(true, alpha=0.3)
plt.tight_layout()
plt.savefig("$visdir/stepwise_model_skill.png", dpi=300, bbox_inches="tight")
plt.close()

# Create importance map (inverse of selection order - earlier = more important)
importance_matrix = fill(NaN, n_lon, n_lat)
max_step = length(selected_indices)
for (step, idx) in enumerate(selected_indices)
    lon_idx, lat_idx = index_to_coords(idx, n_lon, n_lat)
    importance_matrix[lon_idx, lat_idx] = max_step - step + 1  # Higher values = more important
end

# Plot 3: Importance map
fig3 = plot_global_heatmap(
    lat, lon, importance_matrix;
    title="Grid Point Importance (Inverse Selection Order)",
    colorbar_label="Importance Rank",
    cmap=plt.cm.plasma
)
plt.savefig("$visdir/stepwise_importance_map.png", dpi=300, bbox_inches="tight")
plt.close(fig3)

# Summary statistics
println("\n=== Stepwise Regression Summary ===")
println("Total valid grid points: ", sum(valid_grid_indices))
println("Number of points selected: ", length(selected_indices))
println("Final model correlation: ", round(skill_values[end], digits=4))
println("Correlation with first point: ", round(skill_values[1], digits=4))
println("Improvement from stepwise: ", round(skill_values[end] - skill_values[1], digits=4))

# Find coordinates of most important points
println("\nTop 10 most important grid points:")
for i in 1:min(10, length(selected_indices))
    idx = selected_indices[i]
    lon_idx, lat_idx = index_to_coords(idx, n_lon, n_lat)
    println("Step $i: Lon = ", round(lon[lon_idx], digits=2), "°, Lat = ", round(lat[lat_idx], digits=2), "°")
end

println("\nPlots saved to $visdir/")
println("Script completed successfully!")

# Calculate correlation between each grid point and global mean
println("Calculating correlations between each grid point and global mean...")
correlation_matrix = fill(NaN, n_lon, n_lat)

correlation_matrix = cor.(eachslice(weighted_gridded_data, dims=(1,2)), Ref(global_toa_net))[:,:]

# Plot 4: Correlation map
fig4 = plot_global_heatmap(
    lat, lon, correlation_matrix;
    title="Correlation between Local Grid Points and Global Mean TOA Net Radiation",
    colorbar_label="Correlation Coefficient",
    cmap=plt.cm.RdBu_r
)
plt.savefig("$visdir/gridpoint_global_correlations.png", dpi=300, bbox_inches="tight")
plt.close(fig4)

println("Correlation map saved to $visdir/gridpoint_global_correlations.png")
