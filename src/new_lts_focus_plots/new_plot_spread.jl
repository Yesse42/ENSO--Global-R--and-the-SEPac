"""
GSS Plots
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")
include("../pls_regressor/pls_functions.jl")

using CSV, DataFrames, Plots
pythonplot()

deseasonalize_and_detrend_precalculated_groups_twice!(slice, float_times, idx_groups; aggfunc = mean, trendfunc = least_squares_fit) = for _ in 1:2
    deseasonalize_and_detrend_precalculated_groups!(slice, float_times, idx_groups; aggfunc, trendfunc)
    nothing
end

calc_theta(T, p) = T .* (1000 ./ p) .^ 0.286

time_period = (Date(2000, 3), Date(2024, 3, 31))
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/new_lts_focus_plots"
!isdir(visdir) && mkpath(visdir)

ceres_global_rad, grad_times = load_new_ceres_data(["gtoa_net_all_mon"], time_period)
ceres_global_rad = ceres_global_rad["gtoa_net_all_mon"]
global_rad_times = grad_times["time"]
global_rad_times = round.(global_rad_times, Month(1), RoundDown)

ceres_vars = ["toa_net_all_mon", "toa_net_lw_mon", "toa_net_sw_mon"]
ceres_gridded_rad, gridded_rad_coords = load_new_ceres_data(ceres_vars, time_period)
gridded_rad_times = gridded_rad_coords["time"]
gridded_rad_times = round.(gridded_rad_times, Month(1), RoundDown)
ceres_lats = gridded_rad_coords["latitude"]
ceres_lons = gridded_rad_coords["longitude"]

# Load in the ERA5 data using load_funcs.jl
era5_vars = pressure_level_vars = ["t", "t2m"]
era5_data, era5_coords = load_era5_data(era5_vars, time_period, pressure_level_file = "new_pressure_levels.nc")
era5_lat = era5_coords["latitude"]
era5_lon = era5_coords["longitude"]
era5_time = round.(era5_coords["pressure_time"], Month(1), RoundDown)

#Calculate theta_1000, theta_700, and LTS 
p_1000_idx = findfirst(era5_coords["pressure_level"] .== 1000)
p_700_idx = findfirst(era5_coords["pressure_level"] .== 700)
theta_1000 = calc_theta.(era5_data["t"][:,:, p_1000_idx, :], 1000)
theta_700 = calc_theta.(era5_data["t"][:,:, p_700_idx, :], 700)
LTS = theta_700 .- theta_1000
EIS = calculate_EIS.(era5_data["t2m"], era5_data["t"][:,:, p_700_idx, :])

#Add them to the data variables
era5_sl_var_names = ["LTS", "theta_1000", "theta_700", "t2m", "EIS"]
era5_single_level_data = Dictionary(era5_sl_var_names, [LTS, theta_1000, theta_700, era5_data["t2m"], EIS])

#Load in the ENSO index data
enso_data, enso_time = load_enso_data(time_period)
enso_time = enso_time["time"]
enso_time = round.(enso_time, Month(1), RoundDown)

#Load in the raw sepac idxs
sepac_datadir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/local_region_time_series"
ceres_var_file =joinpath(sepac_datadir, "ceres_region_avg_SEPac_feedback_definition.csv")
era5_var_file = joinpath(sepac_datadir, "era5_region_avg_SEPac_feedback_definition.csv")
#Load in the data
ceres_local = CSV.read(ceres_var_file, DataFrame)
ceres_local[!, :date] = round.(ceres_local[!, :date], Month(1), RoundDown)
era5_local = CSV.read(era5_var_file, DataFrame)
sepac_local_df = DataFrames.innerjoin(ceres_local, era5_local; on = :date)
sepac_local_times = sepac_local_df.date

#Create common time vector and associated variables
common_times = intersect(global_rad_times, gridded_rad_times, era5_time, enso_time, sepac_local_times)
float_times = calc_float_time.(common_times)
month_groups = SplitApplyCombine.groupfind(month, common_times)
months = month.(common_times)

#For each dataset, filter to common times. Then, for all datasets except ENSO, detrend_and_deseasonalize_precalculated_groups_twice
ceres_global_rad = ceres_global_rad[in.(global_rad_times, Ref(common_times))]
deseasonalize_and_detrend_precalculated_groups_twice!(ceres_global_rad, float_times, month_groups)
for var in keys(ceres_gridded_rad)
    ceres_gridded_rad[var] = ceres_gridded_rad[var][:, :, in.(gridded_rad_times, Ref(common_times))]
    deseasonalize_and_detrend_precalculated_groups_twice!.(eachslice(ceres_gridded_rad[var], dims=(1, 2)), Ref(float_times), Ref(month_groups))
end
for var in keys(era5_single_level_data)
    era5_single_level_data[var] = era5_single_level_data[var][:, :, in.(era5_time, Ref(common_times))]
    deseasonalize_and_detrend_precalculated_groups_twice!.(eachslice(era5_single_level_data[var], dims=(1, 2)), Ref(float_times), Ref(month_groups))
end
for var in keys(enso_data)
    enso_data[var] = enso_data[var][in.(enso_time, Ref(common_times))]
end
filter!(row -> in(row.date, common_times), sepac_local_df)
for colname in names(sepac_local_df)
    if colname ≠ "date"
        deseasonalize_and_detrend_precalculated_groups_twice!(sepac_local_df[!, colname], float_times, month_groups)
    end
end

#Lastly load in the masks 
using JLD2
mask_dict = load("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/stratocumulus_region_masks.jld2")
sepac_mask = mask_dict["regional_masks_era5"]["SEPac_feedback_definition"]

# Create 2x2 plot showing correlations between global radiation and atmospheric variables
# Calculate correlations for each atmospheric variable with global radiation
corrs_theta_1000_r = cor.(Ref(ceres_global_rad), eachslice(era5_single_level_data["theta_1000"], dims=(1,2)))
corrs_theta_700_r = cor.(Ref(ceres_global_rad), eachslice(era5_single_level_data["theta_700"], dims=(1,2)))
corrs_lts_r = cor.(Ref(ceres_global_rad), eachslice(era5_single_level_data["LTS"], dims=(1,2)))
corrs_eis_r = cor.(Ref(ceres_global_rad), eachslice(era5_single_level_data["EIS"], dims=(1,2)))

# Create 2x2 subplot layout
fig_2x2, axes_2x2 = plt.subplots(2, 2; figsize=(16, 12), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)), layout = "compressed")

# Plot data and titles for each subplot
plot_data = [corrs_theta_1000_r, corrs_theta_700_r, corrs_lts_r, corrs_eis_r]
plot_titles = ["Correlation between θ₁₀₀₀ and Global Net Radiation", 
               "Correlation between θ₇₀₀ and Global Net Radiation",
               "Correlation between LTS and Global Net Radiation",
               "Correlation between EIS and Global Net Radiation"]

# Calculate common color scale for all plots
max_abs_val_2x2 = max(maximum.([abs.(extrema(data)) for data in plot_data])...)
colornorm_2x2 = colors.Normalize(vmin=-max_abs_val_2x2, vmax=max_abs_val_2x2)

# Create each subplot
for i in 1:2, j in 1:2
    plot_idx = (i-1)*2 + j
    ax = axes_2x2[i-1, j-1]
    
    im = plot_global_heatmap_on_ax!(ax, era5_lat, era5_lon, plot_data[plot_idx]; 
                                   title = plot_titles[plot_idx], colornorm=colornorm_2x2)
    
    # Add contour of SEPac region mask
    contour = ax.contour(era5_lon, era5_lat, sepac_mask'; levels=[0.5], colors="black", linewidths=1.5, transform=ccrs.PlateCarree())
end

# Add single colorbar for all subplots
cbar_2x2 = fig_2x2.colorbar(cm.ScalarMappable(norm=colornorm_2x2, cmap=cmr.prinsenvlag.reversed()), ax=axes_2x2, orientation="horizontal", pad=0.08, shrink=0.8)
cbar_2x2.set_label("Correlation Coefficient", fontsize=12)

# Add suptitle
fig_2x2.suptitle("Atmospheric Variables vs Global Net Radiation Correlations", fontsize=16, y=0.98)

# Save the plot
fig_2x2.savefig(joinpath(visdir, "atmospheric_vars_vs_global_radiation_2x2.png"), dpi=300, bbox_inches="tight")
plt.close(fig_2x2)

println("2x2 correlation plot saved to: $(joinpath(visdir, "atmospheric_vars_vs_global_radiation_2x2.png"))")

# Create SEPac LTS and EIS time series plot
# Extract SEPac LTS and EIS data from the local dataframe
sepac_lts = sepac_local_df[!, :LTS_1000]  # LTS calculated from 1000 hPa
sepac_eis = sepac_local_df[!, :EIS]       # EIS data
sepac_dates = sepac_local_df[!, :date]

# Calculate correlation between LTS and EIS
lts_eis_correlation = cor(sepac_lts, sepac_eis)

# Create the plot using Plots.jl
p_sepac = plot(sepac_dates, sepac_lts, 
              label="LTS", 
              linewidth=2, 
              color=:blue,
              title="SEPac LTS and EIS Time Series (r = $(round(lts_eis_correlation, digits=3)))",
              xlabel="Date",
              ylabel="Temperature (K)",
              size=(1000, 600),
              legend=:topright)

plot!(p_sepac, sepac_dates, sepac_eis, 
      label="EIS", 
      linewidth=2, 
      color=:red)

# Save the SEPac LTS and EIS plot
savefig(p_sepac, joinpath(visdir, "sepac_lts_eis_timeseries.png"))

println("SEPac LTS and EIS time series plot saved to: $(joinpath(visdir, "sepac_lts_eis_timeseries.png"))")

# Also create a scatter plot of LTS vs EIS
p_scatter = scatter(sepac_lts, sepac_eis,
                   title="SEPac LTS vs EIS Scatter Plot (r = $(round(lts_eis_correlation, digits=3)))",
                   xlabel="LTS (K)",
                   ylabel="EIS (K)",
                   alpha=0.6,
                   markersize=3,
                   color=:darkgreen,
                   size=(600, 600),
                   legend=false)

# Add trend line
if length(sepac_lts) > 1
    x_mean = mean(sepac_lts)
    y_mean = mean(sepac_eis)
    slope = sum((sepac_lts .- x_mean) .* (sepac_eis .- y_mean)) / sum((sepac_lts .- x_mean).^2)
    intercept = y_mean - slope * x_mean
    lts_range = range(minimum(sepac_lts), maximum(sepac_lts), length=100)
    trend_line = intercept .+ slope .* lts_range
    plot!(p_scatter, lts_range, trend_line, color=:red, linewidth=2, label="")
end

# Save the scatter plot
savefig(p_scatter, joinpath(visdir, "sepac_lts_vs_eis_scatter.png"))

println("SEPac LTS vs EIS scatter plot saved to: $(joinpath(visdir, "sepac_lts_vs_eis_scatter.png"))")

# Print some summary statistics
println("SEPac LTS and EIS Summary:")
println("  LTS mean: $(round(mean(sepac_lts), digits=3)) K")
println("  LTS std: $(round(std(sepac_lts), digits=3)) K")
println("  EIS mean: $(round(mean(sepac_eis), digits=3)) K") 
println("  EIS std: $(round(std(sepac_eis), digits=3)) K")
println("  LTS-EIS correlation: $(round(lts_eis_correlation, digits=4))")

# Construct ENSO and ENSO residual components of SEPac theta_1000 and theta_700
enso_X = hcat([enso_data["oni_lag_$lag"] for lag in -24:24]...)
nonmissing_enso_idx = findall(x -> !(any(ismissing, x)), eachrow(enso_X))
enso_X = enso_X[nonmissing_enso_idx, :]

# Filter SEPac data to non-missing ENSO times
sepac_theta_1000_filtered = sepac_local_df[nonmissing_enso_idx, :θ_1000]
sepac_theta_700_filtered = sepac_local_df[nonmissing_enso_idx, :θ_700]
sepac_dates_filtered = sepac_local_df[nonmissing_enso_idx, :date]

# Use PLS to decompose SEPac theta_1000 into ENSO and residual components
println("Decomposing SEPac θ₁₀₀₀ using PLS...")
sepac_theta_1000_pls = make_pls_regressor(enso_X, sepac_theta_1000_filtered, 1; print_updates=false)
sepac_theta_1000_enso_component = vec(predict(sepac_theta_1000_pls, enso_X))
sepac_theta_1000_residual = sepac_theta_1000_filtered .- sepac_theta_1000_enso_component

# Use PLS to decompose SEPac theta_700 into ENSO and residual components
println("Decomposing SEPac θ₇₀₀ using PLS...")
sepac_theta_700_pls = make_pls_regressor(enso_X, sepac_theta_700_filtered, 1; print_updates=false)
sepac_theta_700_enso_component = vec(predict(sepac_theta_700_pls, enso_X))
sepac_theta_700_residual = sepac_theta_700_filtered .- sepac_theta_700_enso_component

# Calculate R² values for the decompositions
theta_1000_r2 = 1 - sum((sepac_theta_1000_residual).^2) / sum((sepac_theta_1000_filtered .- mean(sepac_theta_1000_filtered)).^2)
theta_700_r2 = 1 - sum((sepac_theta_700_residual).^2) / sum((sepac_theta_700_filtered .- mean(sepac_theta_700_filtered)).^2)

println("SEPac θ₁₀₀₀ ENSO component R²: $(round(theta_1000_r2, digits=4))")
println("SEPac θ₇₀₀ ENSO component R²: $(round(theta_700_r2, digits=4))")

# Calculate correlations for each component pair
full_theta_corr = cor(sepac_theta_1000_filtered, sepac_theta_700_filtered)
enso_components_corr = cor(sepac_theta_1000_enso_component, sepac_theta_700_enso_component)
residual_components_corr = cor(sepac_theta_1000_residual, sepac_theta_700_residual)

println("Correlations:")
println("  Full θ₁₀₀₀ vs θ₇₀₀: $(round(full_theta_corr, digits=4))")
println("  ENSO components: $(round(enso_components_corr, digits=4))")
println("  Residual components: $(round(residual_components_corr, digits=4))")

theta_1000_common_name = "θ_1000"
theta_700_common_name = "θ_700"

# Create three-pane scatterplot using Plots.jl
p1 = scatter(sepac_theta_1000_filtered, sepac_theta_700_filtered,
             title="Full Data (r = $(round(full_theta_corr, digits=3)))",
             xlabel="$theta_1000_common_name (K)",
             ylabel="$theta_700_common_name (K)",
             alpha=0.6,
             markersize=3,
             color=:blue,
             legend=false)

# Add trend line for full data
if length(sepac_theta_1000_filtered) > 1
    x_mean = mean(sepac_theta_1000_filtered)
    y_mean = mean(sepac_theta_700_filtered)
    slope = sum((sepac_theta_1000_filtered .- x_mean) .* (sepac_theta_700_filtered .- y_mean)) / sum((sepac_theta_1000_filtered .- x_mean).^2)
    intercept = y_mean - slope * x_mean
    x_range = range(minimum(sepac_theta_1000_filtered), maximum(sepac_theta_1000_filtered), length=100)
    trend_line = intercept .+ slope .* x_range
    plot!(p1, x_range, trend_line, color=:red, linewidth=2, label="")
end

p2 = scatter(sepac_theta_1000_enso_component, sepac_theta_700_enso_component,
             title="ENSO Components (r = $(round(enso_components_corr, digits=3)))",
             xlabel="$theta_1000_common_name (K)",
             ylabel="$theta_700_common_name (K)",
             alpha=0.6,
             markersize=3,
             color=:red,
             legend=:topleft,
             label = "")

# Add 1 to 1 line for ENSO components
if length(sepac_theta_1000_enso_component) > 1
    x_range = range(minimum(sepac_theta_1000_enso_component), maximum(sepac_theta_1000_enso_component), length=100)
    trend_line = x_range  # 1 to 1 line
    plot!(p2, x_range, trend_line, color=:black, linewidth=2, label="1 to 1 fit", aspect = :equal)
end

#Add a fit line to the enso components plot too 
if length(sepac_theta_1000_enso_component) > 1
    x_mean = mean(sepac_theta_1000_enso_component)
    y_mean = mean(sepac_theta_700_enso_component)
    slope = sum((sepac_theta_1000_enso_component .- x_mean) .* (sepac_theta_700_enso_component .- y_mean)) / sum((sepac_theta_1000_enso_component .- x_mean).^2)
    intercept = y_mean - slope * x_mean
    x_range = range(minimum(sepac_theta_1000_enso_component), maximum(sepac_theta_1000_enso_component), length=100)
    fit_line = intercept .+ slope .* x_range
    plot!(p2, x_range, fit_line, color=:green, linewidth=2, label="Actual fit", aspect = :equal)
end

p3 = scatter(sepac_theta_1000_residual, sepac_theta_700_residual,
             title="ENSO Residuals (r = $(round(residual_components_corr, digits=3)))",
             xlabel="$theta_1000_common_name (K)",
             ylabel="$theta_700_common_name (K)",
             alpha=0.6,
             markersize=3,
             color=:green,
             legend=false)

# Add trend line for residuals
if length(sepac_theta_1000_residual) > 1
    x_mean = mean(sepac_theta_1000_residual)
    y_mean = mean(sepac_theta_700_residual)
    slope = sum((sepac_theta_1000_residual .- x_mean) .* (sepac_theta_700_residual .- y_mean)) / sum((sepac_theta_1000_residual .- x_mean).^2)
    intercept = y_mean - slope * x_mean
    x_range = range(minimum(sepac_theta_1000_residual), maximum(sepac_theta_1000_residual), length=100)
    trend_line = intercept .+ slope .* x_range
    plot!(p3, x_range, trend_line, color=:black, linewidth=2, label="")
end

# Combine into single figure with 1x3 layout
fig_sepac_theta = plot(p1, p2, p3, layout=(1, 3), size=(1500, 500))
plot!(fig_sepac_theta, suptitle="ENSO effect on SEPac Temperatures at 1000 hPa and 700 hPa ")

# Save the three-pane scatterplot
savefig(fig_sepac_theta, joinpath(visdir, "sepac_theta_enso_decomposition_scatterplots.png"))

println("SEPac theta ENSO decomposition scatterplot saved to: $(joinpath(visdir, "sepac_theta_enso_decomposition_scatterplots.png"))")

#Now make plot Theta_1000 vs global R
corrs_theta_1000_r = cor.(Ref(ceres_global_rad), eachslice(era5_single_level_data["theta_1000"], dims=(1,2)))

fig = plot_global_heatmap(era5_lat, era5_lon, corrs_theta_1000_r; title = "Correlation between surface temperature and \nthe global radiative imbalance, 2000-2024", colorbar_label = "Correlation Coefficient")
ax = fig.axes[0]
ax.title.set_fontsize(16)
#Contour in the region mask
contour = ax.contour(era5_lon, era5_lat, sepac_mask'; levels=[0.5], colors="black", linewidths=1.5, transform=ccrs.PlateCarree())
fig.savefig(joinpath(visdir, "theta_1000_vs_global_R_correlation.png"), dpi=300)
plt.close(fig)

#Now do a pls fit if the ENSO indices onto the global theta_1000
println("Calculating PLS regression of global θ₁₀₀₀ onto ENSO indices...")
global_theta_1000 = era5_single_level_data["theta_1000"]
weighted_global_theta_1000 = global_theta_1000
weighted_global_theta_1000 = weighted_global_theta_1000[:, :, nonmissing_enso_idx]
original_shape = size(weighted_global_theta_1000)
pls_y_data = reshape(weighted_global_theta_1000, :, size(weighted_global_theta_1000, 3))'

#Now standardize the data
pls_y_data, y_means, y_stds = normalize_input!(copy(pls_y_data))
pls_enso_X, x_means, x_stds = normalize_input!(copy(enso_X), mean, nostd)

#Now weight the y data by sqrt(cos(lat))
cos_root_weights = sqrt.(cosd.(era5_lat'))
weighted_global_theta_1000 .*= cos_root_weights

#Calculate the PLS regressor
#global_theta_1000_pls = make_pls_regressor(pls_enso_X, pls_y_data, 1; print_updates=true, meanfunc = nomean, stdfunc = nostd)

#Write a function to calculate the r2 map
function calc_r_2(actual, pred)
    ss_total = sum((actual .- mean(actual)).^2)
    ss_residual = sum((actual .- pred).^2)
    r2 = 1 .- ss_residual ./ ss_total
    return r2
end

#predict pls_y using the model 
predicted_pls_y = pointwise_pls(pls_enso_X, pls_y_data; n_components = 3, slice_dims = (2,))
#Calculate the r2 map
r2_map = calc_r_2.(eachslice(pls_y_data; dims=2), eachslice(predicted_pls_y; dims=2))

thing_to_plot = reshape(r2_map, original_shape[1], original_shape[2])

# Plot the R² map
absmax = maximum(abs.(thing_to_plot))
fig_r2_map, ax = plt.subplots(figsize=(10, 6), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))
mappable = ax.contourf(era5_lon, era5_lat, Float64.(thing_to_plot'); levels=20,
                             transform=ccrs.PlateCarree(), 
                             cmap=cmr.prinsenvlag.reversed(), vmin = -absmax, vmax = absmax)
ax_r2_map = fig_r2_map.axes[0]
ax_r2_map.title.set_fontsize(16)
ax_r2_map.coastlines()
plt.colorbar(mappable, ax=ax_r2_map, orientation="horizontal", pad=0.05, shrink=0.8).set_label("Fraction of surface temperature variation explained by El Niño")

# Contour in the region mask
contour_r2 = ax_r2_map.contour(era5_lon, era5_lat, Float64.(sepac_mask'); 
                               levels=[0.5], colors="black", linewidths=1.5, 
                               transform=ccrs.PlateCarree())

# Save the R² map plot
fig_r2_map.savefig(joinpath(visdir, "global_theta_1000_enso_r2_map.png"), dpi=300)
plt.close(fig_r2_map)

println("R² map of global θ₁₀₀₀ PLS fit to ENSO indices saved to: $(joinpath(visdir, "global_theta_1000_enso_r2_map.png"))")

