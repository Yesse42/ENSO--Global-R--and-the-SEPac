"""
GSS Plots
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")
include("../pls_regressor/pls_functions.jl")

using CSV, DataFrames, Plots, JLD2

deseasonalize_and_detrend_precalculated_groups_twice!(slice, float_times, idx_groups; aggfunc = mean, trendfunc = least_squares_fit) = for _ in 1:2
    deseasonalize_and_detrend_precalculated_groups!(slice, float_times, idx_groups; aggfunc, trendfunc)
    nothing
end

calc_theta(T, p) = T .* (1000 ./ p) .^ 0.286

time_period = (Date(2000, 3), Date(2024, 3, 31))
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/grad_student_showcase"
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
EIS = calculate_EIS.(era5_data["t"][:,:, p_1000_idx, :], era5_data["t"][:,:, p_700_idx, :]; RH_0 = 0.8)

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

#Aside: Calculate the correlation between ceres net rad in the sepac and global ceres net rad
sepac_net_rad = sepac_local_df[!, :toa_net_all_mon]
sepac_global_rad_corr = cor(ceres_global_rad, sepac_net_rad)
println("Correlation between CERES net radiation in the SEPac and global CERES net radiation: $sepac_global_rad_corr")

#Also calculate and print the std of ceres_global_rad for context
std_ceres_global_rad = std(ceres_global_rad)
println("Standard deviation of global CERES net radiation: $std_ceres_global_rad W/m²")

#Aside again: Calculate the corr between sepac LTS_1000 and sepac local net rad
sepac_lts = sepac_local_df[!, :LTS_1000]
lts_net_rad_corr = cor(sepac_lts, sepac_net_rad)
println("Correlation between SEPac LTS and SEPac net radiation: $lts_net_rad_corr")

#Corr between sepac LTS and global ceres net rad
lts_global_rad_corr = cor(sepac_lts, ceres_global_rad)
println("Correlation between SEPac LTS and global CERES net radiation: $lts_global_rad_corr")

#Aside again: Calculate the corr between sepac theta_1000 and sepac local net rad
sepac_theta_1000 = sepac_local_df[!, :θ_1000]
theta_1000_net_rad_corr = cor(sepac_theta_1000, sepac_net_rad)
println("Correlation between SEPac θ₁₀₀₀ and SEPac net radiation: $theta_1000_net_rad_corr")

#Aside again: Calculate the corr between sepac theta_700 and sepac local net rad
sepac_theta_700 = sepac_local_df[!, :θ_700]
theta_700_net_rad_corr = cor(sepac_theta_700, sepac_net_rad)
println("Correlation between SEPac θ₇₀₀ and SEPac net radiation: $theta_700_net_rad_corr")

#Aside again: calculate the corr between theta_1000 and global ceres net rad
theta_1000_global_rad_corr = cor(sepac_theta_1000, ceres_global_rad)
println("Correlation between SEPac θ₁₀₀₀ and global CERES net radiation: $theta_1000_global_rad_corr")

#Aside again: Use PLS to predict SEPac net rad from lagged ENSO
lags = -24:24
full_enso_df = DataFrame(collect(enso_data), string.(lags))
nonmissing_enso_times = [!any(ismissing, row) for row in eachrow(full_enso_df)]
nonmissing_common_times = common_times[nonmissing_enso_times]
pls_X = reduce(hcat, [enso_data["oni_lag_$lag"][nonmissing_enso_times] for lag in lags])
pls_Y = sepac_local_df[nonmissing_enso_times, :toa_net_all_mon]
n_components = 1
pls_model = make_pls_regressor(pls_X, pls_Y, n_components; print_updates=false)
#Get the scores
predicted_net_rad = vec(predict(pls_model, pls_X))
enso_sepac_net_rad_corr = cor(predicted_net_rad, pls_Y)
println("Correlation between PLS-predicted SEPac net radiation from ENSO and actual SEPac net radiation: $enso_sepac_net_rad_corr")

#Aside again:Use pls to predict global ceres net rad from lagged ENSO
pls_Y_global = ceres_global_rad[nonmissing_enso_times]
pls_model_global = make_pls_regressor(pls_X, pls_Y_global, n_components; print_updates=false)
#Get the scores
predicted_global_net_rad = vec(predict(pls_model_global, pls_X))
enso_global_net_rad_corr = cor(predicted_global_net_rad, pls_Y_global)
println("Correlation between PLS-predicted global net radiation from ENSO and actual global net radiation: $enso_global_net_rad_corr")

#Aside again: Calculate the corr between theta_700 and theta_1000
theta_700_theta_1000_corr = cor(sepac_theta_700, sepac_theta_1000)
println("Correlation between SEPac θ₇₀₀ and SEPac θ₁₀₀₀: $theta_700_theta_1000_corr")


#Now make plot 1: Theta_1000 vs global R
corrs_theta_1000_r = cor.(Ref(ceres_global_rad), eachslice(era5_single_level_data["theta_1000"], dims=(1,2)))

fig = plot_global_heatmap(era5_lat, era5_lon, corrs_theta_1000_r; title = "Correlation between T_1000 and Global Net Radiation", colorbar_label = "Correlation Coefficient")
ax = fig.axes[0]
#Contour in the region mask
contour = ax.contour(era5_lon, era5_lat, sepac_mask'; levels=[0.5], colors="black", linewidths=1.5, transform=ccrs.PlateCarree())
fig.savefig(joinpath(visdir, "theta_1000_vs_global_R_correlation.png"), dpi=300)
plt.close(fig)

#Now make plot 2: Decompose T_SEPac into an ENSO-related component and a non-ENSO-related component
#Calculate the lag with maximum correlation between T_1000 and ENSO
lags = -24:24
lagged_enso_dict = Dictionary(lags, [enso_data["oni_lag_$lag"] for lag in lags])
corrs = calculate_lag_correlations(sepac_local_df[!, :θ_1000], lagged_enso_dict; lags)
max_corr, max_lag = findmax(corrs)

max_enso_lag_data = enso_data["oni_lag_$max_lag"]
#Calculate the regression coefficient between T_1000 and max_enso_lag_data
sepac_onto_max_enso_lag_fit = least_squares_fit(max_enso_lag_data, sepac_local_df[!, :θ_1000])
sepac_enso_component = sepac_onto_max_enso_lag_fit.slope .* max_enso_lag_data .+ sepac_onto_max_enso_lag_fit.intercept
sepac_total_theta_1000 = sepac_local_df[!, :θ_1000]
sepac_non_enso_component = sepac_total_theta_1000 .- sepac_enso_component

# Create 3-panel plot of SEPac components
p1 = plot(common_times, sepac_total_theta_1000, title="Total θ₁₀₀₀", ylabel="θ₁₀₀₀ (K)", label="")
p2 = plot(common_times, sepac_enso_component, title="ENSO Component", ylabel="θ₁₀₀₀ (K)", label="")
p3 = plot(common_times, sepac_non_enso_component, title="Non-ENSO Component", ylabel="θ₁₀₀₀ (K)", label="")

# Get common y-limits
all_values = vcat(sepac_total_theta_1000, sepac_enso_component, sepac_non_enso_component)
ylims_range = (minimum(all_values), maximum(all_values))

# Apply same y-limits to all plots
plot!(p1, ylims=ylims_range)
plot!(p2, ylims=ylims_range)
plot!(p3, ylims=ylims_range)

# Combine into single figure
fig = plot(p1, p2, p3, layout=(3,1), size=(800, 600))
plot!(fig, suptitle="SEPac θ₁₀₀₀ Decomposition: Total, ENSO, and Non-ENSO Components")
savefig(fig, joinpath(visdir, "sepac_theta_1000_components.png"))


#Aside: calculate corr between enso residual t_1000 and global net rad
residual_theta_1000_global_rad_corr = cor(sepac_non_enso_component, ceres_global_rad)
println("Correlation between SEPac non-ENSO θ₁₀₀₀ and global CERES net radiation: $residual_theta_1000_global_rad_corr")

#Now analyze the pattern of correlations between SEPac T_1000, the enso component, and the non enso component on the gridded net, sw, and lw radiation.
#Ceres rad is the rows, SEPac theta components are the columns

out_arrs = Array{Matrix{Float64}}(undef, 3, 3)
ceres_rad_vars = ["toa_net_all_mon", "toa_net_sw_mon", "toa_net_lw_mon"]
sepac_theta_1000_components = ["Total θ₁₀₀₀", "ENSO Component", "Non-ENSO Component"]

ceres_gridded_rad_data = [ceres_gridded_rad[var] for var in ceres_rad_vars]
sepac_theta_1000_data = [sepac_total_theta_1000, sepac_enso_component, sepac_non_enso_component]

"Regress y onto standardized x"
function calc_1_sigma_regression(x, y)
    x_std = (x .- mean(x)) ./ std(x)
    fit = least_squares_fit(x_std, y)
    return fit.slope
end

residual_sepac_std = std(sepac_non_enso_component)
total_sepac_std = std(sepac_total_theta_1000)
enso_component_std = std(sepac_enso_component)

for (i, ceres_data) in enumerate(ceres_gridded_rad_data)
    for (j, sepac_data) in enumerate(sepac_theta_1000_data)
        out_arrs[i, j] = calc_1_sigma_regression.(eachslice(ceres_data, dims=(1,2)), Ref(sepac_data))

        #Scale by the appropriate factors so the total regression coeff plus the enso regression coeff equal the residual regression coeff
        if j == 1
            out_arrs[i, j] .*= total_sepac_std/residual_sepac_std
        elseif j == 2
            out_arrs[i, j] .*= enso_component_std/residual_sepac_std
        end
    end
end

# Create a 3x3 grid of subplots for Plot 3

fig3, axes3 = plt.subplots(3, 3; figsize=(15, 12), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)), layout = "compressed")

# Plot titles for rows and columns
rad_titles = ["Net Radiation", "SW Radiation", "LW Radiation"]
component_titles = ["Weighted SEPac θ₁₀₀₀", "Weighted ENSO Component", "ENSO Residual Regression Coefficient"]

# Calculate common color scale for Plot 3
max_abs_val_plot3 = max(maximum.([abs.(arr) for arr in out_arrs])...)
colornorm3 = colors.Normalize(vmin=-max_abs_val_plot3, vmax=max_abs_val_plot3)

for i in 1:3, j in 1:3
    ax = axes3[i-1, j-1]
    im = plot_global_heatmap_on_ax!(ax, ceres_lats, ceres_lons, out_arrs[i, j]; 
                                   title = "", colornorm=colornorm3)
    
    # Add titles
    if i == 1
        ax.set_title(component_titles[j])
    end
    if j == 1
        ax.set_yticks(Float64[])
        ax.set_ylabel(rad_titles[i])
    end
end

# Add single colorbar for Plot 3
cbar3 = fig3.colorbar(cm.ScalarMappable(norm=colornorm3, cmap=cmr.prinsenvlag.reversed()), ax=axes3, orientation="horizontal", pad=0.08, shrink=0.8)
cbar3.set_label("Regression Coefficient (W/m²/σ)", fontsize=12)

# Add suptitle for Plot 3
fig3.suptitle("Radiation Response to SEPac θ₁₀₀₀ Components", fontsize=16, y=0.98)

fig3.savefig(joinpath(visdir, "sepac_theta_1000_radiation_regression_patterns.png"), dpi=300)
plt.close(fig3)

# Now create Plot 4: Same style as Plot 3 but with different variables
# Create regression patterns for LTS, negative theta_1000, and theta_700 vs theta_1000 components

out_arrs_plot4 = Array{Matrix{Float64}}(undef, 3, 3)
era5_plot4_vars = ["LTS", "theta_1000", "theta_700"]
plot4_titles = "Gridded " .* ["LTS", "θ₁₀₀₀", "θ₇₀₀"]

era5_plot4_data = [era5_single_level_data["LTS"], era5_single_level_data["theta_1000"], era5_single_level_data["theta_700"]]

for (i, era5_data) in enumerate(era5_plot4_data)
    for (j, sepac_data) in enumerate(sepac_theta_1000_data)
        out_arrs_plot4[i, j] = calc_1_sigma_regression.(eachslice(era5_data, dims=(1,2)), Ref(sepac_data))

        #Scale by the appropriate factors so the total regression coeff plus the enso regression coeff equal the residual regression coeff
        if j == 1
            #out_arrs_plot4[i, j] .*= total_sepac_std/residual_sepac_std
        elseif j == 2
            #out_arrs_plot4[i, j] .*= enso_component_std/residual_sepac_std
        end
    end
end

# Create a 3x3 grid of subplots for Plot 4
fig4, axes4 = plt.subplots(3, 3; figsize=(15, 12), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)), layout = "compressed")

# Calculate common color scale for Plot 4
max_abs_val_plot4 = max(maximum.([abs.(arr) for arr in out_arrs_plot4])...)
colornorm4 = colors.Normalize(vmin=-max_abs_val_plot4, vmax=max_abs_val_plot4)

for i in 1:3, j in 1:3
    ax = axes4[i-1, j-1]
    im = plot_global_heatmap_on_ax!(ax, era5_lat, era5_lon, out_arrs_plot4[i, j]; 
                                   title = "", colornorm=colornorm4)
    
    # Add titles
    if i == 1
        ax.set_title(component_titles[j])
    end
    if j == 1
        ax.set_yticks(Float64[])
        ax.set_ylabel(plot4_titles[i])
    end
end

# Add single colorbar for Plot 4
cbar4 = fig4.colorbar(cm.ScalarMappable(norm=colornorm4, cmap=cmr.prinsenvlag.reversed()), ax=axes4, orientation="horizontal", pad=0.08, shrink=0.8)
cbar4.set_label("Regression Coefficient (K/σ)", fontsize=12)

# Add suptitle for Plot 4
fig4.suptitle("ERA5 Atmospheric Variables Response to SEPac θ₁₀₀₀ Components", fontsize=16, y=0.98)

fig4.savefig(joinpath(visdir, "sepac_theta_1000_era5_regoression_patterns.png"), dpi=300)
plt.close(fig4)

# Plot 5: Same as plot 2 but with LTS instead of theta_1000. Also use PLS instead of selecting the lag with max corr.

full_enso_df = DataFrame(collect(enso_data), string.(lags))
nonmissing_enso_times = [!any(ismissing, row) for row in eachrow(full_enso_df)]
nonmissing_common_times = common_times[nonmissing_enso_times]
pls_X = reduce(hcat, [lagged_enso_dict[lag][nonmissing_enso_times] for lag in lags])
pls_Y = sepac_local_df[nonmissing_enso_times, :LTS_1000]
n_components = 1
lts_pls_model = make_pls_regressor(pls_X, pls_Y, n_components; print_updates=false)
#Get the scores
pls_X_scores = lts_pls_model.X_scores[:, 1]

predicted_lts = vec(predict(lts_pls_model, pls_X))

residual_lts = pls_Y .- predicted_lts

# Plot 5a: PLS weights vs lag
pls_weights = lts_pls_model.X_weights[:, 1]
p_weights = plot(lags, pls_weights, 
                title="PLS Weights vs ENSO Lag", 
                xlabel="ENSO Lag (months)", 
                ylabel="PLS Weight",
                label="", 
                linewidth=2,
                marker=:circle,
                markersize=3)
savefig(p_weights, joinpath(visdir, "lts_pls_weights_vs_lag.png"))

# Plot 5b: Replicate figure 2 but with LTS data (3-panel time series)
sepac_total_lts = pls_Y
sepac_lts_enso_component = predicted_lts
sepac_lts_non_enso_component = residual_lts

# Calculate R² of the ENSO component
ss_total = sum((sepac_total_lts .- mean(sepac_total_lts)).^2)
ss_residual = sum(sepac_lts_non_enso_component.^2)
r_squared = 1 - (ss_residual / ss_total)

# Create 3-panel plot of SEPac LTS components
p1_lts = plot(nonmissing_common_times, sepac_total_lts, title="Total LTS", ylabel="LTS (K)", label="")
p2_lts = plot(nonmissing_common_times, sepac_lts_enso_component, 
              title="ENSO Component (R² = $(round(r_squared, digits=3)))", 
              ylabel="LTS (K)", label="", 
              linewidth=2, color=:blue)
plot!(p2_lts, nonmissing_common_times, sepac_total_lts, 
      label="", linewidth=1, alpha=0.5, color=:gray)
p3_lts = plot(nonmissing_common_times, sepac_lts_non_enso_component, title="Non-ENSO Component", ylabel="LTS (K)", label="")

# Get common y-limits for LTS plots
all_lts_values = vcat(sepac_total_lts, sepac_lts_enso_component, sepac_lts_non_enso_component)
ylims_lts_range = (minimum(all_lts_values), maximum(all_lts_values))

# Apply same y-limits to all LTS plots
plot!(p1_lts, ylims=ylims_lts_range)
plot!(p2_lts, ylims=ylims_lts_range)
plot!(p3_lts, ylims=ylims_lts_range)

# Combine into single figure
fig_lts = plot(p1_lts, p2_lts, p3_lts, layout=(3,1), size=(800, 600))
plot!(fig_lts, suptitle="SEPac LTS Decomposition: Total, ENSO (PLS), and Non-ENSO Components")
savefig(fig_lts, joinpath(visdir, "sepac_lts_components.png"))

# Plot 6: Same as Plot 3 but using LTS components from Plot 5 (restricted to reduced time range)
# Filter gridded radiation data to the reduced time range (nonmissing_common_times)
ceres_gridded_rad_reduced = Dict()
for var in keys(ceres_gridded_rad)
    ceres_gridded_rad_reduced[var] = ceres_gridded_rad[var][:, :, nonmissing_enso_times]
end

# Prepare data for Plot 6 analysis
out_arrs_plot6 = Array{Matrix{Float64}}(undef, 3, 3)
ceres_gridded_rad_data_reduced = [ceres_gridded_rad_reduced[var] for var in ceres_rad_vars]
sepac_lts_data = [sepac_total_lts, sepac_lts_enso_component, sepac_lts_non_enso_component]

# Calculate standard deviations for scaling
residual_lts_std = std(sepac_lts_non_enso_component)
total_lts_std = std(sepac_total_lts)
enso_lts_component_std = std(sepac_lts_enso_component)

# Calculate regression patterns for Plot 6
for (i, ceres_data) in enumerate(ceres_gridded_rad_data_reduced)
    for (j, lts_data) in enumerate(sepac_lts_data)
        out_arrs_plot6[i, j] = calc_1_sigma_regression.(eachslice(ceres_data, dims=(1,2)), Ref(lts_data))

        # Scale by the appropriate factors so the total regression coeff plus the enso regression coeff equal the residual regression coeff
        if j == 1
            #out_arrs_plot6[i, j] .*= total_lts_std/residual_lts_std
        elseif j == 2
            #out_arrs_plot6[i, j] .*= enso_lts_component_std/residual_lts_std
        end
    end
end

# Create a 3x3 grid of subplots for Plot 6
fig6, axes6 = plt.subplots(3, 3; figsize=(15, 12), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)), layout = "compressed")

# Plot titles for rows and columns
component_titles_lts = ["Weighted SEPac LTS", "Weighted ENSO Component", "ENSO Residual Regression Coefficient"]

# Calculate common color scale for Plot 6
max_abs_val_plot6 = max(maximum.([abs.(arr) for arr in out_arrs_plot6])...)
colornorm6 = colors.Normalize(vmin=-max_abs_val_plot6, vmax=max_abs_val_plot6)

for i in 1:3, j in 1:3
    ax = axes6[i-1, j-1]
    im = plot_global_heatmap_on_ax!(ax, ceres_lats, ceres_lons, out_arrs_plot6[i, j]; 
                                   title = "", colornorm=colornorm6)
    
    # Add titles
    if i == 1
        ax.set_title(component_titles_lts[j])
    end
    if j == 1
        ax.set_yticks(Float64[])
        ax.set_ylabel(rad_titles[i])
    end
end

# Add single colorbar for Plot 6
cbar6 = fig6.colorbar(cm.ScalarMappable(norm=colornorm6, cmap=cmr.prinsenvlag.reversed()), ax=axes6, orientation="horizontal", pad=0.08, shrink=0.8)
cbar6.set_label("Regression Coefficient (W/m²/σ)", fontsize=12)

# Add suptitle for Plot 6
fig6.suptitle("Radiation Response to SEPac LTS Components", fontsize=16, y=0.98)

fig6.savefig(joinpath(visdir, "sepac_lts_radiation_regression_patterns.png"), dpi=300)
plt.close(fig6)

# Plot 7: Same as Plot 4 but using LTS components from Plot 5 (restricted to reduced time range)
# Filter ERA5 single level data to the reduced time range (nonmissing_common_times)
era5_single_level_data_reduced = Dict()
for var in keys(era5_single_level_data)
    era5_single_level_data_reduced[var] = era5_single_level_data[var][:, :, nonmissing_enso_times]
end

# Prepare data for Plot 7 analysis
out_arrs_plot7 = Array{Matrix{Float64}}(undef, 3, 3)
era5_plot7_vars = ["LTS", "theta_1000", "theta_700"]
plot7_titles = "Gridded " .* ["LTS", "θ₁₀₀₀", "θ₇₀₀"]

era5_plot7_data = [era5_single_level_data_reduced["LTS"], era5_single_level_data_reduced["theta_1000"], era5_single_level_data_reduced["theta_700"]]

# Calculate regression patterns for Plot 7 (ERA5 variables vs LTS components)
for (i, era5_data) in enumerate(era5_plot7_data)
    for (j, lts_data) in enumerate(sepac_lts_data)
        out_arrs_plot7[i, j] = calc_1_sigma_regression.(eachslice(era5_data, dims=(1,2)), Ref(lts_data))

        # Scale by the appropriate factors so the total regression coeff plus the enso regression coeff equal the residual regression coeff
        if j == 1
            #out_arrs_plot7[i, j] .*= total_lts_std/residual_lts_std
        elseif j == 2
            #out_arrs_plot7[i, j] .*= enso_lts_component_std/residual_lts_std
        end
    end
end

# Create a 3x3 grid of subplots for Plot 7
fig7, axes7 = plt.subplots(3, 3; figsize=(15, 12), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)), layout = "compressed")

# Calculate common color scale for Plot 7
max_abs_val_plot7 = max(maximum.([abs.(arr) for arr in out_arrs_plot7])...)
colornorm7 = colors.Normalize(vmin=-max_abs_val_plot7, vmax=max_abs_val_plot7)

for i in 1:3, j in 1:3
    ax = axes7[i-1, j-1]
    im = plot_global_heatmap_on_ax!(ax, era5_lat, era5_lon, out_arrs_plot7[i, j]; 
                                   title = "", colornorm=colornorm7)
    
    # Add titles
    if i == 1
        ax.set_title(component_titles_lts[j])
    end
    if j == 1
        ax.set_yticks(Float64[])
        ax.set_ylabel(plot7_titles[i])
    end
end

# Add single colorbar for Plot 7
cbar7 = fig7.colorbar(cm.ScalarMappable(norm=colornorm7, cmap=cmr.prinsenvlag.reversed()), ax=axes7, orientation="horizontal", pad=0.08, shrink=0.8)
cbar7.set_label("Regression Coefficient (K/σ)", fontsize=12)

# Add suptitle for Plot 7
fig7.suptitle("ERA5 Atmospheric Variables Response to SEPac LTS Components", fontsize=16, y=0.98)

fig7.savefig(joinpath(visdir, "sepac_lts_era5_regression_patterns.png"), dpi=300)
plt.close(fig7)

# Plot 8: Same as Plot 3 structure but with SEPac theta_1000, SEPac theta_700, and SEPac LTS as columns
# Rows remain CERES gridded radiation (Net, SW, LW)

out_arrs_plot8 = Array{Matrix{Float64}}(undef, 3, 3)
sepac_theta_components_plot8 = [sepac_total_theta_1000, sepac_local_df[!, :θ_700], sepac_local_df[!, :LTS_1000]]

# Calculate regression patterns for Plot 8
for (i, ceres_data) in enumerate(ceres_gridded_rad_data)
    for (j, sepac_data) in enumerate(sepac_theta_components_plot8)
        out_arrs_plot8[i, j] = calc_1_sigma_regression.(eachslice(ceres_data, dims=(1,2)), Ref(sepac_data))
    end
end

# Create a 3x3 grid of subplots for Plot 8
fig8, axes8 = plt.subplots(3, 3; figsize=(15, 12), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)), layout = "compressed")

# Plot titles for rows and columns
component_titles_plot8 = ["SEPac θ₁₀₀₀", "SEPac θ₇₀₀", "SEPac LTS"]

# Calculate common color scale for Plot 8
max_abs_val_plot8 = max(maximum.([abs.(arr) for arr in out_arrs_plot8])...)
colornorm8 = colors.Normalize(vmin=-max_abs_val_plot8, vmax=max_abs_val_plot8)

for i in 1:3, j in 1:3
    ax = axes8[i-1, j-1]
    im = plot_global_heatmap_on_ax!(ax, ceres_lats, ceres_lons, out_arrs_plot8[i, j]; 
                                   title = "", colornorm=colornorm8)
    
    # Add titles
    if i == 1
        ax.set_title(component_titles_plot8[j])
    end
    if j == 1
        ax.set_yticks(Float64[])
        ax.set_ylabel(rad_titles[i])
    end
end

# Add single colorbar for Plot 8
cbar8 = fig8.colorbar(cm.ScalarMappable(norm=colornorm8, cmap=cmr.prinsenvlag.reversed()), ax=axes8, orientation="horizontal", pad=0.08, shrink=0.8)
cbar8.set_label("Regression Coefficient (W/m²/σ)", fontsize=12)

# Add suptitle for Plot 8
fig8.suptitle("Radiation Response to SEPac θ₁₀₀₀, θ₇₀₀, and LTS", fontsize=16, y=0.98)

fig8.savefig(joinpath(visdir, "sepac_theta_comparison_radiation_regression_patterns.png"), dpi=300)
plt.close(fig8)

# Plot 9: New 2x1 plot with theta variables and their PLS ENSO components
# First, calculate PLS ENSO components for theta_700 and theta_1000

# For theta_1000
pls_Y_theta_1000 = sepac_local_df[nonmissing_enso_times, :θ_1000]
theta_1000_pls_model = make_pls_regressor(pls_X, pls_Y_theta_1000, n_components; print_updates=false)
predicted_theta_1000_enso = vec(predict(theta_1000_pls_model, pls_X))

# For theta_700
pls_Y_theta_700 = sepac_local_df[nonmissing_enso_times, :θ_700]
theta_700_pls_model = make_pls_regressor(pls_X, pls_Y_theta_700, n_components; print_updates=false)
predicted_theta_700_enso = vec(predict(theta_700_pls_model, pls_X))

full_theta_corr = cor(pls_Y_theta_1000, pls_Y_theta_700)

# Create the 2x1 plot
fig9 = plot(layout=(2,1), size=(1000, 800))

# Top panel: Original theta_700 and theta_1000
plot!(fig9, nonmissing_common_times, pls_Y_theta_700, 
      subplot=1,
      label="θ₇₀₀", 
      linewidth=2, 
      color=:red,
      title="SEPac θ₇₀₀ and θ₁₀₀₀ Time Series, r=$(round(full_theta_corr, digits=3))",
      ylabel="Temperature (K)",
      legend=:topright)

plot!(fig9, nonmissing_common_times, pls_Y_theta_1000, 
      subplot=1,
      label="θ₁₀₀₀", 
      linewidth=2, 
      color=:blue)

# Bottom panel: Faint original theta variables plus their PLS ENSO components
# (The title is set in the correlation calculation section above)

plot!(fig9, nonmissing_common_times, pls_Y_theta_1000, 
      subplot=2,
      label="θ₁₀₀₀ (original)", 
      linewidth=1, 
      color=:blue,
      alpha=0.3)

plot!(fig9, nonmissing_common_times, predicted_theta_700_enso, 
      subplot=2,
      label="θ₇₀₀ ENSO component", 
      linewidth=2, 
      color=:darkred,
      linestyle=:solid)

plot!(fig9, nonmissing_common_times, predicted_theta_1000_enso, 
      subplot=2,
      label="θ₁₀₀₀ ENSO component", 
      linewidth=2, 
      color=:darkblue,
      linestyle=:solid)

# Calculate correlations for display
theta_700_enso_corr = cor(predicted_theta_700_enso, pls_Y_theta_700)
theta_1000_enso_corr = cor(predicted_theta_1000_enso, pls_Y_theta_1000)

# Calculate additional correlations for the lower panel
# Non-ENSO components (residuals)
theta_700_non_enso = pls_Y_theta_700 - predicted_theta_700_enso
theta_1000_non_enso = pls_Y_theta_1000 - predicted_theta_1000_enso

# Correlations between components
enso_components_corr = cor(predicted_theta_700_enso, predicted_theta_1000_enso)
non_enso_components_corr = cor(theta_700_non_enso, theta_1000_non_enso)
full_theta_corr = cor(pls_Y_theta_700, pls_Y_theta_1000)

# Update the title for the lower panel to include additional correlations
lower_panel_title = "PLS ENSO Components: ENSO r=$(round(enso_components_corr, digits=3)), Non-ENSO r=$(round(non_enso_components_corr, digits=3))"

# Modify the bottom panel plot to use the new title
plot!(fig9, nonmissing_common_times, pls_Y_theta_700, 
      subplot=2,
      label="θ₇₀₀ (original)", 
      linewidth=1, 
      color=:red,
      alpha=0.3,
      title=lower_panel_title,
      ylabel="Temperature (K)",
      xlabel="Date",
      legend=:topright)

# Add correlation info as subtitle
plot!(fig9, suptitle="θ₇₀₀ ENSO r=$(round(theta_700_enso_corr, digits=3)), θ₁₀₀₀ ENSO r=$(round(theta_1000_enso_corr, digits=3))")

# Save the plot
savefig(fig9, joinpath(visdir, "sepac_theta_enso_components_2panel.png"))

# Print correlations
println("θ₇₀₀ ~ ENSO PLS correlation: $(round(theta_700_enso_corr, digits=4))")
println("θ₁₀₀₀ ~ ENSO PLS correlation: $(round(theta_1000_enso_corr, digits=4))")
println("ENSO components correlation (θ₇₀₀ vs θ₁₀₀₀): $(round(enso_components_corr, digits=4))")
println("Non-ENSO components correlation (θ₇₀₀ vs θ₁₀₀₀): $(round(non_enso_components_corr, digits=4))")
println("Full theta correlation (θ₇₀₀ vs θ₁₀₀₀): $(round(full_theta_corr, digits=4))")

# Plot 10: Pointwise correlations between CERES net radiation and LTS/EIS
println("Calculating pointwise correlations between CERES net radiation and LTS/EIS...")

# Load the coordinate mapping indices
coord_mapping_file = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/stratocum_comparison/era5_ceres_coordinate_mapping.jld2"
coord_mapping = JLD2.load(coord_mapping_file)
era5_to_ceres_indices = Tuple.(coord_mapping["era5_to_ceres_indices"])
ceres_to_era5_indices = coord_mapping["ceres_to_era5_indices"]

# Get CERES net radiation data (use the full time series, not the reduced one)
ceres_net_rad = ceres_gridded_rad["toa_net_all_mon"]

# Get LTS and EIS data from ERA5 (use the full time series)
era5_lts = era5_single_level_data["LTS"]
era5_eis = era5_single_level_data["EIS"]  # This was calculated earlier in the script

time_idxs = collect(1:size(ceres_net_rad, 3))

eis_ceres_grid = [era5_eis[i_ceres, j_ceres, time_idx] for (i_ceres, j_ceres) in era5_to_ceres_indices, time_idx in time_idxs]
lts_ceres_grid = [era5_lts[i_ceres, j_ceres, time_idx] for (i_ceres, j_ceres) in era5_to_ceres_indices, time_idx in time_idxs]

# Initialize correlation arrays for CERES grid
ceres_lts_corr = cor.(eachslice(ceres_net_rad, dims=(1,2)), eachslice(lts_ceres_grid, dims=(1,2)))
ceres_eis_corr = cor.(eachslice(ceres_net_rad, dims=(1,2)), eachslice(eis_ceres_grid, dims=(1,2)))

# Create a 1x2 plot showing both correlations
fig10, axes10 = plt.subplots(1, 2; figsize=(16, 6), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))

# Calculate common color scale for both plots
max_abs_corr = max(maximum(abs.(filter(!isnan, ceres_lts_corr))), maximum(abs.(filter(!isnan, ceres_eis_corr))))
colornorm_corr = colors.Normalize(vmin=-max_abs_corr, vmax=max_abs_corr)

# Plot LTS correlations
ax1 = axes10[0]
im1 = plot_global_heatmap_on_ax!(ax1, ceres_lats, ceres_lons, ceres_lts_corr; 
                                title="CERES Net Radiation vs LTS Correlation", 
                                colornorm=colornorm_corr)

# Plot EIS correlations  
ax2 = axes10[1]
im2 = plot_global_heatmap_on_ax!(ax2, ceres_lats, ceres_lons, ceres_eis_corr; 
                                title="CERES Net Radiation vs EIS Correlation", 
                                colornorm=colornorm_corr)

# Add single colorbar
cbar10 = fig10.colorbar(cm.ScalarMappable(norm=colornorm_corr, cmap=cmr.prinsenvlag.reversed()), 
                        ax=axes10, orientation="horizontal", pad=0.08, shrink=0.8)
cbar10.set_label("Correlation Coefficient", fontsize=12)

# Add overall title
fig10.suptitle("Pointwise Correlations: CERES Net Radiation vs Atmospheric Stability", fontsize=16, y=0.95)

# Save the plot
fig10.savefig(joinpath(visdir, "ceres_radiation_vs_lts_eis_correlations.png"), dpi=300, bbox_inches="tight")
plt.close(fig10)

println("Saved pointwise correlation plots: ceres_radiation_vs_lts_eis_correlations.png")

# Print some summary statistics
valid_lts_corrs = filter(!isnan, ceres_lts_corr)
valid_eis_corrs = filter(!isnan, ceres_eis_corr)

println("LTS correlation statistics:")
println("  Mean: $(round(mean(valid_lts_corrs), digits=4))")
println("  Std:  $(round(std(valid_lts_corrs), digits=4))")
println("  Min:  $(round(minimum(valid_lts_corrs), digits=4))")
println("  Max:  $(round(maximum(valid_lts_corrs), digits=4))")

println("EIS correlation statistics:")
println("  Mean: $(round(mean(valid_eis_corrs), digits=4))")
println("  Std:  $(round(std(valid_eis_corrs), digits=4))")
println("  Min:  $(round(minimum(valid_eis_corrs), digits=4))")
println("  Max:  $(round(maximum(valid_eis_corrs), digits=4))")

# Plot 11: Companion plot showing gridwise standard deviation of LTS and EIS
println("Calculating gridwise standard deviations of LTS and EIS...")

# Calculate standard deviations for each grid point
lts_std = std.(eachslice(lts_ceres_grid, dims=(1,2)))
eis_std = std.(eachslice(eis_ceres_grid, dims=(1,2)))

# Create a 1x2 plot showing both standard deviations
fig11, axes11 = plt.subplots(1, 2; figsize=(16, 6), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))

# Calculate separate color scales for each plot since they have different units/ranges
lts_std_max = maximum(filter(!isnan, lts_std))
eis_std_max = maximum(filter(!isnan, eis_std))
colornorm_lts_std = colors.Normalize(vmin=0, vmax=lts_std_max)
colornorm_eis_std = colors.Normalize(vmin=0, vmax=eis_std_max)

# Plot LTS standard deviation
ax1_std = axes11[0]
im1_std = plot_global_heatmap_on_ax!(ax1_std, ceres_lats, ceres_lons, lts_std; 
                                    title="LTS Standard Deviation", 
                                    colornorm=colornorm_lts_std,
                                    cmap="viridis")

# Plot EIS standard deviation
ax2_std = axes11[1]
im2_std = plot_global_heatmap_on_ax!(ax2_std, ceres_lats, ceres_lons, eis_std; 
                                    title="EIS Standard Deviation", 
                                    colornorm=colornorm_eis_std,
                                    cmap="viridis")

# Add separate colorbars for each subplot
cbar11_lts = fig11.colorbar(cm.ScalarMappable(norm=colornorm_lts_std, cmap="viridis"), 
                           ax=ax1_std, orientation="horizontal", pad=0.08, shrink=0.8)
cbar11_lts.set_label("LTS Standard Deviation (K)", fontsize=12)

cbar11_eis = fig11.colorbar(cm.ScalarMappable(norm=colornorm_eis_std, cmap="viridis"), 
                           ax=ax2_std, orientation="horizontal", pad=0.08, shrink=0.8)
cbar11_eis.set_label("EIS Standard Deviation (K)", fontsize=12)

# Add overall title
fig11.suptitle("Gridwise Standard Deviations: Atmospheric Stability Measures", fontsize=16, y=0.95)

# Save the plot
fig11.savefig(joinpath(visdir, "lts_eis_standard_deviations.png"), dpi=300, bbox_inches="tight")
plt.close(fig11)

println("Saved standard deviation plots: lts_eis_standard_deviations.png")

# Print summary statistics for standard deviations
valid_lts_std = filter(!isnan, lts_std)
valid_eis_std = filter(!isnan, eis_std)

println("LTS standard deviation statistics:")
println("  Mean: $(round(mean(valid_lts_std), digits=4)) K")
println("  Std:  $(round(std(valid_lts_std), digits=4)) K")
println("  Min:  $(round(minimum(valid_lts_std), digits=4)) K")
println("  Max:  $(round(maximum(valid_lts_std), digits=4)) K")

println("EIS standard deviation statistics:")
println("  Mean: $(round(mean(valid_eis_std), digits=4)) K")
println("  Std:  $(round(std(valid_eis_std), digits=4)) K")
println("  Min:  $(round(minimum(valid_eis_std), digits=4)) K")
println("  Max:  $(round(maximum(valid_eis_std), digits=4)) K")

# Plot 12: Regression coefficients of net radiation onto LTS/EIS multiplied by their standard deviations
println("Calculating regression coefficients of CERES net radiation regressed onto LTS/EIS...")

# Calculate regression coefficients for each grid point
lts_regression_coeff = calc_1_sigma_regression.(eachslice(lts_ceres_grid, dims=(1,2)), eachslice(ceres_net_rad, dims=(1,2)))
eis_regression_coeff = calc_1_sigma_regression.(eachslice(eis_ceres_grid, dims=(1,2)), eachslice(ceres_net_rad, dims=(1,2)))

# Multiply by standard deviations to get expected change in net radiation for 1-std change in LTS/EIS
lts_sensitivity = lts_regression_coeff .* lts_std
eis_sensitivity = eis_regression_coeff .* eis_std

# Create a 1x2 plot showing both sensitivity patterns
fig12, axes12 = plt.subplots(1, 2; figsize=(16, 6), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))

# Calculate common color scale for both plots to enable comparison
max_abs_sensitivity = max(maximum(abs.(filter(!isnan, lts_sensitivity))), maximum(abs.(filter(!isnan, eis_sensitivity))))
colornorm_sensitivity = colors.Normalize(vmin=-max_abs_sensitivity, vmax=max_abs_sensitivity)

# Plot LTS sensitivity (regression coeff * std)
ax1_sens = axes12[0]
im1_sens = plot_global_heatmap_on_ax!(ax1_sens, ceres_lats, ceres_lons, lts_sensitivity; 
                                     title="Net Radiation Sensitivity to LTS", 
                                     colornorm=colornorm_sensitivity)

# Plot EIS sensitivity (regression coeff * std)
ax2_sens = axes12[1]
im2_sens = plot_global_heatmap_on_ax!(ax2_sens, ceres_lats, ceres_lons, eis_sensitivity; 
                                     title="Net Radiation Sensitivity to EIS", 
                                     colornorm=colornorm_sensitivity)

# Add single colorbar
cbar12 = fig12.colorbar(cm.ScalarMappable(norm=colornorm_sensitivity, cmap=cmr.prinsenvlag.reversed()), 
                        ax=axes12, orientation="horizontal", pad=0.08, shrink=0.8)
cbar12.set_label("Net Radiation Change per 1σ Stability Change (W/m²)", fontsize=12)

# Add overall title
fig12.suptitle("Net Radiation onto EIS/LTS (gridwise), W/m^2/σ", fontsize=16, y=0.95)

# Save the plot
fig12.savefig(joinpath(visdir, "net_radiation_stability_sensitivity.png"), dpi=300, bbox_inches="tight")
plt.close(fig12)

println("Saved sensitivity plots: net_radiation_stability_sensitivity.png")

# Print summary statistics for sensitivity
valid_lts_sensitivity = filter(!isnan, lts_sensitivity)
valid_eis_sensitivity = filter(!isnan, eis_sensitivity)

println("LTS sensitivity statistics (W/m² per 1σ LTS change):")
println("  Mean: $(round(mean(valid_lts_sensitivity), digits=4)) W/m²")
println("  Std:  $(round(std(valid_lts_sensitivity), digits=4)) W/m²")
println("  Min:  $(round(minimum(valid_lts_sensitivity), digits=4)) W/m²")
println("  Max:  $(round(maximum(valid_lts_sensitivity), digits=4)) W/m²")

println("EIS sensitivity statistics (W/m² per 1σ EIS change):")
println("  Mean: $(round(mean(valid_eis_sensitivity), digits=4)) W/m²")
println("  Std:  $(round(std(valid_eis_sensitivity), digits=4)) W/m²")
println("  Min:  $(round(minimum(valid_eis_sensitivity), digits=4)) W/m²")
println("  Max:  $(round(maximum(valid_eis_sensitivity), digits=4)) W/m²")

# Plot 13: Correlation between global net radiation and gridwise LTS/EIS
println("Calculating correlations between global net radiation and gridwise LTS/EIS...")

# Calculate correlations between global radiation and each grid point's LTS/EIS
global_lts_corr = cor.(eachslice(lts_ceres_grid, dims=(1,2)), Ref(ceres_global_rad))
global_eis_corr = cor.(eachslice(eis_ceres_grid, dims=(1,2)), Ref(ceres_global_rad))

# Create a 1x2 plot showing both correlations
fig13, axes13 = plt.subplots(1, 2; figsize=(16, 6), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))

# Calculate common color scale for both plots
max_abs_global_corr = max(maximum(abs.(filter(!isnan, global_lts_corr))), maximum(abs.(filter(!isnan, global_eis_corr))))
colornorm_global_corr = colors.Normalize(vmin=-max_abs_global_corr, vmax=max_abs_global_corr)

# Plot LTS vs global radiation correlations
ax1_global = axes13[0]
im1_global = plot_global_heatmap_on_ax!(ax1_global, ceres_lats, ceres_lons, global_lts_corr; 
                                       title="Local LTS vs Global Net Radiation Correlation", 
                                       colornorm=colornorm_global_corr)

# Plot EIS vs global radiation correlations
ax2_global = axes13[1]
im2_global = plot_global_heatmap_on_ax!(ax2_global, ceres_lats, ceres_lons, global_eis_corr; 
                                       title="Local EIS vs Global Net Radiation Correlation", 
                                       colornorm=colornorm_global_corr)

# Add single colorbar
cbar13 = fig13.colorbar(cm.ScalarMappable(norm=colornorm_global_corr, cmap=cmr.prinsenvlag.reversed()), 
                        ax=axes13, orientation="horizontal", pad=0.08, shrink=0.8)
cbar13.set_label("Correlation Coefficient", fontsize=12)

# Add overall title
fig13.suptitle("Local Atmospheric Stability vs Global Net Radiation Correlations", fontsize=16, y=0.95)

# Save the plot
fig13.savefig(joinpath(visdir, "local_stability_vs_global_radiation_correlations.png"), dpi=300, bbox_inches="tight")
plt.close(fig13)

println("Saved local-global correlation plots: local_stability_vs_global_radiation_correlations.png")

# Plot 14: Regression coefficients and sensitivity of global radiation to gridwise LTS/EIS
println("Calculating regression of global net radiation onto gridwise LTS/EIS...")

# Calculate regression coefficients of global radiation onto each grid point's LTS/EIS
global_lts_regression = calc_1_sigma_regression.(eachslice(lts_ceres_grid, dims=(1,2)), Ref(ceres_global_rad))
global_eis_regression = calc_1_sigma_regression.(eachslice(eis_ceres_grid, dims=(1,2)), Ref(ceres_global_rad))

# Multiply by standard deviations to get sensitivity
global_lts_sensitivity = global_lts_regression .* lts_std
global_eis_sensitivity = global_eis_regression .* eis_std

# Create a 1x2 plot showing both sensitivities
fig14, axes14 = plt.subplots(1, 2; figsize=(16, 6), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)))

# Calculate common color scale for both plots
max_abs_global_sensitivity = max(maximum(abs.(filter(!isnan, global_lts_sensitivity))), maximum(abs.(filter(!isnan, global_eis_sensitivity))))
colornorm_global_sensitivity = colors.Normalize(vmin=-max_abs_global_sensitivity, vmax=max_abs_global_sensitivity)

# Plot LTS sensitivity to global radiation
ax1_global_sens = axes14[0]
im1_global_sens = plot_global_heatmap_on_ax!(ax1_global_sens, ceres_lats, ceres_lons, global_lts_sensitivity; 
                                            title="Global Radiation Sensitivity to Local LTS", 
                                            colornorm=colornorm_global_sensitivity)

# Plot EIS sensitivity to global radiation
ax2_global_sens = axes14[1]
im2_global_sens = plot_global_heatmap_on_ax!(ax2_global_sens, ceres_lats, ceres_lons, global_eis_sensitivity; 
                                            title="Global Radiation Sensitivity to Local EIS", 
                                            colornorm=colornorm_global_sensitivity)

# Add single colorbar
cbar14 = fig14.colorbar(cm.ScalarMappable(norm=colornorm_global_sensitivity, cmap=cmr.prinsenvlag.reversed()), 
                        ax=axes14, orientation="horizontal", pad=0.08, shrink=0.8)
cbar14.set_label("Global Radiation Change per 1σ Local Stability Change (W/m²)", fontsize=12)

# Add overall title
fig14.suptitle("Global Net Radiation Sensitivity to Local Atmospheric Stability (β × σ)", fontsize=16, y=0.95)

# Save the plot
fig14.savefig(joinpath(visdir, "global_radiation_local_stability_sensitivity.png"), dpi=300, bbox_inches="tight")
plt.close(fig14)

println("Saved global-local sensitivity plots: global_radiation_local_stability_sensitivity.png")

# Print summary statistics
valid_global_lts_corr = filter(!isnan, global_lts_corr)
valid_global_eis_corr = filter(!isnan, global_eis_corr)
valid_global_lts_sensitivity = filter(!isnan, global_lts_sensitivity)
valid_global_eis_sensitivity = filter(!isnan, global_eis_sensitivity)

println("Local LTS vs Global Radiation correlation statistics:")
println("  Mean: $(round(mean(valid_global_lts_corr), digits=4))")
println("  Std:  $(round(std(valid_global_lts_corr), digits=4))")
println("  Min:  $(round(minimum(valid_global_lts_corr), digits=4))")
println("  Max:  $(round(maximum(valid_global_lts_corr), digits=4))")

println("Local EIS vs Global Radiation correlation statistics:")
println("  Mean: $(round(mean(valid_global_eis_corr), digits=4))")
println("  Std:  $(round(std(valid_global_eis_corr), digits=4))")
println("  Min:  $(round(minimum(valid_global_eis_corr), digits=4))")
println("  Max:  $(round(maximum(valid_global_eis_corr), digits=4))")

println("Global Radiation sensitivity to local LTS (W/m² per 1σ local LTS change):")
println("  Mean: $(round(mean(valid_global_lts_sensitivity), digits=4)) W/m²")
println("  Std:  $(round(std(valid_global_lts_sensitivity), digits=4)) W/m²")
println("  Min:  $(round(minimum(valid_global_lts_sensitivity), digits=4)) W/m²")
println("  Max:  $(round(maximum(valid_global_lts_sensitivity), digits=4)) W/m²")

println("Global Radiation sensitivity to local EIS (W/m² per 1σ local EIS change):")
println("  Mean: $(round(mean(valid_global_eis_sensitivity), digits=4)) W/m²")
println("  Std:  $(round(std(valid_global_eis_sensitivity), digits=4)) W/m²")
println("  Min:  $(round(minimum(valid_global_eis_sensitivity), digits=4)) W/m²")
println("  Max:  $(round(maximum(valid_global_eis_sensitivity), digits=4)) W/m²")

# Plot 15: 3x3 correlation plot showing radiation response to SEPac theta_1000 components
# Rows: Net, SW, LW radiation
# Columns: Full theta_1000, ENSO component, ENSO residual

out_arrs_plot15 = Array{Matrix{Float64}}(undef, 3, 3)

std_total_theta_1000 = std(sepac_total_theta_1000)
std_theta_1000_enso = std(sepac_enso_component)
std_theta_1000_non_enso = std(sepac_non_enso_component)

std_net_rad_gridwise = std(ceres_gridded_rad_data[1]; dims = 3)[:, :, 1]
std_sw_rad_gridwise = std(ceres_gridded_rad_data[2]; dims = 3)[:, :, 1]
std_lw_rad_gridwise = std(ceres_gridded_rad_data[3]; dims = 3)[:,:,1]

# Calculate correlations for Plot 15
for (i, ceres_data) in enumerate(ceres_gridded_rad_data)
    for (j, sepac_data) in enumerate(sepac_theta_1000_data)
        temp_arr = cor.(eachslice(ceres_data, dims=(1,2)), Ref(sepac_data))

        if j == 1
            temp_arr .*= std_theta_1000_enso ./ std_theta_1000_non_enso
        elseif j == 2
            temp_arr .*= std_theta_1000_enso ./ std_theta_1000_non_enso
        end

        if i == 2
            temp_arr .*= std_sw_rad_gridwise ./ std_net_rad_gridwise
        elseif i == 3
            temp_arr .*= std_lw_rad_gridwise ./ std_net_rad_gridwise
        end

        out_arrs_plot15[i, j] = temp_arr
    end
end



# Create a 3x3 grid of subplots for Plot 15
fig15, axes15 = plt.subplots(3, 3; figsize=(15, 12), subplot_kw=Dict("projection" => ccrs.Robinson(central_longitude=180)), layout = "compressed")

# Calculate separate color scales for first two columns vs last column
max_abs_val_plot15_cols12 = max(maximum.([(arr->abs.(arr)).(out_arrs_plot15[i, j]) for i in 1:3, j in 1:2])...)
max_abs_val_plot15_col3 = max(maximum.((arr->abs.(arr)).(out_arrs_plot15[:, 3]))...)
colornorm15_cols12 = colors.Normalize(vmin=-max_abs_val_plot15_cols12, vmax=max_abs_val_plot15_cols12)
colornorm15_col3 = colors.Normalize(vmin=-max_abs_val_plot15_col3, vmax=max_abs_val_plot15_col3)

for i in 1:3, j in 1:3
    ax = axes15[i-1, j-1]
    colornorm = j <= 2 ? colornorm15_cols12 : colornorm15_col3
    im = plot_global_heatmap_on_ax!(ax, ceres_lats, ceres_lons, out_arrs_plot15[i, j]; 
                                   title = "", colornorm=colornorm)
    
    # Add titles
    if i == 1
        ax.set_title(sepac_theta_1000_components[j])
    end
    if j == 1
        ax.set_yticks(Float64[])
        ax.set_ylabel(rad_titles[i])
    end
end

# Add separate colorbars - collect axes for first two columns
axes_cols12 = [axes15[i, j] for i in 0:2, j in 0:1]
axes_col3 = [axes15[i, 2] for i in 0:2]

cbar15_cols12 = fig15.colorbar(cm.ScalarMappable(norm=colornorm15_cols12, cmap=cmr.prinsenvlag.reversed()), ax=pylist(vec(axes_cols12)), orientation="horizontal", pad=0.08, shrink=0.8)
cbar15_cols12.set_label("Weighted Correlation Coefficient", fontsize=12)

cbar15_col3 = fig15.colorbar(cm.ScalarMappable(norm=colornorm15_col3, cmap=cmr.prinsenvlag.reversed()), ax=pylist(axes_col3), orientation="horizontal", pad=0.08, shrink=0.8)
cbar15_col3.set_label("Correlation Coefficient", fontsize=12)

# Add suptitle for Plot 15
fig15.suptitle("Radiation Correlations with SEPac θ₁₀₀₀ Components", fontsize=16, y=0.98)

fig15.savefig(joinpath(visdir, "sepac_theta_1000_radiation_correlation_patterns.png"), dpi=300)
plt.close(fig15)
