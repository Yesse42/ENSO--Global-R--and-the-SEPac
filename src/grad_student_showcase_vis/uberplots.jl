"""
Investigate the utter bizarreness of the hemispheric correlation patterns
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")
include("../pls_regressor/pls_functions.jl")

using CSV, DataFrames, Plots

deseasonalize_and_detrend_precalculated_groups_twice!(slice, float_times, idx_groups; aggfunc = mean, trendfunc = least_squares_fit) = for _ in 1:3
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

#Add them to the data variables
era5_sl_var_names = ["LTS", "theta_1000", "theta_700", "t2m"]
era5_single_level_data = Dictionary(era5_sl_var_names, [LTS, theta_1000, theta_700, era5_data["t2m"]])

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

#Now make plot 1: Theta_1000 vs global R
corrs_theta_1000_r = cor.(Ref(ceres_global_rad), eachslice(era5_single_level_data["theta_1000"], dims=(1,2)))

fig = plot_global_heatmap(era5_lat, era5_lon, corrs_theta_1000_r; title = "Correlation between T_1000 and Global Net Radiation", colorbar_label = "Correlation Coefficient")
fig.savefig(joinpath(visdir, "theta_1000_vs_global_R_correlation.png"), dpi=300)
plt.close(fig)

#Aside: Calculate the correlation between ceres net rad in the sepac and global ceres net rad
sepac_net_rad = sepac_local_df[!, :toa_net_all_mon]
sepac_global_rad_corr = cor(ceres_global_rad, sepac_net_rad)
println("Correlation between CERES net radiation in the SEPac and global CERES net radiation: $sepac_global_rad_corr")

#Also calculate and print the std of ceres_global_rad for context
std_ceres_global_rad = std(ceres_global_rad)
println("Standard deviation of global CERES net radiation: $std_ceres_global_rad W/m²")

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
            out_arrs_plot4[i, j] .*= total_sepac_std/residual_sepac_std
        elseif j == 2
            out_arrs_plot4[i, j] .*= enso_component_std/residual_sepac_std
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
              title="ENSO Component (R² = $(round(r_squared, digits=3))", 
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
            out_arrs_plot6[i, j] .*= total_lts_std/residual_lts_std
        elseif j == 2
            out_arrs_plot6[i, j] .*= enso_lts_component_std/residual_lts_std
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
            out_arrs_plot7[i, j] .*= total_lts_std/residual_lts_std
        elseif j == 2
            out_arrs_plot7[i, j] .*= enso_lts_component_std/residual_lts_std
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