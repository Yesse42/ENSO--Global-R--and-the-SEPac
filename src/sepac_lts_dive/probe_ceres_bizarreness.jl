"""
Investigate the utter bizarreness of the hemispheric correlation patterns
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

time_period = (Date(2000, 3), Date(2024, 3, 31))
visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/why_ceres_why"

ceres_global_rad, grad_times = load_new_ceres_data(["gtoa_net_all_mon"], time_period)
ceres_global_rad = ceres_global_rad["gtoa_net_all_mon"]
grad_times = grad_times["time"]

ceres_gridded_rad, gridded_rad_coords = load_new_ceres_data(["toa_net_all_mon"], time_period)
ceres_gridded_rad = ceres_gridded_rad["toa_net_all_mon"]
gridded_rad_times = gridded_rad_coords["time"]
lats = gridded_rad_coords["latitude"]
lons = gridded_rad_coords["longitude"]

#Ensure both datasets have the same time points
common_times = intersect(grad_times, gridded_rad_times)
ceres_global_rad = ceres_global_rad[in.(grad_times, Ref(common_times))]
copy_ceres_global_rad = copy(ceres_global_rad)
ceres_gridded_rad = ceres_gridded_rad[:, :, in.(gridded_rad_times, Ref(common_times))]
common_times = sort!(common_times)

point_to_extract = (-175 + 360, -50) # (lon, lat)
savedir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES"

#Now find the nearest lon lat point in the gridded data
lon_idx = findmin(abs.(lons .- point_to_extract[1]))[2]
lat_idx = findmin(abs.(lats .- point_to_extract[2]))[2]

nearest_lon_lat = (lons[lon_idx], lats[lat_idx])
println("Nearest grid point to $point_to_extract is $nearest_lon_lat at indices (lon: $lon_idx, lat: $lat_idx)")

#Extract the time series at that point
gridded_point_series = ceres_gridded_rad[lon_idx, lat_idx, :]
#Save the time series to a CSV for external analysis
using CSV, DataFrames
df = DataFrame(Date=common_times, Gridded_Radiation=gridded_point_series)
CSV.write(joinpath(savedir, "new_zealand_point_series.csv"), df)

float_times = calc_float_time.(common_times)
month_groups = SplitApplyCombine.groupfind(month, common_times)
months = month.(common_times)

no_trend_func(x, times) = (;slope = 0.0, intercept = 0.0)

trend_funcs = (no_trend_func, least_squares_fit)

raw_global_data = [copy(ceres_global_rad) for trend_func in trend_funcs]
raw_gridded_data = [copy(ceres_gridded_rad) for trend_func in trend_funcs]

agg_func = mean

corr_mats = Matrix{Float64}[]

for (global_data, gridded_data, trend_func) in zip(raw_global_data, raw_gridded_data, trend_funcs)

    detrend_and_deseasonalize!(global_data, float_times, months; aggfunc=agg_func, trendfunc=trend_func)

    detrend_and_deseasonalize_precalculated_groups!.(eachslice(gridded_data; dims=(1,2)), Ref(float_times), Ref(month_groups); aggfunc=agg_func, trendfunc=trend_func)

    #Now calculate the corr matrix
    corr_mat = cor.(eachslice(gridded_data; dims=(1,2)), Ref(global_data))[:,:,1]

    push!(corr_mats, corr_mat)
end

no_trend_data, theil_sen_data = corr_mats
diff_mat = theil_sen_data .- no_trend_data

# Plot the three matrices separately
println("Plotting correlation matrices...")

# Plot 1: No trend removal correlation matrix
fig1 = plot_global_heatmap(lats, lons, corr_mats[1]; 
    title="Global-Gridded Radiation Correlation (No Detrending)", 
    colorbar_label="Correlation Coefficient")
fig1.savefig(joinpath(visdir, "no_trend_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig1)

# Plot 2: Theil-Sen detrended correlation matrix
fig2 = plot_global_heatmap(lats, lons, corr_mats[2]; 
    title="Global-Gridded Radiation Correlation (Least Squares Detrended)", 
    colorbar_label="Correlation Coefficient")
fig2.savefig(joinpath(visdir, "lstsq_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig2)

# Plot 3: Difference between the two correlation matrices
fig3 = plot_global_heatmap(lats, lons, diff_mat; 
    title="Difference in Correlation (Least Squares - No Trend)", 
    colorbar_label="Correlation Difference")
fig3.savefig(joinpath(visdir, "correlation_difference_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig3)

println("Plots saved to: $visdir")

# Additional analysis: Apply detrend and deseasonalize 4 times iteratively
println("\nPerforming 4-fold detrending and deseasonalizing analysis...")

# Create fresh copies for the 4-fold analysis
fourfold_global_data = copy(copy_ceres_global_rad)
fourfold_gridded_data = copy(ceres_gridded_rad)

# Apply detrend_and_deseasonalize 4 times iteratively
for iteration in 1:4
    println("Iteration $iteration of detrending and deseasonalizing...")
    
    # Detrend and deseasonalize global data
    detrend_and_deseasonalize!(fourfold_global_data, float_times, months; 
                              aggfunc=agg_func, trendfunc=least_squares_fit)
    
    # Detrend and deseasonalize gridded data
    detrend_and_deseasonalize_precalculated_groups!.(eachslice(fourfold_gridded_data; dims=(1,2)), 
                                                    Ref(float_times), Ref(month_groups); 
                                                    aggfunc=agg_func, trendfunc=least_squares_fit)
end

# Calculate correlations after 4-fold processing
fourfold_corr_mat = cor.(eachslice(fourfold_gridded_data; dims=(1,2)), Ref(fourfold_global_data))[:,:,1]

# Calculate difference between 4-fold processed and no detrending
fourfold_diff_mat = fourfold_corr_mat .- corr_mats[1]

# Plot the 4-fold processed correlation matrix
println("Plotting 4-fold detrended and deseasonalized correlation matrix...")
fig4 = plot_global_heatmap(lats, lons, fourfold_corr_mat; 
    title="Global-Gridded Radiation Correlation (4x Detrended & Deseasonalized)", 
    colorbar_label="Correlation Coefficient")
fig4.savefig(joinpath(visdir, "fourfold_detrend_deseason_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig4)

# Plot the difference between 4-fold processed and no detrending
fig5 = plot_global_heatmap(lats, lons, fourfold_diff_mat; 
    title="Difference in Correlation (4x Detrended & Deseasonalized - No Trend)", 
    colorbar_label="Correlation Difference")
fig5.savefig(joinpath(visdir, "fourfold_vs_notrend_difference_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig5)

println("4-fold analysis complete. Additional plots saved to: $visdir")

# Additional analysis: Deseasonalize first, then detrend
println("\nPerforming deseasonalize-first-then-detrend analysis...")

# Create fresh copies for the deseasonalize-first analysis
deseason_first_global_data = copy(copy_ceres_global_rad)
deseason_first_gridded_data = copy(ceres_gridded_rad)

# Step 1: Remove seasonal cycle first for global data
idx_groups = get_seasonal_cycle(months)
global_seasonal_means = calc_seasonal_cycle(deseason_first_global_data, months; aggfunc=agg_func)
deseasonalize!(deseason_first_global_data, idx_groups, global_seasonal_means)

# Step 1: Remove seasonal cycle first for gridded data
for i in axes(deseason_first_gridded_data, 1), j in axes(deseason_first_gridded_data, 2)
    gridded_slice = @view deseason_first_gridded_data[i, j, :]
    gridded_seasonal_means = calc_seasonal_cycle(gridded_slice, months; aggfunc=agg_func)
    deseasonalize!(gridded_slice, idx_groups, gridded_seasonal_means)
end

# Step 2: Then detrend the deseasonalized data for global data
global_fit = least_squares_fit(float_times, deseason_first_global_data)
detrend!(deseason_first_global_data, float_times, global_fit.slope, global_fit.intercept)

# Step 2: Then detrend the deseasonalized data for gridded data
for i in axes(deseason_first_gridded_data, 1), j in axes(deseason_first_gridded_data, 2)
    gridded_slice = @view deseason_first_gridded_data[i, j, :]
    gridded_fit = least_squares_fit(float_times, gridded_slice)
    detrend!(gridded_slice, float_times, gridded_fit.slope, gridded_fit.intercept)
end

# Calculate correlations after deseasonalize-first-then-detrend processing
deseason_first_corr_mat = cor.(eachslice(deseason_first_gridded_data; dims=(1,2)), Ref(deseason_first_global_data))[:,:,1]

# Calculate difference between deseasonalize-first and no detrending
deseason_first_diff_mat = deseason_first_corr_mat .- corr_mats[1]

# Plot the deseasonalize-first-then-detrend correlation matrix
println("Plotting deseasonalize-first-then-detrend correlation matrix...")
fig6 = plot_global_heatmap(lats, lons, deseason_first_corr_mat; 
    title="Global-Gridded Radiation Correlation (Deseasonalize First, Then Detrend)", 
    colorbar_label="Correlation Coefficient")
fig6.savefig(joinpath(visdir, "deseason_first_detrend_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig6)

# Plot the difference between deseasonalize-first and no detrending
fig7 = plot_global_heatmap(lats, lons, deseason_first_diff_mat; 
    title="Difference in Correlation (Deseasonalize First Then Detrend - No Trend)", 
    colorbar_label="Correlation Difference")
fig7.savefig(joinpath(visdir, "deseason_first_vs_notrend_difference_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig7)

println("Deseasonalize-first analysis complete. Additional plots saved to: $visdir")

# Additional analysis: Apply deseasonalize-then-detrend 4 times iteratively
println("\nPerforming 4-fold deseasonalize-then-detrend analysis...")

# Create fresh copies for the 4-fold deseasonalize-then-detrend analysis
fourfold_deseason_first_global_data = copy(copy_ceres_global_rad)
fourfold_deseason_first_gridded_data = copy(ceres_gridded_rad)

# Apply deseasonalize-then-detrend 4 times iteratively
for iteration in 1:4
    println("Iteration $iteration of deseasonalize-then-detrend...")
    
    # Step 1: Remove seasonal cycle first for global data
    idx_groups = get_seasonal_cycle(months)
    global_seasonal_means = calc_seasonal_cycle(fourfold_deseason_first_global_data, months; aggfunc=agg_func)
    deseasonalize!(fourfold_deseason_first_global_data, idx_groups, global_seasonal_means)
    
    # Step 1: Remove seasonal cycle first for gridded data
    for i in axes(fourfold_deseason_first_gridded_data, 1), j in axes(fourfold_deseason_first_gridded_data, 2)
        gridded_slice = @view fourfold_deseason_first_gridded_data[i, j, :]
        gridded_seasonal_means = calc_seasonal_cycle(gridded_slice, months; aggfunc=agg_func)
        deseasonalize!(gridded_slice, idx_groups, gridded_seasonal_means)
    end
    
    # Step 2: Then detrend the deseasonalized data for global data
    global_fit = least_squares_fit(float_times, fourfold_deseason_first_global_data)
    detrend!(fourfold_deseason_first_global_data, float_times, global_fit.slope, global_fit.intercept)
    
    # Step 2: Then detrend the deseasonalized data for gridded data
    for i in axes(fourfold_deseason_first_gridded_data, 1), j in axes(fourfold_deseason_first_gridded_data, 2)
        gridded_slice = @view fourfold_deseason_first_gridded_data[i, j, :]
        gridded_fit = least_squares_fit(float_times, gridded_slice)
        detrend!(gridded_slice, float_times, gridded_fit.slope, gridded_fit.intercept)
    end
end

# Calculate correlations after 4-fold deseasonalize-then-detrend processing
fourfold_deseason_first_corr_mat = cor.(eachslice(fourfold_deseason_first_gridded_data; dims=(1,2)), Ref(fourfold_deseason_first_global_data))[:,:,1]

# Calculate difference between 4-fold deseasonalize-then-detrend and no detrending
fourfold_deseason_first_diff_mat = fourfold_deseason_first_corr_mat .- corr_mats[1]

# Plot the 4-fold deseasonalize-then-detrend correlation matrix
println("Plotting 4-fold deseasonalize-then-detrend correlation matrix...")
figX1 = plot_global_heatmap(lats, lons, fourfold_deseason_first_corr_mat; 
    title="Global-Gridded Radiation Correlation (4x Deseasonalize Then Detrend)", 
    colorbar_label="Correlation Coefficient")
figX1.savefig(joinpath(visdir, "fourfold_deseason_first_detrend_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(figX1)

# Plot the difference between 4-fold deseasonalize-then-detrend and no detrending
figX2 = plot_global_heatmap(lats, lons, fourfold_deseason_first_diff_mat; 
    title="Difference in Correlation (4x Deseasonalize Then Detrend - No Trend)", 
    colorbar_label="Correlation Difference")
figX2.savefig(joinpath(visdir, "fourfold_deseason_first_vs_notrend_difference_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(figX2)

println("4-fold deseasonalize-then-detrend analysis complete. Additional plots saved to: $visdir")

# Additional analysis: Plot slopes of trends remaining after detrending and deseasonalizing once
println("\nAnalyzing residual trends after detrending and deseasonalizing once...")

# Create fresh copies for residual trend analysis
residual_trend_global_data = copy(copy_ceres_global_rad)
residual_trend_gridded_data = copy(ceres_gridded_rad)

# Apply detrend_and_deseasonalize once
detrend_and_deseasonalize!(residual_trend_global_data, float_times, months; 
                          aggfunc=agg_func, trendfunc=least_squares_fit)

detrend_and_deseasonalize_precalculated_groups!.(eachslice(residual_trend_gridded_data; dims=(1,2)), 
                                                Ref(float_times), Ref(month_groups); 
                                                aggfunc=agg_func, trendfunc=least_squares_fit)

# Calculate residual trends for global data
global_residual_fit = least_squares_fit(float_times, residual_trend_global_data)
println("Global residual trend slope: $(global_residual_fit.slope)")

# Calculate residual trends for each grid point
residual_slopes = Matrix{Float64}(undef, size(residual_trend_gridded_data, 1), size(residual_trend_gridded_data, 2))

for i in axes(residual_trend_gridded_data, 1), j in axes(residual_trend_gridded_data, 2)
    gridded_slice = @view residual_trend_gridded_data[i, j, :]
    residual_fit = least_squares_fit(float_times, gridded_slice)
    residual_slopes[i, j] = residual_fit.slope
end

# Plot the residual slopes
println("Plotting residual trend slopes after detrending and deseasonalizing once...")
fig8 = plot_global_heatmap(lats, lons, residual_slopes; 
    title="Residual Trend Slopes After Detrending & Deseasonalizing Once", 
    colorbar_label="Slope (units/year)")
fig8.savefig(joinpath(visdir, "residual_trend_slopes_after_preprocessing.png"), dpi=300, bbox_inches="tight")
plt.close(fig8)

# Print some statistics about the residual slopes
println("Residual slope statistics:")
println("  Mean: $(mean(residual_slopes))")
println("  Std:  $(std(residual_slopes))")
println("  Min:  $(minimum(residual_slopes))")
println("  Max:  $(maximum(residual_slopes))")
println("  Median: $(median(residual_slopes))")

println("Residual trend analysis complete. Plot saved to: $visdir")

# Additional analysis: Plot amplitude of seasonal cycle remaining after deseasonalizing then detrending
println("\nAnalyzing residual seasonal cycle amplitude after deseasonalizing then detrending...")

# Create fresh copies for residual seasonal cycle analysis
residual_seasonal_global_data = copy(copy_ceres_global_rad)
residual_seasonal_gridded_data = copy(ceres_gridded_rad)

# Step 1: Remove seasonal cycle first for global data
idx_groups = get_seasonal_cycle(months)
global_seasonal_means = calc_seasonal_cycle(residual_seasonal_global_data, months; aggfunc=agg_func)
deseasonalize!(residual_seasonal_global_data, idx_groups, global_seasonal_means)

# Step 1: Remove seasonal cycle first for gridded data
for i in axes(residual_seasonal_gridded_data, 1), j in axes(residual_seasonal_gridded_data, 2)
    gridded_slice = @view residual_seasonal_gridded_data[i, j, :]
    gridded_seasonal_means = calc_seasonal_cycle(gridded_slice, months; aggfunc=agg_func)
    deseasonalize!(gridded_slice, idx_groups, gridded_seasonal_means)
end

# Step 2: Then detrend for global data
global_fit = least_squares_fit(float_times, residual_seasonal_global_data)
detrend!(residual_seasonal_global_data, float_times, global_fit.slope, global_fit.intercept)

# Step 2: Then detrend for gridded data
for i in axes(residual_seasonal_gridded_data, 1), j in axes(residual_seasonal_gridded_data, 2)
    gridded_slice = @view residual_seasonal_gridded_data[i, j, :]
    gridded_fit = least_squares_fit(float_times, gridded_slice)
    detrend!(gridded_slice, float_times, gridded_fit.slope, gridded_fit.intercept)
end

# Calculate residual seasonal cycle amplitude for global data
global_residual_seasonal_cycle = calc_seasonal_cycle(residual_seasonal_global_data, months; aggfunc=agg_func)
global_seasonal_amplitude = maximum(global_residual_seasonal_cycle) - minimum(global_residual_seasonal_cycle)
println("Global residual seasonal cycle amplitude: $(global_seasonal_amplitude)")

# Calculate residual seasonal cycle amplitude for each grid point
residual_seasonal_amplitudes = Matrix{Float64}(undef, size(residual_seasonal_gridded_data, 1), size(residual_seasonal_gridded_data, 2))

for i in axes(residual_seasonal_gridded_data, 1), j in axes(residual_seasonal_gridded_data, 2)
    gridded_slice = @view residual_seasonal_gridded_data[i, j, :]
    gridded_residual_seasonal_cycle = calc_seasonal_cycle(gridded_slice, months; aggfunc=agg_func)
    seasonal_amplitude = maximum(gridded_residual_seasonal_cycle) - minimum(gridded_residual_seasonal_cycle)
    residual_seasonal_amplitudes[i, j] = seasonal_amplitude
end

# Plot the residual seasonal cycle amplitudes
println("Plotting residual seasonal cycle amplitudes after deseasonalizing then detrending...")
fig9 = plot_global_heatmap(lats, lons, residual_seasonal_amplitudes; 
    title="Residual Seasonal Cycle Amplitude After Deseasonalizing Then Detrending", 
    colorbar_label="Amplitude (units)")
fig9.savefig(joinpath(visdir, "residual_seasonal_amplitudes_after_preprocessing.png"), dpi=300, bbox_inches="tight")
plt.close(fig9)

# Print some statistics about the residual seasonal amplitudes
println("Residual seasonal amplitude statistics:")
println("  Mean: $(mean(residual_seasonal_amplitudes))")
println("  Std:  $(std(residual_seasonal_amplitudes))")
println("  Min:  $(minimum(residual_seasonal_amplitudes))")
println("  Max:  $(maximum(residual_seasonal_amplitudes))")
println("  Median: $(median(residual_seasonal_amplitudes))")

println("Residual seasonal cycle analysis complete. Plot saved to: $visdir")

# Additional analysis: Detrend each season individually and calculate correlations
println("\nPerforming seasonal detrending analysis...")

# Create fresh copies for seasonal detrending analysis
seasonal_detrend_global_data = copy(copy_ceres_global_rad)
seasonal_detrend_gridded_data = copy(ceres_gridded_rad)

# Detrend each season individually for global data
detrend_each_season_individually!(seasonal_detrend_global_data, float_times, month_groups; trendfunc=least_squares_fit)

# Detrend each season individually for gridded data
for i in axes(seasonal_detrend_gridded_data, 1), j in axes(seasonal_detrend_gridded_data, 2)
    gridded_slice = @view seasonal_detrend_gridded_data[i, j, :]
    detrend_each_season_individually!(gridded_slice, float_times, month_groups; trendfunc=least_squares_fit)
end

# Calculate correlations after seasonal detrending
seasonal_detrend_corr_mat = cor.(eachslice(seasonal_detrend_gridded_data; dims=(1,2)), Ref(seasonal_detrend_global_data))[:,:,1]

# Calculate difference between seasonal detrending and no detrending
seasonal_detrend_diff_mat = seasonal_detrend_corr_mat .- corr_mats[1]

# Plot the seasonal detrending correlation matrix
println("Plotting seasonal detrending correlation matrix...")
fig10 = plot_global_heatmap(lats, lons, seasonal_detrend_corr_mat; 
    title="Global-Gridded Radiation Correlation (Each Season Detrended Individually)", 
    colorbar_label="Correlation Coefficient")
fig10.savefig(joinpath(visdir, "seasonal_detrend_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig10)

# Plot the difference between seasonal detrending and no detrending
fig11 = plot_global_heatmap(lats, lons, seasonal_detrend_diff_mat; 
    title="Difference in Correlation (Seasonal Detrending - No Trend)", 
    colorbar_label="Correlation Difference")
fig11.savefig(joinpath(visdir, "seasonal_detrend_vs_notrend_difference_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig11)

# Calculate residual trends after seasonal detrending
println("Calculating residual trends after seasonal detrending...")

# Calculate residual trend for global data
global_residual_trend_seasonal = least_squares_fit(float_times, seasonal_detrend_global_data)
println("Global residual trend slope after seasonal detrending: $(global_residual_trend_seasonal.slope)")

# Calculate residual trends for each grid point after seasonal detrending
residual_trends_after_seasonal = Matrix{Float64}(undef, size(seasonal_detrend_gridded_data, 1), size(seasonal_detrend_gridded_data, 2))

for i in axes(seasonal_detrend_gridded_data, 1), j in axes(seasonal_detrend_gridded_data, 2)
    gridded_slice = @view seasonal_detrend_gridded_data[i, j, :]
    residual_fit = least_squares_fit(float_times, gridded_slice)
    residual_trends_after_seasonal[i, j] = residual_fit.slope
end

# Plot the residual trends after seasonal detrending
println("Plotting residual trends after seasonal detrending...")
fig12 = plot_global_heatmap(lats, lons, residual_trends_after_seasonal; 
    title="Residual Trend Slopes After Seasonal Detrending", 
    colorbar_label="Slope (units/year)")
fig12.savefig(joinpath(visdir, "residual_trends_after_seasonal_detrending.png"), dpi=300, bbox_inches="tight")
plt.close(fig12)

# Print statistics about residual trends after seasonal detrending
println("Residual trend statistics after seasonal detrending:")
println("  Mean: $(mean(residual_trends_after_seasonal))")
println("  Std:  $(std(residual_trends_after_seasonal))")
println("  Min:  $(minimum(residual_trends_after_seasonal))")
println("  Max:  $(maximum(residual_trends_after_seasonal))")
println("  Median: $(median(residual_trends_after_seasonal))")

println("Seasonal detrending analysis complete. Plots saved to: $visdir")

# Additional analysis: Plot residual seasonal cycle after seasonal detrending
println("\nAnalyzing residual seasonal cycle after seasonal detrending...")

# Calculate residual seasonal cycle amplitude for global data after seasonal detrending
global_residual_seasonal_cycle_after_seasonal_detrend = calc_seasonal_cycle(seasonal_detrend_global_data, months; aggfunc=agg_func)
global_seasonal_amplitude_after_seasonal_detrend = maximum(global_residual_seasonal_cycle_after_seasonal_detrend) - minimum(global_residual_seasonal_cycle_after_seasonal_detrend)
println("Global residual seasonal cycle amplitude after seasonal detrending: $(global_seasonal_amplitude_after_seasonal_detrend)")

# Calculate residual seasonal cycle amplitude for each grid point after seasonal detrending
residual_seasonal_amplitudes_after_seasonal_detrend = Matrix{Float64}(undef, size(seasonal_detrend_gridded_data, 1), size(seasonal_detrend_gridded_data, 2))

for i in axes(seasonal_detrend_gridded_data, 1), j in axes(seasonal_detrend_gridded_data, 2)
    gridded_slice = @view seasonal_detrend_gridded_data[i, j, :]
    gridded_residual_seasonal_cycle = calc_seasonal_cycle(gridded_slice, months; aggfunc=agg_func)
    seasonal_amplitude = maximum(gridded_residual_seasonal_cycle) - minimum(gridded_residual_seasonal_cycle)
    residual_seasonal_amplitudes_after_seasonal_detrend[i, j] = seasonal_amplitude
end

# Plot the residual seasonal cycle amplitudes after seasonal detrending
println("Plotting residual seasonal cycle amplitudes after seasonal detrending...")
fig13 = plot_global_heatmap(lats, lons, residual_seasonal_amplitudes_after_seasonal_detrend; 
    title="Residual Seasonal Cycle Amplitude After Seasonal Detrending", 
    colorbar_label="Amplitude (units)")
fig13.savefig(joinpath(visdir, "residual_seasonal_amplitudes_after_seasonal_detrending.png"), dpi=300, bbox_inches="tight")
plt.close(fig13)

# Print some statistics about the residual seasonal amplitudes after seasonal detrending
println("Residual seasonal amplitude statistics after seasonal detrending:")
println("  Mean: $(mean(residual_seasonal_amplitudes_after_seasonal_detrend))")
println("  Std:  $(std(residual_seasonal_amplitudes_after_seasonal_detrend))")
println("  Min:  $(minimum(residual_seasonal_amplitudes_after_seasonal_detrend))")
println("  Max:  $(maximum(residual_seasonal_amplitudes_after_seasonal_detrend))")
println("  Median: $(median(residual_seasonal_amplitudes_after_seasonal_detrend))")

println("Residual seasonal cycle analysis after seasonal detrending complete. Plot saved to: $visdir")

# Additional analysis: Detrend each season individually using Theil-Sen regression
println("\nPerforming seasonal detrending analysis with Theil-Sen regression...")

# Create fresh copies for Theil-Sen seasonal detrending analysis
theilsen_seasonal_detrend_global_data = copy(copy_ceres_global_rad)
theilsen_seasonal_detrend_gridded_data = copy(ceres_gridded_rad)

# Detrend each season individually for global data using Theil-Sen
detrend_each_season_individually!(theilsen_seasonal_detrend_global_data, float_times, month_groups; trendfunc=theil_sen_fit)

# Detrend each season individually for gridded data using Theil-Sen
for i in axes(theilsen_seasonal_detrend_gridded_data, 1), j in axes(theilsen_seasonal_detrend_gridded_data, 2)
    gridded_slice = @view theilsen_seasonal_detrend_gridded_data[i, j, :]
    detrend_each_season_individually!(gridded_slice, float_times, month_groups; trendfunc=theil_sen_fit)
end

# Calculate correlations after Theil-Sen seasonal detrending
theilsen_seasonal_detrend_corr_mat = corkendall.(eachslice(theilsen_seasonal_detrend_gridded_data; dims=(1,2)), Ref(theilsen_seasonal_detrend_global_data))[:,:,1]

# Calculate difference between Theil-Sen seasonal detrending and no detrending
theilsen_seasonal_detrend_diff_mat = theilsen_seasonal_detrend_corr_mat .- corr_mats[1]

# Plot the Theil-Sen seasonal detrending correlation matrix
println("Plotting Theil-Sen seasonal detrending correlation matrix...")
fig14 = plot_global_heatmap(lats, lons, theilsen_seasonal_detrend_corr_mat; 
    title="Global-Gridded Radiation Correlation (Theil-Sen Seasonal Detrending)", 
    colorbar_label="Correlation Coefficient")
fig14.savefig(joinpath(visdir, "theilsen_seasonal_detrend_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig14)

# Plot the difference between Theil-Sen seasonal detrending and no detrending
fig15 = plot_global_heatmap(lats, lons, theilsen_seasonal_detrend_diff_mat; 
    title="Difference in Correlation (Theil-Sen Seasonal Detrending - No Trend)", 
    colorbar_label="Correlation Difference")
fig15.savefig(joinpath(visdir, "theilsen_seasonal_detrend_vs_notrend_difference_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig15)

# Calculate residual trends after Theil-Sen seasonal detrending
println("Calculating residual trends after Theil-Sen seasonal detrending...")

# Calculate residual trend for global data
global_residual_trend_theilsen_seasonal = theil_sen_fit(float_times, theilsen_seasonal_detrend_global_data)
println("Global residual trend slope after Theil-Sen seasonal detrending: $(global_residual_trend_theilsen_seasonal.slope)")

# Calculate residual trends for each grid point after Theil-Sen seasonal detrending
residual_trends_after_theilsen_seasonal = Matrix{Float64}(undef, size(theilsen_seasonal_detrend_gridded_data, 1), size(theilsen_seasonal_detrend_gridded_data, 2))

for i in axes(theilsen_seasonal_detrend_gridded_data, 1), j in axes(theilsen_seasonal_detrend_gridded_data, 2)
    gridded_slice = @view theilsen_seasonal_detrend_gridded_data[i, j, :]
    residual_fit = theil_sen_fit(float_times, gridded_slice)
    residual_trends_after_theilsen_seasonal[i, j] = residual_fit.slope
end

# Plot the residual trends after Theil-Sen seasonal detrending
println("Plotting residual trends after Theil-Sen seasonal detrending...")
fig16 = plot_global_heatmap(lats, lons, residual_trends_after_theilsen_seasonal; 
    title="Residual Trend Slopes After Theil-Sen Seasonal Detrending", 
    colorbar_label="Slope (units/year)")
fig16.savefig(joinpath(visdir, "residual_trends_after_theilsen_seasonal_detrending.png"), dpi=300, bbox_inches="tight")
plt.close(fig16)

# Print statistics about residual trends after Theil-Sen seasonal detrending
println("Residual trend statistics after Theil-Sen seasonal detrending:")
println("  Mean: $(mean(residual_trends_after_theilsen_seasonal))")
println("  Std:  $(std(residual_trends_after_theilsen_seasonal))")
println("  Min:  $(minimum(residual_trends_after_theilsen_seasonal))")
println("  Max:  $(maximum(residual_trends_after_theilsen_seasonal))")
println("  Median: $(median(residual_trends_after_theilsen_seasonal))")

println("Theil-Sen seasonal detrending analysis complete. Plots saved to: $visdir")

# Additional analysis: Plot remaining trend for the fourfold case
println("\nAnalyzing residual trends after 4-fold detrending and deseasonalizing...")

# Calculate residual trend for global data after 4-fold processing
global_residual_trend_fourfold = least_squares_fit(float_times, fourfold_global_data)
println("Global residual trend slope after 4-fold detrending and deseasonalizing: $(global_residual_trend_fourfold.slope)")

# Calculate residual trends for each grid point after 4-fold processing
residual_trends_after_fourfold = Matrix{Float64}(undef, size(fourfold_gridded_data, 1), size(fourfold_gridded_data, 2))

for i in axes(fourfold_gridded_data, 1), j in axes(fourfold_gridded_data, 2)
    gridded_slice = @view fourfold_gridded_data[i, j, :]
    residual_fit = least_squares_fit(float_times, gridded_slice)
    residual_trends_after_fourfold[i, j] = residual_fit.slope
end

# Plot the residual trends after 4-fold processing
println("Plotting residual trends after 4-fold detrending and deseasonalizing...")
fig17 = plot_global_heatmap(lats, lons, residual_trends_after_fourfold; 
    title="Residual Trend Slopes After 4x Detrending & Deseasonalizing", 
    colorbar_label="Slope (units/year)")
fig17.savefig(joinpath(visdir, "residual_trends_after_fourfold_processing.png"), dpi=300, bbox_inches="tight")
plt.close(fig17)

# Print statistics about residual trends after 4-fold processing
println("Residual trend statistics after 4-fold processing:")
println("  Mean: $(mean(residual_trends_after_fourfold))")
println("  Std:  $(std(residual_trends_after_fourfold))")
println("  Min:  $(minimum(residual_trends_after_fourfold))")
println("  Max:  $(maximum(residual_trends_after_fourfold))")
println("  Median: $(median(residual_trends_after_fourfold))")

println("4-fold residual trend analysis complete. Plot saved to: $visdir")

# Additional analysis: Plot remaining trends and seasonal cycles for the 4-fold deseasonalize-then-detrend case
println("\nAnalyzing residual trends and seasonal cycles after 4-fold deseasonalize-then-detrend...")

# Calculate residual trend for global data after 4-fold deseasonalize-then-detrend processing
global_residual_trend_fourfold_deseason_first = least_squares_fit(float_times, fourfold_deseason_first_global_data)
println("Global residual trend slope after 4-fold deseasonalize-then-detrend: $(global_residual_trend_fourfold_deseason_first.slope)")

# Calculate residual trends for each grid point after 4-fold deseasonalize-then-detrend processing
residual_trends_after_fourfold_deseason_first = Matrix{Float64}(undef, size(fourfold_deseason_first_gridded_data, 1), size(fourfold_deseason_first_gridded_data, 2))

for i in axes(fourfold_deseason_first_gridded_data, 1), j in axes(fourfold_deseason_first_gridded_data, 2)
    gridded_slice = @view fourfold_deseason_first_gridded_data[i, j, :]
    residual_fit = least_squares_fit(float_times, gridded_slice)
    residual_trends_after_fourfold_deseason_first[i, j] = residual_fit.slope
end

# Plot the residual trends after 4-fold deseasonalize-then-detrend processing
println("Plotting residual trends after 4-fold deseasonalize-then-detrend...")
figY1 = plot_global_heatmap(lats, lons, residual_trends_after_fourfold_deseason_first; 
    title="Residual Trend Slopes After 4x Deseasonalize Then Detrend", 
    colorbar_label="Slope (units/year)")
figY1.savefig(joinpath(visdir, "residual_trends_after_fourfold_deseason_first_processing.png"), dpi=300, bbox_inches="tight")
plt.close(figY1)

# Print statistics about residual trends after 4-fold deseasonalize-then-detrend processing
println("Residual trend statistics after 4-fold deseasonalize-then-detrend:")
println("  Mean: $(mean(residual_trends_after_fourfold_deseason_first))")
println("  Std:  $(std(residual_trends_after_fourfold_deseason_first))")
println("  Min:  $(minimum(residual_trends_after_fourfold_deseason_first))")
println("  Max:  $(maximum(residual_trends_after_fourfold_deseason_first))")
println("  Median: $(median(residual_trends_after_fourfold_deseason_first))")

# Calculate residual seasonal cycle amplitude for global data after 4-fold deseasonalize-then-detrend
global_residual_seasonal_cycle_fourfold_deseason_first = calc_seasonal_cycle(fourfold_deseason_first_global_data, months; aggfunc=agg_func)
global_seasonal_amplitude_fourfold_deseason_first = maximum(global_residual_seasonal_cycle_fourfold_deseason_first) - minimum(global_residual_seasonal_cycle_fourfold_deseason_first)
println("Global residual seasonal cycle amplitude after 4-fold deseasonalize-then-detrend: $(global_seasonal_amplitude_fourfold_deseason_first)")

# Calculate residual seasonal cycle amplitude for each grid point after 4-fold deseasonalize-then-detrend
residual_seasonal_amplitudes_fourfold_deseason_first = Matrix{Float64}(undef, size(fourfold_deseason_first_gridded_data, 1), size(fourfold_deseason_first_gridded_data, 2))

for i in axes(fourfold_deseason_first_gridded_data, 1), j in axes(fourfold_deseason_first_gridded_data, 2)
    gridded_slice = @view fourfold_deseason_first_gridded_data[i, j, :]
    gridded_residual_seasonal_cycle = calc_seasonal_cycle(gridded_slice, months; aggfunc=agg_func)
    seasonal_amplitude = maximum(gridded_residual_seasonal_cycle) - minimum(gridded_residual_seasonal_cycle)
    residual_seasonal_amplitudes_fourfold_deseason_first[i, j] = seasonal_amplitude
end

# Plot the residual seasonal cycle amplitudes after 4-fold deseasonalize-then-detrend
println("Plotting residual seasonal cycle amplitudes after 4-fold deseasonalize-then-detrend...")
figY2 = plot_global_heatmap(lats, lons, residual_seasonal_amplitudes_fourfold_deseason_first; 
    title="Residual Seasonal Cycle Amplitude After 4x Deseasonalize Then Detrend", 
    colorbar_label="Amplitude (units)")
figY2.savefig(joinpath(visdir, "residual_seasonal_amplitudes_after_fourfold_deseason_first.png"), dpi=300, bbox_inches="tight")
plt.close(figY2)

# Print some statistics about the residual seasonal amplitudes after 4-fold deseasonalize-then-detrend
println("Residual seasonal amplitude statistics after 4-fold deseasonalize-then-detrend:")
println("  Mean: $(mean(residual_seasonal_amplitudes_fourfold_deseason_first))")
println("  Std:  $(std(residual_seasonal_amplitudes_fourfold_deseason_first))")
println("  Min:  $(minimum(residual_seasonal_amplitudes_fourfold_deseason_first))")
println("  Max:  $(maximum(residual_seasonal_amplitudes_fourfold_deseason_first))")
println("  Median: $(median(residual_seasonal_amplitudes_fourfold_deseason_first))")

println("4-fold deseasonalize-then-detrend residual analysis complete. Plots saved to: $visdir")

# Verification: Check that original data arrays have not been modified
println("\nVerifying that original data arrays remain unchanged...")

# Check if copy_ceres_global_rad still has trends and seasonal cycles
original_global_trend = least_squares_fit(float_times, copy_ceres_global_rad)
original_global_seasonal_cycle = calc_seasonal_cycle(copy_ceres_global_rad, months; aggfunc=agg_func)
original_global_seasonal_amplitude = maximum(original_global_seasonal_cycle) - minimum(original_global_seasonal_cycle)

println("Original global data verification:")
println("  Trend slope: $(original_global_trend.slope) (should be non-zero if original trends preserved)")
println("  Seasonal amplitude: $(original_global_seasonal_amplitude) (should be large if original seasonality preserved)")

# Check a few sample grid points from ceres_gridded_rad
sample_points = [(10, 10), (50, 100), (120, 180)]  # Sample lat/lon indices
println("\nOriginal gridded data verification (sample points):")

for (i, (lat_idx, lon_idx)) in enumerate(sample_points)
    if lat_idx <= size(ceres_gridded_rad, 1) && lon_idx <= size(ceres_gridded_rad, 2)
        gridded_slice = @view ceres_gridded_rad[lat_idx, lon_idx, :]
        gridded_trend = least_squares_fit(float_times, gridded_slice)
        gridded_seasonal_cycle = calc_seasonal_cycle(gridded_slice, months; aggfunc=agg_func)
        gridded_seasonal_amplitude = maximum(gridded_seasonal_cycle) - minimum(gridded_seasonal_cycle)
        
        println("  Point $i (lat_idx=$lat_idx, lon_idx=$lon_idx):")
        println("    Trend slope: $(gridded_trend.slope)")
        println("    Seasonal amplitude: $(gridded_seasonal_amplitude)")
    end
end

# Quick statistical check: compare standard deviations
# Original data should have higher variability than processed data
original_global_std = std(copy_ceres_global_rad)
processed_global_std = std(fourfold_global_data)  # One of the heavily processed versions

println("\nStandard deviation comparison:")
println("  Original global data std: $(original_global_std)")
println("  4x processed global data std: $(processed_global_std)")
println("  Ratio (original/processed): $(original_global_std / processed_global_std) (should be > 1)")

if original_global_std > processed_global_std && abs(original_global_trend.slope) > 1e-10 && original_global_seasonal_amplitude > 0.1
    println("\n✓ VERIFICATION PASSED: Original data arrays appear to be unchanged")
else
    println("\n✗ VERIFICATION FAILED: Original data arrays may have been modified!")
    println("  Check if any analysis accidentally modified copy_ceres_global_rad or ceres_gridded_rad")
end

println("\nAnalysis complete. All results saved to: $visdir")

# Final sanity check: Plot correlation between original global and gridded radiation data
println("\nFinal sanity check: Plotting correlation between original global and gridded radiation data...")

# Calculate correlations using the original, unprocessed data
original_corr_mat = cor.(eachslice(ceres_gridded_rad; dims=(1,2)), Ref(copy_ceres_global_rad))[:,:,1]

# Plot the original correlation matrix
println("Plotting original (unprocessed) correlation matrix...")
fig_original = plot_global_heatmap(lats, lons, original_corr_mat; 
    title="Global-Gridded Radiation Correlation (Original Unprocessed Data)", 
    colorbar_label="Correlation Coefficient")
fig_original.savefig(joinpath(visdir, "original_unprocessed_correlation_matrix.png"), dpi=300, bbox_inches="tight")
plt.close(fig_original)

# Print some statistics about the original correlations
println("Original correlation statistics:")
println("  Mean: $(mean(original_corr_mat))")
println("  Std:  $(std(original_corr_mat))")
println("  Min:  $(minimum(original_corr_mat))")
println("  Max:  $(maximum(original_corr_mat))")
println("  Median: $(median(original_corr_mat))")

# Create 4x4 cross-correlation difference matrix plot
println("\nCreating 4x4 cross-correlation difference matrix plot...")

# Select 4 key correlation matrices for comparison
corr_matrices = [
    deseason_first_corr_mat,           # Deseasonalize first, then detrend
    fourfold_corr_mat,                 # 4x detrend and deseasonalize  
    fourfold_deseason_first_corr_mat,  # 4x deseasonalize then detrend
    seasonal_detrend_corr_mat          # Each season detrended individually
]

method_names = [
    "Deseason→Detrend",
    "4x Detrend&Deseason", 
    "4x Deseason→Detrend",
    "Seasonal Detrend"
]

# Create figure with 4x4 subplots
fig_cross, plot_axes = plt.subplots(4, 4, figsize=(20, 20))
fig_cross.suptitle("Cross-Correlation Difference Matrix (Method A - Method B)", fontsize=16, y=0.98)

# Define a symmetric colorbar range for differences
vmin_diff, vmax_diff = -0.3, 0.3

for i in 1:4
    for j in 1:4
        ax = plot_axes[i-1, j-1]
        
        if i == j
            # Diagonal: Show the correlation matrix itself
            corr_data = corr_matrices[i]
            im = ax.imshow(corr_data, cmap="RdBu_r", vmin=-1, vmax=1, 
                          extent=[minimum(lons), maximum(lons), minimum(lats), maximum(lats)],
                          aspect="auto", origin="lower")
            ax.set_title("$(method_names[i])\n(Correlation Matrix)", fontsize=10)
            
            # Add colorbar for correlation
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Correlation", fontsize=8)
        else
            # Off-diagonal: Show difference between methods
            diff_data = corr_matrices[i] .- corr_matrices[j]
            im = ax.imshow(diff_data, cmap="RdBu_r", vmin=vmin_diff, vmax=vmax_diff,
                          extent=[minimum(lons), maximum(lons), minimum(lats), maximum(lats)], 
                          aspect="auto", origin="lower")
            ax.set_title("$(method_names[i]) -\n$(method_names[j])", fontsize=10)
            
            # Add colorbar for differences
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Difference", fontsize=8)
        end
        
        # Set axis labels and formatting
        if i == 4  # Bottom row
            ax.set_xlabel("Longitude", fontsize=9)
        else
            ax.set_xticklabels([])
        end
            
        if j == 1  # Left column  
            ax.set_ylabel("Latitude", fontsize=9)
        else
            ax.set_yticklabels([])
        end
            
        # Add grid lines
        ax.grid(true, alpha=0.3)
        
        # Set tick label sizes
        ax.tick_params(labelsize=8)
    end
end

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Save the figure
fig_cross.savefig(joinpath(visdir, "cross_correlation_difference_matrix_4x4.png"), 
                  dpi=300, bbox_inches="tight")
plt.close(fig_cross)

println("4x4 cross-correlation difference matrix plot saved to: $visdir")

# Print some statistics about the differences
println("\nCross-correlation difference statistics:")
for i in 1:4
    for j in 1:4
        if i != j
            diff_data = corr_matrices[i] .- corr_matrices[j]
            mean_diff = mean(diff_data)
            std_diff = std(diff_data)
            max_abs_diff = maximum(abs.(diff_data))
            println("  $(method_names[i]) - $(method_names[j]):")
            println("    Mean difference: $(round(mean_diff, digits=4))")
            println("    Std difference:  $(round(std_diff, digits=4))")
            println("    Max |difference|: $(round(max_abs_diff, digits=4))")
        end
    end
end

