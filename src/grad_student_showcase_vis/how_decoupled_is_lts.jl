"""
Decompose LTS into a component where the surface and aloft are correlated and a component where they are wholly uncorrelated.
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")
include("../pls_regressor/pls_functions.jl")

using Plots, LinearAlgebra, Statistics, CSV, DataFrames, Dates

time_period = (Date(2000, 3), Date(2024, 3, 31))

data_savedir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/local_aloft_separated"

sepac_datadir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/sepac_lts_data/local_region_time_series"
era5_var_file = joinpath(sepac_datadir, "era5_region_avg_SEPac_feedback_definition.csv")
era5_data = CSV.read(era5_var_file, DataFrame)
era5_times = era5_data.date

enso_data, enso_time = load_enso_data(time_period)
enso_time = enso_time["time"]
enso_time = round.(enso_time, Month(1), RoundDown)

times = sort!(intersect(era5_times, enso_time))
float_times = calc_float_time.(times)
month_groups = get_seasonal_cycle(month.(times))

enso_valid_times = findall(t -> t in times, enso_time)

filter!(row -> row.date in times, era5_data)
for (lag, vec) in pairs(enso_data)
    enso_data[lag] = vec[enso_valid_times]
end

for col in eachcol(era5_data)
    eltype(col) <: Dates.AbstractDateTime && continue
    deseasonalize_and_detrend_precalculated_groups_twice!(col, float_times, month_groups)
end

full_enso_df = DataFrame([[enso_time[enso_valid_times]]; collect(enso_data)], [["date"]; string.(collect(keys(enso_data)))])

joint_df = DataFrames.innerjoin(era5_data, full_enso_df, on=:date)
dropmissing!(joint_df)

theta_1000_data = era5_data[!, :θ_1000]
theta_700_data = era5_data[!, :θ_700]
lts_data = era5_data[!, :LTS_1000]

"Decompose y1 into a component correlated with y2 and a component uncorrelated with y2."
function decompose_via_pls(joint_df, y)
    # Define lags from -24 to 24
    lags = -24:24
    
    # Create PLS X matrix from all lagged ENSO data in joint_df
    lag_columns = ["oni_lag_$lag" for lag in lags]
    pls_X = Matrix(joint_df[!, lag_columns])
    pls_Y = y
    
    # Set number of components and create PLS model
    n_components = 1
    pls_model = make_pls_regressor(pls_X, pls_Y, n_components; print_updates=false)
    
    # Get the scores (predicted values)
    predicted_y = vec(predict(pls_model, pls_X))
    
    # Calculate correlation
    pls_corr = cor(predicted_y, pls_Y)
    println("Correlation between PLS-predicted values from ENSO and actual values: $pls_corr")
    
    # Return the correlated component (predicted) and uncorrelated component (residual)
    correlated_component = predicted_y
    uncorrelated_component = pls_Y - predicted_y
    
    return correlated_component, uncorrelated_component, pls_corr
end

function remove_via_linear_correlaton(y1, y2)
    lstqfit = least_squares_fit(y1, y2)
    predicted_y2 = y1 .* lstqfit.slope .+ lstqfit.intercept
    residual_y2 = y2 .- predicted_y2
    corr = cor(predicted_y2, y2)
    return predicted_y2, residual_y2, corr
end

# Extract variables from joint_df to avoid missing data issues
theta_1000_joint = joint_df[!, :θ_1000]
theta_700_joint = joint_df[!, :θ_700]

# Apply PLS to remove ENSO influence from theta_1000 and theta_700 using all lags
println("Applying PLS to remove ENSO influence from theta_1000...")
theta_1000_corr_comp, theta_1000_residual, theta_1000_pls_corr = decompose_via_pls(joint_df, theta_1000_joint)

println("Applying PLS to remove ENSO influence from theta_700...")
theta_700_corr_comp, theta_700_residual, theta_700_pls_corr = decompose_via_pls(joint_df, theta_700_joint)

# Calculate correlations between original time series
original_corr = cor(theta_1000_joint, theta_700_joint)

# Calculate correlations between residual time series (ENSO-removed)
residual_corr = cor(theta_1000_residual, theta_700_residual)

# Calculate correlations between ENSO components (ENSO-explained parts)
enso_component_corr = cor(theta_1000_corr_comp, theta_700_corr_comp)

# Print results
println("\n" * "="^60)
println("CORRELATION ANALYSIS RESULTS")
println("="^60)
println("Original theta_1000 vs theta_700 correlation: $(round(original_corr, digits=4))")
println("Residual theta_1000 vs theta_700 correlation (ENSO removed): $(round(residual_corr, digits=4))")
println("ENSO component theta_1000 vs theta_700 correlation: $(round(enso_component_corr, digits=4))")
println("\nChange in correlation (residual vs original): $(round(residual_corr - original_corr, digits=4))")
println("Percentage change (residual vs original): $(round(100 * (residual_corr - original_corr) / original_corr, digits=2))%")
println("\nDecomposition verification:")
println("  Original = ENSO component + Residual component")
println("  Residual component correlation: $(round(residual_corr, digits=4))")
println("  ENSO component correlation: $(round(enso_component_corr, digits=4))")
println("\nPLS model performance:")
println("  theta_1000 ~ ENSO correlation: $(round(theta_1000_pls_corr, digits=4))")
println("  theta_700 ~ ENSO correlation: $(round(theta_700_pls_corr, digits=4))")
println("="^60)

# Calculate LTS from residual components (ENSO-removed)
lts_residual = theta_700_residual - theta_1000_residual
lts_original = joint_df[!, :LTS_1000]

# Calculate correlation between original and residual LTS
lts_corr_original_residual = cor(lts_original, lts_residual)

# Function to calculate PLS fit skill metrics
function calculate_pls_skill(joint_df, target_var)
    lags = -24:24
    lag_columns = ["oni_lag_$lag" for lag in lags]
    pls_X = Matrix(joint_df[!, lag_columns])
    pls_Y = target_var
    
    n_components = 1
    pls_model = make_pls_regressor(pls_X, pls_Y, n_components; print_updates=false)
    predicted_y = vec(predict(pls_model, pls_X))
    
    # Calculate skill metrics
    correlation = cor(predicted_y, pls_Y)
    r_squared = correlation^2
    rmse = sqrt(mean((predicted_y - pls_Y).^2))
    mae = mean(abs.(predicted_y - pls_Y))
    bias = mean(predicted_y - pls_Y)
    
    # Normalized metrics
    rmse_normalized = rmse / std(pls_Y)
    mae_normalized = mae / std(pls_Y)
    
    return (
        correlation = correlation,
        r_squared = r_squared,
        rmse = rmse,
        mae = mae,
        bias = bias,
        rmse_normalized = rmse_normalized,
        mae_normalized = mae_normalized,
        predicted = predicted_y
    )
end

# Calculate PLS skill for original LTS
println("\nCalculating PLS skill for original LTS...")
lts_original_skill = calculate_pls_skill(joint_df, lts_original)

# Calculate PLS skill for ENSO-removed LTS
println("Calculating PLS skill for ENSO-removed LTS...")
lts_residual_skill = calculate_pls_skill(joint_df, lts_residual)

# Create enhanced plot comparing original LTS, residual-based LTS, and PLS predictions
dates = joint_df[!, :date]
plot_title = "LTS Analysis: Original vs ENSO-Removed vs PLS Predictions"

p = plot(dates, lts_original, 
         label="Original LTS (r²=$(round(lts_original_skill.r_squared, digits=3)))", 
         linewidth=2, 
         color=:blue,
         title=plot_title,
         xlabel="Date",
         ylabel="LTS (K)",
         legend=:topright,
         size=(1000, 600))

plot!(p, dates, lts_residual, 
      label="ENSO-removed LTS (r²=$(round(lts_residual_skill.r_squared, digits=3)))", 
      linewidth=2, 
      color=:red,
      linestyle=:dash)

plot!(p, dates, lts_original_skill.predicted, 
      label="PLS prediction (original)", 
      linewidth=1, 
      color=:green,
      alpha=0.7)

plot!(p, dates, lts_residual_skill.predicted, 
      label="PLS prediction (residual)", 
      linewidth=1, 
      color=:orange,
      alpha=0.7,
      linestyle=:dot)

display(p)

# Create a scatter plot showing PLS skill comparison
p2 = plot(layout=(1,2), size=(1200, 500))

# Original LTS vs PLS prediction
scatter!(p2, lts_original, lts_original_skill.predicted, 
         subplot=1,
         alpha=0.6,
         color=:blue,
         xlabel="Actual Original LTS (K)",
         ylabel="PLS Predicted LTS (K)",
         title="Original LTS PLS Skill\n(r² = $(round(lts_original_skill.r_squared, digits=3)))",
         legend=false)

# Add 1:1 line
lts_range = [minimum(lts_original), maximum(lts_original)]
plot!(p2, lts_range, lts_range, subplot=1, color=:black, linestyle=:dash, alpha=0.5)

# ENSO-removed LTS vs PLS prediction  
scatter!(p2, lts_residual, lts_residual_skill.predicted,
         subplot=2,
         alpha=0.6,
         color=:red,
         xlabel="Actual ENSO-removed LTS (K)",
         ylabel="PLS Predicted LTS (K)",
         title="ENSO-removed LTS PLS Skill\n(r² = $(round(lts_residual_skill.r_squared, digits=3)))",
         legend=false)

# Add 1:1 line
lts_res_range = [minimum(lts_residual), maximum(lts_residual)]
plot!(p2, lts_res_range, lts_res_range, subplot=2, color=:black, linestyle=:dash, alpha=0.5)

display(p2)

println("\nLTS Analysis:")
println("Correlation between original LTS and ENSO-removed LTS: $(round(lts_corr_original_residual, digits=4))")
println("Original LTS std: $(round(std(lts_original), digits=4)) K")
println("Residual LTS std: $(round(std(lts_residual), digits=4)) K")
println("Variance reduction: $(round(100 * (1 - var(lts_residual)/var(lts_original)), digits=1))%")

# Print comprehensive PLS skill analysis
println("\n" * "="^80)
println("PLS FIT SKILL ANALYSIS")
println("="^80)

println("\nORIGINAL LTS ~ ENSO PLS SKILL:")
println("  Correlation (r): $(round(lts_original_skill.correlation, digits=4))")
println("  R-squared (r²): $(round(lts_original_skill.r_squared, digits=4))")
println("  RMSE: $(round(lts_original_skill.rmse, digits=4)) K")
println("  MAE: $(round(lts_original_skill.mae, digits=4)) K")
println("  Bias: $(round(lts_original_skill.bias, digits=4)) K")
println("  Normalized RMSE: $(round(lts_original_skill.rmse_normalized, digits=4))")
println("  Normalized MAE: $(round(lts_original_skill.mae_normalized, digits=4))")

println("\nENSO-REMOVED LTS ~ ENSO PLS SKILL:")
println("  Correlation (r): $(round(lts_residual_skill.correlation, digits=4))")
println("  R-squared (r²): $(round(lts_residual_skill.r_squared, digits=4))")
println("  RMSE: $(round(lts_residual_skill.rmse, digits=4)) K")
println("  MAE: $(round(lts_residual_skill.mae, digits=4)) K")
println("  Bias: $(round(lts_residual_skill.bias, digits=4)) K")
println("  Normalized RMSE: $(round(lts_residual_skill.rmse_normalized, digits=4))")
println("  Normalized MAE: $(round(lts_residual_skill.mae_normalized, digits=4))")

println("\nSKILL COMPARISON (Original vs ENSO-removed):")
skill_diff_r2 = lts_original_skill.r_squared - lts_residual_skill.r_squared
skill_diff_rmse = lts_original_skill.rmse - lts_residual_skill.rmse
skill_diff_mae = lts_original_skill.mae - lts_residual_skill.mae

println("  Δ R-squared: $(round(skill_diff_r2, digits=4)) (reduction of $(round(100*skill_diff_r2/lts_original_skill.r_squared, digits=1))%)")
println("  Δ RMSE: $(round(skill_diff_rmse, digits=4)) K")
println("  Δ MAE: $(round(skill_diff_mae, digits=4)) K")

if lts_residual_skill.r_squared < 0.01
    println("  ✓ ENSO removal successful: residual LTS shows minimal ENSO predictability")
else
    println("  ⚠ ENSO removal incomplete: residual LTS still shows ENSO predictability")
end

println("="^80)
