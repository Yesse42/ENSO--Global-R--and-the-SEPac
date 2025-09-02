using CSV, DataFrames, Dates, Plots, LinearAlgebra, Statistics
using StatsBase # For spearman correlation
gr() # Use GR backend for Plots.jl

# Add utils to path
include("../utils/utilfuncs.jl")

# Load data
println("Loading data...")

# Load SEPac SST, ENSO, and residual data
enso_sepac_data = CSV.read("../../data/v1draft_saved_data/enso_sepac_correlation_results.csv", DataFrame)

# Load flux data
flux_data = CSV.read("../../data/SEPac_SST/sepac_flux_time_series.csv", DataFrame)

# Convert date columns to DateTime
enso_sepac_data.Date = DateTime.(enso_sepac_data.Date)
flux_data.Date = DateTime.(flux_data.Date)

# Create year-month keys for matching (since dates don't align exactly)
enso_sepac_data.YearMonth = [Date(year(d), month(d), 1) for d in enso_sepac_data.Date]
flux_data.YearMonth = [Date(year(d), month(d), 1) for d in flux_data.Date]

# Merge the datasets on YearMonth
merged_data = DataFrames.innerjoin(enso_sepac_data, flux_data, on=:YearMonth, makeunique=true)

# Use the flux dates as the primary date column (since they're monthly data points)
merged_data.Date = merged_data.Date_1  # Use flux dates

println("Data loaded. $(nrow(merged_data)) time points available.")

# Extract time series
dates = merged_data.Date
sepac_sst = merged_data.sepac_sst_index
oni = merged_data.oni_at_optimal_lag
sepac_residual = merged_data.sepac_sst_residual

# Extract flux variables
latent_heat = merged_data.SEPac_Latent_Heat_Flux
sensible_heat = merged_data.SEPac_Sensible_Heat_Flux
net_sw = merged_data.SEPac_Net_SW_Radiation
net_lw = merged_data.SEPac_Net_LW_Radiation

# Calculate float times for detrending
float_times = [calc_float_time(d) for d in dates]
months = [month(d) for d in dates]

println("Detrending and deseasonalizing time series...")

# Detrend and deseasonalize all time series
sepac_sst_detrended = copy(sepac_sst)
detrend_and_deseasonalize!(sepac_sst_detrended, float_times, months)

oni_detrended = copy(oni)
detrend_and_deseasonalize!(oni_detrended, float_times, months)

# Detrend and deseasonalize flux variables
latent_heat_detrended = copy(latent_heat)
detrend_and_deseasonalize!(latent_heat_detrended, float_times, months)

sensible_heat_detrended = copy(sensible_heat)
detrend_and_deseasonalize!(sensible_heat_detrended, float_times, months)

net_sw_detrended = copy(net_sw)
detrend_and_deseasonalize!(net_sw_detrended, float_times, months)

net_lw_detrended = copy(net_lw)
detrend_and_deseasonalize!(net_lw_detrended, float_times, months)

println("Removing ENSO effects from flux time series via linear regression...")

# Remove ENSO effects from flux variables via linear regression
function remove_enso_effect(flux_var, oni_var)
    # Create design matrix [ones(n) oni_var]
    n = length(flux_var)
    X = hcat(ones(n), oni_var)
    
    # Solve for coefficients: β = (X'X)^(-1)X'y
    β = X \ flux_var
    
    # Calculate residuals (flux with ENSO effect removed)
    residuals = flux_var - X * β
    
    return residuals, β
end

# Remove ENSO effects from detrended flux variables
latent_heat_enso_removed, latent_β = remove_enso_effect(latent_heat_detrended, oni_detrended)
sensible_heat_enso_removed, sensible_β = remove_enso_effect(sensible_heat_detrended, oni_detrended)
net_sw_enso_removed, sw_β = remove_enso_effect(net_sw_detrended, oni_detrended)
net_lw_enso_removed, lw_β = remove_enso_effect(net_lw_detrended, oni_detrended)

println("Saving processed data...")

# Create output DataFrame with processed data
output_data = DataFrame(
    Date = dates,
    sepac_sst_detrended_deseasonalized = sepac_sst_detrended,
    oni_detrended_deseasonalized = oni_detrended,
    latent_heat_detrended_deseasonalized = latent_heat_detrended,
    sensible_heat_detrended_deseasonalized = sensible_heat_detrended,
    net_sw_detrended_deseasonalized = net_sw_detrended,
    net_lw_detrended_deseasonalized = net_lw_detrended,
    latent_heat_enso_removed = latent_heat_enso_removed,
    sensible_heat_enso_removed = sensible_heat_enso_removed,
    net_sw_enso_removed = net_sw_enso_removed,
    net_lw_enso_removed = net_lw_enso_removed
)

# Save to CSV
CSV.write("../../data/v1draft_saved_data/sepac_flux_processed_data.csv", output_data)

println("Creating visualization directory...")

# Create visualization directory
vis_dir = "../../vis/sepac_flux_effects"
if !isdir(vis_dir)
    mkpath(vis_dir)
end

println("Creating plots...")

# Create plots
function create_flux_comparison_plot(dates, sepac_sst, flux_vars, flux_names, title_suffix, filename)
    # Create subplot layout
    p_sst = plot(dates, sepac_sst, 
                title="SEPac SST Index $title_suffix",
                ylabel="SST Index",
                color=:red,
                linewidth=2,
                legend=false)
    hline!([0], color=:black, linestyle=:dash, alpha=0.5)
    
    # Create flux plots with correlation information
    flux_plots = []
    for (flux_var, flux_name) in zip(flux_vars, flux_names)
        # Calculate correlations
        pearson_corr = cor(sepac_sst, flux_var)
        spearman_corr = corspearman(sepac_sst, flux_var)
        
        # Create title with correlation info
        corr_title = "$flux_name\nPearson: $(round(pearson_corr, digits=3)), Spearman: $(round(spearman_corr, digits=3))"
        
        p = plot(dates, flux_var, 
                title=corr_title,
                ylabel=flux_name,
                color=:blue,
                linewidth=1.5,
                legend=false,
                titlefontsize=8)
        hline!([0], color=:black, linestyle=:dash, alpha=0.5)
        push!(flux_plots, p)
    end
    
    # Combine all plots
    combined_plot = plot(p_sst, flux_plots..., 
                        layout=(5,1), 
                        size=(1200, 1000),
                        xlabel="Year")
    
    # Save plot
    savefig(combined_plot, joinpath(vis_dir, filename))
    return combined_plot
end

# Plot 1: Detrended and deseasonalized data
flux_vars_detrended = [latent_heat_detrended, sensible_heat_detrended, net_sw_detrended, net_lw_detrended]
flux_names = ["Latent Heat Flux", "Sensible Heat Flux", "Net SW Radiation", "Net LW Radiation"]

fig1 = create_flux_comparison_plot(
    dates, 
    sepac_sst_detrended, 
    flux_vars_detrended, 
    flux_names,
    "(Detrended & Deseasonalized)",
    "sepac_sst_flux_detrended_deseasonalized.png"
)

# Plot 2: ENSO effects removed
flux_vars_enso_removed = [latent_heat_enso_removed, sensible_heat_enso_removed, net_sw_enso_removed, net_lw_enso_removed]

fig2 = create_flux_comparison_plot(
    dates, 
    sepac_sst_detrended, 
    flux_vars_enso_removed, 
    flux_names,
    "(Detrended & Deseasonalized, ENSO Effects Removed from Fluxes)",
    "sepac_sst_flux_enso_removed.png"
)

println("Analysis complete!")
println("Processed data saved to: data/v1draft_saved_data/sepac_flux_processed_data.csv")
println("Plots saved to: vis/sepac_flux_effects/")
println()
println("ENSO regression coefficients:")
println("Latent Heat: intercept = $(round(latent_β[1], sigdigits=3)), slope = $(round(latent_β[2], sigdigits=3))")
println("Sensible Heat: intercept = $(round(sensible_β[1], sigdigits=3)), slope = $(round(sensible_β[2], sigdigits=3))")
println("Net SW: intercept = $(round(sw_β[1], sigdigits=3)), slope = $(round(sw_β[2], sigdigits=3))")
println("Net LW: intercept = $(round(lw_β[1], sigdigits=3)), slope = $(round(lw_β[2], sigdigits=3))")
