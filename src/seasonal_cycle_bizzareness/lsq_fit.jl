"""
Investigate the order dependency of detrending and deseasonalizing operations
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

visdir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/vis/why_ceres_why"

using Statistics, Plots

# Create synthetic data with both trend and seasonal cycle
t = 1:100
months = repeat(1:10, 10)
y = @. t/33 + sin(2π * months / 10)  # Linear trend + seasonal cycle
y .-= mean(y)

display(plot(t, y, label="Original Data", lw=2, title="Synthetic Data with Trend and Seasonal Cycle", xlabel="Time", ylabel="Value"))

using LsqFit
trend_and_cycle(t, p) = @. (p[1] + p[2] * t + p[3] * sin(p[4] * t - p[5]))
p0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
fit = curve_fit(trend_and_cycle, t, y, p0)

fit_values = trend_and_cycle.(t, Ref(fit.param))

display(plot(t, fit_values, label="Fitted Trend + Seasonal Cycle", lw=2, title="Fitted Model", xlabel="Time", ylabel="Value"))