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
y = t/33 + sin.(2π .* months ./ 10)  # Linear trend + seasonal cycle
y .-= mean(y)

n_months_in_year = 10  # Define the number of months in a year

function remove_seasonal_cycle_via_FFT!(y, n_months_in_year)

end