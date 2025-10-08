"""
Investigate the utter bizarreness of the hemispheric correlation patterns
"""

cd(@__DIR__)
include("../utils/load_funcs.jl")
include("../utils/utilfuncs.jl")
include("../utils/plot_global.jl")

using Statistics, Plots

t = 1:100
months = repeat(1:10, 10)
y = randn(100)

#Plot and remove the seasonal cycle
seasonal_cycle = calc_seasonal_cycle(y, months)
seasonal_cycle_full_time = [seasonal_cycle[m] for m in months]

#Now plot the data with the seasonal cycle
p = plot(t, y, label="Original Data", lw=2)
plot!(t, seasonal_cycle_full_time, label="Seasonal Cycle", lw=2)
display(p)

#Now remove the seasonal cycle
y_season_removed = copy(y)
deseasonalize!(y_season_removed, months)

#Now calculate the trend
deseason_trend = least_squares_fit(t, y_season_removed)

#Now plot the trend and the deseasonalized data
p2 = plot(t, y_season_removed, label="Deseasonalized Data", lw=2)
plot!(t, deseason_trend.slope .* t .+ deseason_trend.intercept, label="Trend", lw=2)
display(p2)

#Now remove the trend, recalculate the seasonal cycle, and plot
y_deseason_trend_removed = copy(y_season_removed)
detrend!(y_deseason_trend_removed, t, deseason_trend.slope, deseason_trend.intercept)
recalc_seasonal_cycle = calc_seasonal_cycle(y_deseason_trend_removed, months)
recalc_seasonal_cycle_full_time = [recalc_seasonal_cycle[m] for m in months]

p3 = plot(t, y_deseason_trend_removed, label="Deseasonalized & Detrended Data", lw=2)
plot!(t, recalc_seasonal_cycle_full_time, label="Recalculated Seasonal Cycle", lw=2)
display(p3)

#Now return to the original data, calculate and remove the trend first, then the seasonal cycle
y_trend_removed = copy(y)
y_trend = least_squares_fit(t, y_trend_removed)

#Plot the original data with the trend
p4 = plot(t, y_trend_removed, label="Original Data", lw=2)
plot!(t, y_trend.slope .* t .+ y_trend.intercept, label="Trend", lw=2)
display(p4)

#Remove the trend and plot the resultant seasonal cycle
detrend!(y_trend_removed, t, y_trend.slope, y_trend.intercept)
seasonal_cycle = calc_seasonal_cycle(y_trend_removed, months)
seasonal_cycle_full_time = [seasonal_cycle[m] for m in months]
p5 = plot(t, y_trend_removed, label="Detrended Data", lw=2)
plot!(t, seasonal_cycle_full_time, label="Seasonal Cycle", lw=2)
display(p5)

#Now calculate the trend associated with the seasonal cycle
seasonal_cycle_trend = least_squares_fit(t, seasonal_cycle_full_time)
println("Trend associated with seasonal cycle: slope=$(seasonal_cycle_trend.slope), intercept=$(seasonal_cycle_trend.intercept)")

#Now remove the seasonal cycle and calculate the resultant trend
y_season_removed = copy(y_trend_removed)
deseasonalize!(y_season_removed, months)
deseason_trend = least_squares_fit(t, y_season_removed)
println("Trend after removing seasonal cycle: slope=$(deseason_trend.slope), intercept=$(deseason_trend.intercept)")
