cd(@__DIR__)
include("../utils/utilfuncs.jl")

using DataFrames, CSV, Statistics, Plots, Dates

df = CSV.read("/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/new_zealand_point_series.csv", DataFrame)

# Extract time and data series
time = df[:, 1]
data_series = df[:, 2]

float_times = calc_float_time.(time)
months = month.(time)
initial_fit = least_squares_fit(float_times, data_series)

p = plot(time, data_series, title="Original Data Series", xlabel="Time", ylabel="Value", legend=false)
plot!(p, time, @.(initial_fit.slope * float_times + initial_fit.intercept), color=:red, label="Initial Trend Line")
display(p)

#Now detrend and deseasonalize the data, and plot the residual trend as well
detrended = copy(data_series)

detrend!(detrended, float_times, initial_fit.slope, initial_fit.intercept)
residual_trend = least_squares_fit(float_times, detrended)

p = plot(time, detrended, title="Detrended Data Series", xlabel="Time", ylabel="Value", legend=false)
plot!(p, time, @.(residual_trend.slope * float_times + residual_trend.intercept), color=:red, label="Residual Trend Line")
display(p)

#Now remove the seasonal cycle too
detrended_deseasoned = copy(detrended)
current_seasonal_cycle = calc_seasonal_cycle(detrended_deseasoned, months)
seasonal_cycle_only_time_series = [current_seasonal_cycle[m] for m in months]

#Now plot the current seasonal cycle over the detrended data
p = plot(time, detrended, title="Detrended Data with Seasonal Cycle", xlabel="Time", ylabel="Value", legend=false)
plot!(p, time, seasonal_cycle_only_time_series, color=:green, label="Seasonal Cycle")
display(p)

deseasonalize!(detrended_deseasoned, months)

final_trend = least_squares_fit(float_times, detrended_deseasoned)
p = plot(time, detrended_deseasoned, title="Detrended and Deseasonalized Data Series", xlabel="Time", ylabel="Value", legend=false)
plot!(p, time, @.(final_trend.slope * float_times + final_trend.intercept), color=:red, label="Final Trend Line")
display(p)

#Now remove that trend too
fully_detrended_deseasoned = copy(detrended_deseasoned)
detrend!(fully_detrended_deseasoned, float_times, final_trend.slope, final_trend.intercept)

p = plot(time, fully_detrended_deseasoned, title="Fully Detrended and Deseasonalized Data Series", xlabel="Time", ylabel="Value", legend=false)
display(p)

#Now plot this data's seasonal cycle
current_seasonal_cycle = calc_seasonal_cycle(fully_detrended_deseasoned, months)
seasonal_cycle_only_time_series = [current_seasonal_cycle[m] for m in months]
p = plot(time, fully_detrended_deseasoned, title="Fully Detrended and Deseasonalized Data with Seasonal Cycle", xlabel="Time", ylabel="Value", legend=false)
plot!(p, time, seasonal_cycle_only_time_series, color=:green, label="Seasonal Cycle")
display(p)

#Returning to the original dataset, deseasonalize it and then detrend it
deseason_only_data = randn(289)
current_raw_seasonal_cycle = calc_seasonal_cycle(deseason_only_data, months)
raw_seasonal_cycle_only_time_series = [current_raw_seasonal_cycle[m] for m in months]

#Calculate the trend too 

p = plot(time, data_series, title="Original Data with Seasonal Cycle", xlabel="Time", ylabel="Value", legend=false)
plot!(p, time, raw_seasonal_cycle_only_time_series, color=:green, label="Seasonal Cycle")
display(p)

#Now remove the seasonal cycle