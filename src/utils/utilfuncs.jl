using SplitApplyCombine, Dates, Statistics, StatsBase, NCDatasets, Dictionaries, Unitful, RollingFunctions, LinearAlgebra

current_dir = pwd()
cd(@__DIR__)
include("calculate_eis.jl")
include("regression.jl")
include("seasonal.jl")
include("detrending.jl")
include("daily.jl")
include("spatial.jl")
include("time_utils.jl")
include("enso_utils.jl")
cd(current_dir)
