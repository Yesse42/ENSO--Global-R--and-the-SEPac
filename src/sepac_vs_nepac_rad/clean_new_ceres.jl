"""
This script cleans and processes the new CERES dataset. It will help me avoid annoying sign errors lol
"""
cd(@__DIR__)
include("../utils/plot_global.jl")
include("../utils/load_funcs.jl")
using NCDatasets, StatsBase, Dates

ceres_data_dir = "/Users/C837213770/Desktop/Research Code/ENSO, Global R, and the SEPac/data/CERES/NEW"
ceres_gridded_ds = Dataset(joinpath(ceres_data_dir, "ceres_gridded.nc"), "r")
ceres_global_ds = Dataset(joinpath(ceres_data_dir, "ceres_global.nc"), "r")

