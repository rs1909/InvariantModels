module InvariantModels

using ApproxFun
using ManifoldsBase, Manifolds
using Manopt
using DynamicPolynomials, MultivariatePolynomials
using TaylorSeries
using Tullio
using TensorOperations
using LoopVectorization
using Clustering
using Printf
using Statistics
using RecursiveArrayTools
using UnicodePlots
using StatsBase
using DifferentialEquations
using Random
using BSON
using BSON: @load, @save

# add ApproxFun ManifoldsBase Manifolds Manopt DynamicPolynomials MultivariatePolynomials TaylorSeries Tullio TensorOperations LoopVectorization Clustering Printf Statistics CairoMakie GLMakie UnicodePlots StatsBase StatsPlots DifferentialEquations Random BSON RecursiveArrayTools Revise Plots

# fundamentals and semi-analytic part
include("qppolynomial.jl")
export getgrid
export QPConstant
export QPPolynomial, fromFunction!, fromFunction
export Eval!, Eval
export mapFromODE
export toFourier

include("qptorusid.jl")
include("qpbundles.jl")
include("qpdirectmanifold.jl")
export QPODETorusManifold
export QPMAPTorusManifold

include("qpdirectfoliation.jl")
export QPGraphStyleFoliations

include("qpfrequencydamping.jl")
export ODEFrequencyDamping
export MAPFrequencyDamping

include("qplinearid.jl")
export findLinearModel

# the data-driven part
include("qpgenerate.jl")
export generate, generateMap

include("qptensor.jl")
include("qpcompressedpolynomial.jl")
include("qpcombinedfoliation.jl")
export scale_epsilon
export norm_beta
export bumpU_D, bumpU_C, bumpU_C, bumpU_F, bumpU_P
export bumpV_D, bumpV_C, bumpV_S, bumpV_F, bumpV_P
export QPCombinedFoliation
export makeCache, updateCache!
export QPPreProcess, QPOptimise, QPPostProcess
export EvalU, EvalV
export Loss, ResidualU, ErrorStatistics, ResidualLoss
export QPKernel, copyMost!, SetTorus!
export to_zero
export bincuts

end
