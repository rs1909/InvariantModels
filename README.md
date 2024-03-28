# InvariantModels
## A Julia package to calculate data-driven and equation-driven reduced order models

See detailed documentation and a case study [here](https://rs1909.github.io/InvariantModels/).

The key concept behind methods implemented here is invariance. The following methods are implemented both for autonomous and (quasi-) peridocically forced systems.

  * Invariant foliations

    + from data
    + from discrete-time systems
    + from differential equations
    
  * Invariant manifolds from
  
    + differential equations
    + discrete-time systems (maps)
    + cannot be done directly from data (regardless of what others claim), but can be extracted from foliations
    
  * Invariant manifolds from two invariant foliations
  
  * Accurate instantaneous frequency and damping ratio calculations

The package has been used to demonstrate the methods in papers

  1. *R. Szalai.* Machine-learning invariant foliations in forced systems for reduced order modelling, *[preprint](https://arxiv.org/abs/2403.14514)*, 2024 
  2. *R. Szalai.* Non-resonant invariant foliations of quasi-periodically forced systems, *[preprint](https://arxiv.org/abs/2403.14771)*, 2024

There are two other papers that explain the background of the methods.

  3. *R. Szalai*, Data-Driven Reduced Order Models Using Invariant Foliations, Manifolds and Autoencoders, J Nonlinear Sci 33, 75 (2023). [link](https://doi.org/10.1007/s00332-023-09932-y)
  4. *R. Szalai*, Invariant spectral foliations with applications to model order reduction and synthesis. Nonlinear Dyn 101, 2645–2669 (2020). [link](https://doi.org/10.1007/s11071-020-05891-1)
 
Paper [4] introduced the idea of using invariant foliations for reduced order modelling, paper [3] has shown that only invariant foliations can be used for genuine data-driven reduced order modelling (when we classify all possible methods into: a) autoencoders, b) invariant foliations, c) invariant manifolds, d) equation-free models.

There are four examples. For paper 1:

  * A forced [traffic dynamics model](examples/carfollow)
  * The forced [Shaw-Pierre](examples/carfollow) example

  For paper 2:

  * A [geometrically nonlinear](examples/onemass) two degree-of-freedom oscillator
  * A [two-mass oscillator](examples/onemass)

This package makes [the previous version](https://rs1909.github.io/FMA/) obsolete.

[![Build Status](https://github.com/rs1909/InvariantModels/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rs1909/InvariantModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
