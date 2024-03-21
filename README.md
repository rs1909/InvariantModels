# InvariantModels
## A Julia package to calculate data-driven and equation-driven reduced order models

See detailed documentation and a case study here: [Documentation](https://rs1909.github.io/InvariantModels/)

The key concept behind methods implemented here is invariance. The following methods are implemented both for autonomous and (quasi-) peridocically forced systems.
    * Invariant foliations
        * from data
        * from discrete-time systems
        * from differential equations
    * Invariant manifolds from 
        * differential equations
        * discrete-time systems (maps)
        * cannot be done directly from data (regardless of what others claim), but can be done from foliations
    * Invariant manifolds from two invariant foliations
    * Accurate instantaneous frequency and damping ratio calculations

The package has been used to demonstrate the methods in papers
    1. R. Szalai. Machine-learning invariant foliations in forced systems for reduced order modelling, preprint, 2024.
    2. R. Szalai. Non-resonant invariant foliations of quasi-periodically forced systems, preprint, 2024

There are four examples. For paper 1
    * A forced [traffic dynamics model](examples/carfollow).
    * The forced [Shaw-Pierre](examples/carfollow) example.
For paper 2:
    * A [Geometrically nonlinear](examples/onemass) two degree-of-freedom oscillator
    * A [two-mass oscillator](examples/onemass).

This package makes [the previous version](https://rs1909.github.io/FMA/) obsolete.

[![Build Status](https://github.com/rs1909/InvariantModels/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rs1909/InvariantModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
