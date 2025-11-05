This repository contains a Simulation based inference (SBI) framework to fine-tune kinetics-ODE models parameters. Classical inference often relies on an explicit likelihood function, which can be intractable or unavailable for complex mechanistic models. SBI bypasses this limitation by treating the mechanistic model as a simulator: given a set of inputs x, the simulator produces outputs y.  Instead of relying on closed-form likelihoods, SBI leverages these simulated input–output pairs to train a neural density estimator (Coupling Flow), which can approximate the posterior distribution p(y | x). This makes it possible to infer which inputs (e.g., reaction fluxes or constraints) are most consistent with observed experimental data, even when the mechanistic model is nonlinear or high-dimensional.

![photo](data/BOIS WP6 (7).png)

The framework draws inspiration from:
- Cranmera et al (2020): methodlogy
- Millard et al (2021): data
- https://bayesflow.org/main/_examples/SIR_Posterior_Estimation.html (Code implementation)

Overflow metabolism refers to the production of seemingly wasteful by-products by cells during growth on glucose even when oxygen is abundant. Millard develop a kinetic model of E. coli metabolism that quantitatively accounts for observed behaviours and successfully predicts the response of E. coli to new perturbations. However: the model has 10 paramters to fine-tune and NPE is applied. We initialized the inference by uniformly sampling each parameter within its respective log-scale range as reported by Millard. The training procedure employed simulated extracellular glucose and acetate concentrations, together with concentration trajectories (including biomass), as inputs to a Coupling-Flow network to learn their joint distribution. Once trained, the network is fitted with experimental biomass curve to apporximate its parameters. These result is confirmed by fitting back to ODE models for generating a biomass curve for comparision 
