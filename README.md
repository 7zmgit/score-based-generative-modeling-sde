# Score-Based Generative Modeling with Variance-Exploding SDEs

**Author:** Hazem Ajlan

This project implements a score-based generative model following Song et al. (2021), using a variance-exploding (VE) stochastic differential equation.

I train a neural network via denoising score matching to approximate the score function ∇ₓ log pₜ(x), and generate samples by solving the reverse-time SDE using the Euler–Maruyama method.

## Focus

The main goal is to study the numerical behavior of reverse-time sampling, specifically:

- how the number of discretization steps affects sample quality  
- how the diffusion scale (σ) influences stability  

## Key Observations

- Increasing the number of steps significantly improves sample quality  
- Coarse discretization leads to noisy outputs  
- Sampling is sensitive to mismatch in σ  

## Structure

- `code/` — implementation in PyTorch  
- `Figures/` — generated samples and plots  

## References

- Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021  
- Anderson (1982), *Reverse-Time Diffusion Equation Models*
