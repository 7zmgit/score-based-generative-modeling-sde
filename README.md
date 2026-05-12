# Score-Based Sampling on MNIST: Euler–Maruyama vs. Predictor–Corrector

**Author:** Hazem Ajlan

Score-based generative models turn Gaussian noise into images by simulating a reverse-time stochastic differential equation. The simplest way to do this is the Euler–Maruyama method. Alternatively, Song et al. (2020) propose the Predictor–Corrector (PC) sampler, which adds Langevin Markov Chain Monte Carlo (MCMC) steps at each noise level, and report that it gives better samples on CIFAR-10.

But each Langevin step costs another neural-network evaluation, so PC trades compute for quality. So, I thought of the next question.

## The question
Does Predictor-Corrector still win when both samplers get the same compute budget?

## What I did
- Trained a small (~1.6M param) U-Net on MNIST for 15 epochs using denoising score matching
- Implemented Euler-Maruyama to match the official code & Predictor-Corrector to match the paper's algorithm
- Tuned the SNR hyperparameter for PC separately for `M=1` and `M=2` corrector steps
- Compared Euler-Maruyama vs Predictor-Corrector (M=1) vs Predictor-Corrector (M=2) at four matched NFE budgets, 3 seeds each
- Verified PC reduces to Euler-Maruyama in the M=0 limit (sanity check)
- Evaluated with Frechet Inception Distance (FID) against 10,000 real MNIST test images
  
## What I found

**EM beats PC at every compute budget tested.** This is the opposite of the paper's CIFAR-10 result.

| NFE | EM | PC (M=1) | PC (M=2) |
|----:|---:|---------:|---------:|
| 121 | 236.16 | 357.58 | 378.47 |
| 241 | 141.70 | 232.40 | 311.98 |
| 481 | **11.71** | 138.85 | 182.23 |
| 961 | **10.86** | 17.76 | 99.37 |

FID, lower is better. Mean ± std over 3 seeds in the full report.

The gap is dramatic at intermediate budgets; at NFE 481, EM is roughly 12× better than PC(M=1). At the largest budget the gap narrows but doesn't close. Adding more corrector steps (M=2) consistently makes things worse, not better.

My best guess for why: with a small score model trained briefly, the score function has meaningful error. The Langevin corrector runs MCMC toward the stationary distribution of the *learned* score, not the true one, so running it harder amplifies the error rather than correcting it. With a large, well-trained model (like the paper's CIFAR-10 setup) the score error is small enough that PC helps. With my setup, it hurts.

In the report, I include more detail and discuss the limitations.

---

## Repo structure

```
code/      PyTorch implementation (training, samplers, FID, sweeps)
figures/   Generated samples, training curves, FID plots
report.pdf Full write-up
```

## Running it

```bash
pip install torch torchvision pytorch-fid tqdm matplotlib
# Then open code/main.ipynb in Colab or Jupyter
```

## References

- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021. ([paper](https://arxiv.org/abs/2011.13456), [code](https://github.com/yang-song/score_sde_pytorch))
- Anderson, *Reverse-time diffusion equation models*, Stoch. Process. Appl., 1982.
- Vincent, *A connection between score matching and denoising autoencoders*, Neural Comp., 2011.
