# Score-Based Sampling on MNIST: Euler–Maruyama vs. Predictor–Corrector

**Author:** Hazem Ajlan

This repository contains a small empirical study of reverse-time sampling for score-based generative models. I train a variance-exploding (VE) score model on MNIST and compare Euler–Maruyama (EM) sampling with Predictor–Corrector (PC) sampling under matched numbers of score-function evaluations (NFE).

The goal is not to reproduce state-of-the-art results. The goal is to test a narrower question:

> If score-network evaluations are the main sampling cost, is it better to spend them on more Euler–Maruyama predictor steps, or to trade some of them for Langevin corrector steps?

## What is implemented

- VE-SDE score model on MNIST
- Small time-conditioned U-Net score network (~1.6M parameters)
- Continuous-time denoising score matching
- Euler–Maruyama sampler
- Predictor–Corrector sampler with `M=1` and `M=2` corrector steps
- Final Tweedie denoising for all samplers
- SNR tuning for PC
- Matched-NFE FID comparison
- `M=0` sanity check showing that PC reduces to EM when no corrector steps are used

## Main result

In this MNIST setup, EM outperforms PC at every matched NFE budget tested.

| NFE | EM FID | PC (M=1) FID | PC (M=2) FID |
|----:|-------:|-------------:|-------------:|
| 121 | 236.16 | 357.58 | 378.47 |
| 241 | 141.70 | 232.40 | 311.98 |
| 481 | **11.71** | 138.85 | 182.23 |
| 961 | **10.86** | 17.76 | 99.37 |

Lower FID is better. Values are means over 3 seeds; standard deviations are reported in the paper.

The result suggests that, for this small MNIST VE-SDE model, the score evaluations spent on Langevin corrector steps were less useful than additional Euler–Maruyama predictor steps.

This should not be read as a general claim that PC sampling is worse than EM. It is a controlled result for this dataset, architecture, training budget, SDE, and evaluation setup.

## Interpretation

A plausible explanation is that the learned score model is imperfect. Langevin correction uses the learned score, so if the score has systematic error, extra corrector steps may move samples toward the distribution implied by the learned score rather than the true noisy data distribution. This could make corrector steps less useful, especially for a small model trained briefly.

This explanation is only a hypothesis; the project does not directly measure score error.

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

A GPU helps a lot. The full pipeline (train + tune + sweep) is ~45 minutes on an L4.

## References

- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021. ([paper](https://arxiv.org/abs/2011.13456), [code](https://github.com/yang-song/score_sde_pytorch))
- Anderson, *Reverse-time diffusion equation models*, Stoch. Process. Appl., 1982.
- Vincent, *A connection between score matching and denoising autoencoders*, Neural Comp., 2011.
