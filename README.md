# Score-Based Sampling on MNIST: Euler–Maruyama vs. Predictor–Corrector

**Author:** Hazem Ajlan

After reading Song et al.’s paper *Score-Based Generative Modeling through Stochastic Differential Equations* many times, I thought of this question: 
> Predictor–Corrector sampling sounds better than plain Euler–Maruyama; but is it still better if both samplers get the same compute budget?

That question drives this project. I train a small variance-exploding (VE) score-based generative model on the MNIST (handwritten digits) dataset, then compare two reverse-time samplers:

- **Euler–Maruyama (EM):** spend every score-network call on moving through the reverse SDE.
- **Predictor–Corrector (PC):** spend some score-network calls on Langevin correction steps.

Since score-network evaluations are the expensive part of sampling, I compare the samplers at matched number of score-function evaluations a.k.a **NFE**.

---

## What this repo contains

- A time-conditioned U-Net (Convolutional Neural Network type) score model, around 1.6M parameters
- VE-SDE training on MNIST using denoising score matching
- Euler–Maruyama sampling
- Predictor–Corrector sampling with `M=1` and `M=2` corrector steps
- Final Tweedie denoising for all samplers
- SNR tuning for the PC corrector
- A matched-NFE FID comparison
- A sanity check showing that PC with `M=0` behaves like EM
---

## The main result

In this setup, **Euler–Maruyama wins**.

| NFE | EM FID | PC (M=1) FID | PC (M=2) FID |
|----:|-------:|-------------:|-------------:|
| 121 | 236.16 | 357.58 | 378.47 |
| 241 | 141.70 | 232.40 | 311.98 |
| 481 | **11.71** | 138.85 | 182.23 |
| 961 | **10.86** | 17.76 | 99.37 |

Lower FID is better. The full report includes standard deviations over 3 seeds.

The surprising part is not just that PC loses. It loses even after tuning the corrector SNR separately for `M=1` and `M=2`.

So my takeaway is:

> For this small MNIST VE-SDE model, the score evaluations spent on Langevin corrector steps were better spent on more Euler–Maruyama predictor steps.

---

Song et al.’s paper reports strong results for PC on larger natural-image settings. My result does **not** contradict that in general. It just shows that PC is not automatically better once you charge it fairly for the extra score-network evaluations.

---

## My hypothesis

The corrector uses Langevin dynamics based on the learned score model. If the score model is imperfect, the corrector may refine samples toward the distribution implied by the learned score, not the true noisy data distribution.

So in a small model trained briefly, running more corrector steps might amplify score-model errors instead of fixing sampling errors.

That is only a hypothesis. I did not directly measure score bias here. But it fits the observation that `M=2` is usually worse than `M=1`.

---

## Sanity check

To make sure PC was not simply implemented incorrectly, I tested the `M=0` case.

When `M=0`, PC has no corrector steps, so it should reduce to Euler–Maruyama.

At `N = P = 240`:

| Sampler | FID |
|--------:|----:|
| EM | 147.376 |
| PC, M=0 | 147.745 |

The difference is tiny relative to the FID scale, so the predictor path and NFE accounting appear consistent.

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

A GPU helps a lot. The full pipeline (train + tune + sweep) is ~45 minutes on an L4.

## References

- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021. ([paper](https://arxiv.org/abs/2011.13456), [code](https://github.com/yang-song/score_sde_pytorch))
- Anderson, *Reverse-time diffusion equation models*, Stoch. Process. Appl., 1982.
- Vincent, *A connection between score matching and denoising autoencoders*, Neural Comp., 2011.
