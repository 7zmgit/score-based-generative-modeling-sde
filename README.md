# Score-Based Generative Modeling: Predictor–Corrector vs. Euler–Maruyama

**Author:** Hazem Ajlan

A small empirical study of two ways to sample from a score-based generative model trained on MNIST. The model and forward process follow Song et al. (2021), using a variance-exploding SDE. The interesting part is the comparison between samplers: **does the fancier Predictor–Corrector sampler actually beat plain Euler–Maruyama when you give them the same compute budget?**

## The question

Generating an image with a score-based model means simulating a reverse-time SDE that turns Gaussian noise into data. The simplest way to do this is the **Euler–Maruyama (EM)** method — discretize time, take one step at a time, done. Song et al. propose a more sophisticated **Predictor–Corrector (PC)** sampler that adds extra Langevin MCMC steps at each noise level, and report that it gives better samples on CIFAR-10.

But each Langevin step costs another neural-network evaluation. So a fair comparison has to fix the total number of score evaluations (NFE) and ask whether the corrector steps earn their cost. That's the experiment here.

## What I did

- Trained a small (~1.6M param) U-Net on MNIST for 15 epochs using denoising score matching
- Implemented both samplers, matching the official reference code line-for-line
- Tuned the SNR hyperparameter for PC separately for `M=1` and `M=2` corrector steps
- Compared EM vs PC(M=1) vs PC(M=2) at four matched NFE budgets, 3 seeds each
- Verified PC reduces to EM in the M=0 limit (sanity check)
- Evaluated with FID against 10,000 real MNIST test images

## What I found

**EM beats PC at every compute budget tested.** This is the opposite of the paper's CIFAR-10 result.

| NFE | EM | PC (M=1) | PC (M=2) |
|----:|---:|---------:|---------:|
| 121 | 236.16 | 357.58 | 378.47 |
| 241 | 141.70 | 232.40 | 311.98 |
| 481 | **11.71** | 138.85 | 182.23 |
| 961 | **10.86** | 17.76 | 99.37 |

FID, lower is better. Mean ± std over 3 seeds in the full report.

The gap is dramatic at intermediate budgets — at NFE 481, EM is roughly 12× better than PC(M=1). At the largest budget the gap narrows but doesn't close. Adding more corrector steps (M=2) consistently makes things worse, not better.

My best guess for *why*: with a small score model trained briefly, the score function has meaningful error. The Langevin corrector runs MCMC toward the stationary distribution of the *learned* score, not the true one — so running it harder amplifies the error rather than correcting it. With a large, well-trained model (like the paper's CIFAR-10 setup) the score error is small enough that PC helps. With our setup it hurts.

The full write-up explains this in more detail and discusses limitations.

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
