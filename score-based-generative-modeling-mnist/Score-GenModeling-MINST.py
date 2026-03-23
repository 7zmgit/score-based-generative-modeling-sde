# Score-Based Generative Modeling with Variance-Exploding SDEs
#
# Author: Hazem Ajlan
# Date: July 2025
#
# Description:
# This script implements a score-based generative model using the
# variance-exploding (VE) stochastic differential equation (SDE) framework
# of Song et al. (2021). We train neural network via denoising score
# matching to approximate the time-dependent score function ∇_x log p_t(x),
# and generate samples by numerically solving the reverse-time SDE
# using the Euler–Maruyama method.
#
# The implementation is adapted from the official tutorial code:
# https://github.com/yang-song/score_sde
#
# This code also examines the effect of discretization (number of reverse-time steps) and diffusion scale (sigma) on sample quality.

import math
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Reproducibility 
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if device.type == "cuda":
    torch.cuda_manual_seed_all(seed)

# We use the variance-exploding (VE) SDE
#     dX_t = sigma^t dW_t,
# so the diffusion coefficient is g(t) = sigma^t.
#
# For this SDE, the perturbation kernel p_{0t}(x(t) | x(0)) is Gaussian, and
# its standard deviation can be written in closed form.
# This lets us sample noisy training inputs directly, and scale the network output appropriately across noise levels.

def marginal_prob_std(t, sigma):
    """
    Standard deviation of p_{0t}(x(t) | x(0)) for the VE SDE:
      dx = sigma^t dW_t

    Args:
        t: tensor of shape (batch,)
        sigma: scalar > 1
    Returns:
        std: tensor of shape (batch,)
    """
    t = t.to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / (2. * math.log(sigma)))

def diffusion_coeff(t, sigma):
    """
    Diffusion coefficient g(t) = sigma^t.
    """
    t = t.to(device)
    return sigma**t

sigma = 25.0  # as in the tutorial
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn    = functools.partial(diffusion_coeff,    sigma=sigma)

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier features for time embedding.
    Maps scalar t to high-dimensional embedding using sin/cos.
    """
    def __init__(self, embed_dim=128, scale=30.):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = scale
        # Random frequencies; fixed
        self.register_buffer(
            "W", torch.randn(embed_dim // 2) * scale
        )

    def forward(self, t):
        # Reshape from (B,) to (B,1) so each time value can be multiplied by
        # the random frequencies.
        t = t.view(-1, 1)

        # Project time onto random Fourier frequencies, then apply sin/cos.
        # Output shape: (B, embed_dim)
        proj = t * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TimeMLP(nn.Module):
    """
    Simple MLP that converts time embedding -> per-channel bias.
    """
    def __init__(self, time_embed_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t_emb):
        # t_emb: (B, time_embed_dim)
        return self.net(t_emb)  # (B, out_dim)


class ResidualBlock(nn.Module):
    """
    Residual convolutional block with time conditioning.

    Each block processes image features x and receives a time embedding.
    The time embedding is converted into a channel-wise bias and added to the
    feature map so that the network can adapt its behavior across noise levels.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = TimeMLP(time_embed_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        # First normalization + nonlinearity + convolution
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # # Convert time embedding to a channel-wise bias and inject it into the
        # feature map. This lets the same network behave differently for small
        # t (low noise) and large t (high noise).
        time_out = self.time_mlp(t_emb)
        h = h + time_out[..., None, None]

        # Second normalization + nonlinearity + convolution
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class ScoreNet(nn.Module):
    """
    Time-dependent score network s_theta(x, t) for MNIST.

    Goal:
        Approximate the score i.e. the gradient of the log-density of the perturbed data distribution.

    Architecture:
        A small U-Net-style model:
        - encoder extracts coarse features,
        - bottleneck processes compressed representation,
        - decoder reconstructs spatial detail using skip connections.

    Why U-Net? The score is an image-shaped vector field, so we want both local detail and global context.
    """
    def __init__(self, marginal_prob_std, time_embed_dim=128, base_channels=32):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.time_embed = GaussianFourierProjection(embed_dim=time_embed_dim)

        # Encoder
        self.conv_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(base_channels, base_channels, time_embed_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.res2 = ResidualBlock(base_channels * 2, base_channels * 2, time_embed_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.res3 = ResidualBlock(base_channels * 4, base_channels * 4, time_embed_dim)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.res4 = ResidualBlock(base_channels * 4, base_channels * 2, time_embed_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.res5 = ResidualBlock(base_channels * 2, base_channels, time_embed_dim)

        self.conv_out = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x: noisy image, shape (B, 1, 28, 28)
            t: time / noise level, shape (B,)
        Returns:
            Estimated score field of the same shape as x
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_embed_dim)

        # Encoder
        h1 = self.conv_in(x)
        h1 = self.res1(h1, t_emb)   # (B, C, 28, 28)

        h2 = self.down1(h1)         # (B, 2C, 14, 14)
        h2 = self.res2(h2, t_emb)   # (B, 2C, 14, 14)

        h3 = self.down2(h2)         # (B, 4C, 7, 7)
        h3 = self.res3(h3, t_emb)   # (B, 4C, 7, 7)

        # Decoder with skip connections
        u1 = self.up1(h3)           # (B, 2C, 14, 14)
        u1 = torch.cat([u1, h2], dim=1)  # (B, 4C, 14, 14)
        u1 = self.res4(u1, t_emb)   # (B, 2C, 14, 14)

        u2 = self.up2(u1)           # (B, C, 28, 28)
        u2 = torch.cat([u2, h1], dim=1)  # (B, 2C, 28, 28)
        u2 = self.res5(u2, t_emb)   # (B, C, 28, 28)

        out = self.conv_out(u2)     # (B, 1, 28, 28)

        # We scale by 1 / std(t) like in the tutorial
        std = self.marginal_prob_std(t)  # (B,)
        out = out / std[:, None, None, None]

        return out

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """
    Continuous-time denoising score matching loss.

    Training idea:
        1. Sample a random time t
        2. Corrupt clean data x with Gaussian noise at that time
        3. Ask the network to estimate the score of the noisy distribution

    For the VE SDE, the perturbation kernel is Gaussian, so the score of
    p_{0t}(x(t) | x(0)) is known analytically. This gives a tractable target
    for training.
    """
    # Random times, avoid exactly 0 or 1
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)  # (B,)

    # Perturb data
    perturbed_x = x + z * std[:, None, None, None]

    # Model prediction
    score = model(perturbed_x, random_t)

    # Weighted MSE
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2,
                                dim=(1, 2, 3)))
    return loss

# Transform to tensor in [0,1]
transform = transforms.ToTensor()

train_dataset = MNIST(root=".", train=True, transform=transform, download=True)
test_dataset  = MNIST(root=".", train=False, transform=transform, download=True)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

len(train_dataset), len(test_dataset)

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)


optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-4)

n_epochs = 15

train_losses = []

# (Training loop)
# Each epoch:
#   - draws a mini batch of clean MNIST images,
#   - corrupts them at random times t,
#   - trains the network to estimate the corresponding score.

score_model.train()

for epoch in range(1, n_epochs + 1):
    running_loss = 0.0
    num_batches = 0

    for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}"):
        x = x.to(device)

        optimizer.zero_grad()
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

if epoch % 5 == 0:
    torch.save(score_model.state_dict(), f"mnist_score_model_epoch{epoch}.pt")

# Plot training loss
plt.figure()
plt.plot(train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Score model training loss on MNIST")
plt.grid(True)
plt.show()

def show_samples(samples, nrow=8):
    """
    Display a grid of generated samples (B, 1, 28, 28)
    """
    import torch
    import matplotlib.pyplot as plt

    samples = samples[:nrow*nrow]
    grid = torch.zeros(1, nrow * 28, nrow * 28)

    idx = 0
    for i in range(nrow):
        for j in range(nrow):
            grid[:, i*28:(i+1)*28, j*28:(j+1)*28] = samples[idx]
            idx += 1

    plt.figure(figsize=(6,6))
    plt.imshow(grid.squeeze(0), cmap='gray')
    plt.axis('off')
    plt.title('Generated Samples')
    plt.show()

@torch.no_grad()
def euler_maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=500,
                           device=device,
                           eps=1e-3):
    """
    Reverse SDE sampler using Euler–Maruyama.

    Sample from the learned model by discretizing the reverse-time SDE.

    We start from x(T), which is approximately pure Gaussian noise, and move
    backward toward t=0. At each step:
        - the score network provides the drift direction,
        - a Gaussian term accounts for stochasticity of the reverse diffusion.

    Euler-Maruyama is a first-order numerical method, so sample quality is
    sensitive to the number of steps used.
    """
    score_model.eval()

    # Initial time and initial x ~ p_1 (Gaussian)
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]

    # Discretize time from 1 to small epsilon rather than 0 for numerical stability
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]  # positive number

    x = init_x

    for i, time_step in enumerate(tqdm(time_steps, desc="Sampling")):
        batch_t = torch.ones(batch_size, device=device) * time_step
        g = diffusion_coeff(batch_t)  # g(t)

        # Reverse SDE: x_{t-dt} = x_t + g(t)^2 s_theta(x_t, t) dt + g(t) sqrt(dt) z_t
        score = score_model(x, batch_t)
        mean_x = x + (g**2)[:, None, None, None] * score * step_size

        # Add noise except possibly at final step
        z = torch.randn_like(x)
        x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * z

    return mean_x


# Generate some samples
samples = euler_maruyama_sampler(
    score_model,
    marginal_prob_std_fn,
    diffusion_coeff_fn,
    batch_size=64,
    num_steps=200,  # you can increase to 500 if you want better quality
    device=device,
    eps=1e-3
).cpu()

samples = samples.clamp(0.0, 1.0)
show_samples(samples, nrow=8)

"""## Ablation Study: Effect of $\sigma$ and Sampling Steps

In this section, we study how two hyperparameters affect the sample quality:

1. The **diffusion strength** $\sigma$, which controls how much noise is added in the forward SDE.
2. The **number of reverse SDE steps**, which controls how finely we integrate backward from noise to data.

We train a small model for a few epochs (reuse the same trained weights) and vary these parameters during sampling.
"""

# Ablation experiment: effect of sigma and num_steps

sigmas = [10.0, 25.0, 50.0]
steps_list = [50, 200, 500]

# We'll reuse the trained model, but redefine the SDE coefficients per sigma.
ablation_results = {}

for sigma_val in sigmas:
    print(f"\n=== Sigma = {sigma_val} ===")
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma_val)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma_val)

    for steps in steps_list:
        print(f"Sampling with {steps} steps...")
        samples = euler_maruyama_sampler(
            score_model,
            marginal_prob_std_fn,
            diffusion_coeff_fn,
            batch_size=36,   # fewer to fit in memory
            num_steps=steps,
            device=device,
            eps=1e-3
        ).cpu().clamp(0, 1)
        ablation_results[(sigma_val, steps)] = samples

        # visualize
        show_samples(samples, nrow=6)
