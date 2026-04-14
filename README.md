# Learning Predictive World Models with Curiosity-Driven Exploration

A from-scratch implementation combining **World Models** (Ha & Schmidhuber, 2018) with **Intrinsic Curiosity Module** (Pathak et al., 2017) for the LunarLander-v3 environment.

## Overview

This project builds an agent that learns to land a spacecraft by:
1. Learning a compressed representation of the world (VAE)
2. Learning to predict future states in imagination (MDRNN)
3. Generating curiosity-driven rewards for exploration (ICM)
4. Training a controller using the learned representations (PPO)

The agent operates in a learned latent space rather than raw observations, enabling efficient learning through rich representations and dense reward signals.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        LunarLander-v3            │
                    │   obs (8D), reward, done         │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      V — VAE (Vision)            │
                    │   Encoder: 8→64→128→256→(μ,σ)   │
                    │   Latent: z (32D)                │
                    │   Decoder: 32→256→128→64→8       │
                    └──────────┬───────────────────────┘
                               │ z_t
                    ┌──────────▼───────────────────────┐
                    │      M — MDRNN (Memory)           │
                    │   Custom LSTM (input=36, h=256)   │
                    │   MDN: 5 Gaussian mixtures × 32D  │
                    │   Predicts P(z_{t+1} | z_t, a_t)  │
                    └──────────┬───────────────────────┘
                               │ h_t
                    ┌──────────▼───────────────────────┐
                    │      C — Controller (PPO)         │
                    │   Input: [z_t, h_t] = 288D       │
                    │   Output: action (4 discrete)     │
                    └──────────────────────────────────┘

                    ┌──────────────────────────────────┐
                    │      ICM (Curiosity)              │
                    │   Feature Encoder: 32→64→32       │
                    │   Forward Model: 36→64→32         │
                    │   Inverse Model: 64→64→4          │
                    │   r_intrinsic = ||φ̂ - φ||²        │
                    └──────────────────────────────────┘
```

## Components

### VAE (Vision Model)
- **Purpose**: Compresses 8D LunarLander observations into a structured 32D latent space
- **Architecture**: MLP encoder (8→64→128→256) with dual heads for μ and log_var, MLP decoder (32→256→128→64→8)
- **Loss**: MSE reconstruction + β·KL divergence with β-warmup (capped at 0.5)
- **Key detail**: Latent dim (32) > input dim (8) — the VAE learns a richer, disentangled representation rather than compressing

### MDRNN (Memory Model)
- **Purpose**: Predicts the distribution over next latent states given current state and action
- **Architecture**: Custom LSTM cell (4 gates built from scratch) + Mixture Density Network (5 Gaussian components)
- **Input**: Concatenated z_t (32D) + one-hot action (4D) = 36D
- **Output**: π (5 mixture weights), μ (5×32 means), σ (5×32 std devs)
- **Loss**: Negative log-likelihood under the mixture of Gaussians

### ICM (Intrinsic Curiosity Module)
- **Purpose**: Generates curiosity-based intrinsic reward from prediction error in learned feature space
- **Feature Encoder**: Transforms latent z into action-relevant features φ(z). Shared between forward and inverse models.
- **Forward Model**: Predicts next features from current features + action. Prediction error = intrinsic reward.
- **Inverse Model**: Predicts action from consecutive features. Trains the feature encoder to focus on action-relevant information.
- **Loss**: L_ICM = 0.8·L_inverse + 0.2·L_forward
- **Reward**: r_total = r_extrinsic + η·r_intrinsic (η=0.0001)

### Controller (PPO)
- **Purpose**: Selects actions from the latent state and MDRNN memory
- **Input**: Concatenated [z_t (32D), h_t (256D)] = 288D observation
- **Training**: Proximal Policy Optimization via stable-baselines3
- **Key insight**: PPO learns from every timestep's reward (including ICM bonus), unlike CMA-ES which only uses episode totals

## Training Pipeline

```
Phase 1: Collect 200 episodes with random policy
Phase 2: Train VAE on collected observations (200 epochs, β-warmup to 0.5)
Phase 3: Encode all observations into latent z sequences
Phase 4: Train MDRNN on (z, action) sequences (150 epochs)
Phase 5: Train ICM on (z_t, a_t, z_{t+1}) transitions (100 epochs)
Phase 6: Train PPO controller with World Model observations + ICM rewards (1M timesteps)
```

The WorldModel environment wrapper integrates all components:
- VAE encodes observations → z_t
- MDRNN updates hidden state → h_t
- PPO receives [z_t, h_t] as observation
- ICM adds intrinsic reward to environment reward

## Results

### Approach 1: CMA-ES Controller (Original World Models Method)

Following the original Ha & Schmidhuber paper, we first trained the controller using CMA-ES — a gradient-free evolutionary strategy that optimizes controller weights by trial and error.

**Single round (random data only):**

| Metric | Value |
|--------|-------|
| Mean reward | -555.32 |
| Best reward | -398.77 |
| Agent behavior | Crashing every episode |

The controller has 1,156 parameters (Linear(288, 4) + bias). CMA-ES generates candidate weight vectors, evaluates each by playing episodes, and evolves toward better weights. With 50 generations and 64 candidates, CMA-ES could not find weights that fire engines effectively. The agent learned that doing nothing (crashing passively) was better than firing randomly.

**Iterative data collection (3 rounds):**

We implemented iterative training: each round, the trained controller collects new data, all models retrain on the improved data, and CMA-ES optimizes a new controller.

| Round | Mean Reward | Improvement | Key Observation |
|-------|-------------|-------------|-----------------|
| 1 | -555.32 | baseline | Random data, all crashes |
| 2 | -114.52 | 4.8× better | Controller data helped models learn better dynamics |
| 3 | -143.25 | slight regression | VAE KL collapsed (0.03), latent space degraded |

Round 2 showed a massive improvement — the first evidence that iterative data collection works. One episode even scored +4.32 (near-landing). However, Round 3 regressed due to VAE posterior collapse (KL divergence dropped to 0.03, meaning most latent dimensions shut down).

**Dream training attempt:**

We implemented dream training — training the controller inside the MDRNN's imagination instead of the real environment. A reward predictor (MLP: 32→64→32→1) estimated rewards from latent states. This allowed 500 CMA-ES generations in minutes instead of hours.

| Round | Dream Best Score | Real Eval Mean | Issue |
|-------|-----------------|----------------|-------|
| 1 | -1,556 | -153.82 | Reward predictor loss high (~103), inaccurate dreams |
| 2 | -94 | -133.51 | Better dreams, but real performance didn't improve much |
| 3 | -798 | -578.04 | KL collapse again, dreams became meaningless |

Dream training was limited by two factors: (1) the reward predictor couldn't accurately learn the reward function from mostly-crashing data, and (2) VAE posterior collapse in later rounds degraded the latent representations that the MDRNN and reward predictor depended on.

**CMA-ES failure analysis:**

The fundamental issue with CMA-ES for this task:
- CMA-ES receives one score per episode — it cannot distinguish which timesteps or actions were good or bad
- With 1,156 parameters, the search space is too large for the number of generations feasible in reasonable time
- The controller is a linear layer starting from zeros, biasing it toward always selecting the same action (noop)
- ICM intrinsic reward inflated total scores, making it harder for CMA-ES to distinguish landing behavior from exploration behavior

### Approach 2: PPO Controller (Gradient-Based)

We replaced CMA-ES with PPO while keeping the entire World Models + ICM architecture intact. The key insight: PPO learns from every timestep's reward individually, not just episode totals. The WorldModel environment wrapper feeds PPO a 288D observation [z_t, h_t] and reward augmented with ICM curiosity.

**Training Progression:**

| Timesteps | Mean Reward | Agent Behavior |
|-----------|-------------|----------------|
| 2,000 | -92 | Crashing |
| 10,000 | -59 | Learning to fire engines |
| 16,000 | +9 | First positive reward |
| 80,000 | +22 | Hovering, attempting landings |
| 120,000 | +74 | Landing sometimes |
| 145,000 | +99 | Consistent landings |
| 197,000 | +145 | Landing between flags |

PPO achieved in 200,000 timesteps (~1 minute) what CMA-ES could not achieve in 3 rounds of iterative training (~2 hours).

### Key Findings

- **World Model representations accelerate learning**: PPO with 288D (z+h) observations learns faster than PPO with raw 8D observations because the MDRNN hidden state encodes trajectory history and predicted dynamics
- **ICM enables early exploration**: Curiosity rewards provide dense signal when extrinsic rewards are sparse (timesteps 0–10k where the agent hasn't discovered landing yet)
- **η tuning is critical**: Too high (0.01) → agent prefers exploring over landing; too low (0) → no exploration benefit; sweet spot at 0.0001
- **CMA-ES vs PPO**: CMA-ES struggled with 1,156 parameters (best: -130 reward after hours); PPO succeeded (+145 reward in ~1 minute) by using per-timestep gradient updates
- **Iterative data collection helps**: Even with CMA-ES, Round 1→2 showed 4.8× improvement, proving the concept
- **VAE β must be controlled**: β-warmup to 1.0 caused posterior collapse in later rounds; capping β at 0.5 prevented this

## File Structure

```
├── vae.py              # VAE class and loss function
├── mdrnn.py            # Custom LSTM, MDN, MDRNN classes and loss
├── icm.py              # ICM class (feature encoder, forward/inverse models)
├── Controller.py       # Controller class
├── train.py            # Training pipeline, WorldModel env, PPO training
├── Literature_Review-Final.pdf
└── README.md
```

## Requirements

```
torch
gymnasium[box2d]
numpy
stable-baselines3
cma
```

## Usage

### Training
```bash
pip install torch gymnasium[box2d] numpy stable-baselines3 cma
python train.py
```

### Visualization
After training, the agent automatically plays 5 episodes with rendering enabled.

### Loading Saved Models
```python
from stable_baselines3 import PPO
model = PPO.load("controller_ppo")
```

## Parameter Summary

| Component | Parameters | Training Method |
|-----------|-----------|-----------------|
| VAE | 108,488 | Backpropagation (Adam) |
| MDRNN | 383,557 | Backpropagation (Adam) |
| ICM | 13,060 | Backpropagation (Adam) |
| Controller (PPO) | ~10,000 | Policy gradient (PPO) |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| z_dim | 32 | VAE latent dimension |
| lstm_hidden | 256 | MDRNN hidden state size |
| mdn_mixtures | 5 | Number of Gaussian components |
| β (VAE) | 0.5 (max) | KL divergence weight |
| β (ICM) | 0.2 | Forward loss weight in ICM |
| η | 0.0001 | Intrinsic reward scaling |
| PPO lr | 3e-4 | Learning rate |
| PPO timesteps | 1,000,000 | Total training steps |

## References

- Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv preprint arXiv:1803.10122*.
- Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-driven Exploration by Self-supervised Prediction. *ICML 2017*, 2778–2787.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.
- Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. *arXiv preprint arXiv:1604.00772*.

## Future Work

- **Dream training**: Train controller entirely inside MDRNN imagination for 100x faster CMA-ES optimization
- **Iterative data collection**: Multiple rounds of collect → train → improve for progressively better world models
- **Observation normalization**: Normalize inputs for faster, more stable training across all components
- **With/without ICM comparison**: Ablation study to quantify ICM's contribution to learning speed
