import matplotlib.pyplot as plt

# 1. PPO Training Curve
timesteps = [2000, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
             40960, 61440, 81920, 102400, 122880, 143360, 163840, 184320, 196608, 200704]
rewards = [-92.6, -96.8, -104, -92.2, -59.3, -41.2, -11.2, 9.61, 20.1, 2.74,
           1.96, -2.85, 22.8, 11, 74.8, 99.6, 83.9, 110, 135, 124]

plt.figure(figsize=(10, 5))
plt.plot(timesteps, rewards, linewidth=2)
plt.xlabel('Training Timesteps')
plt.ylabel('Mean Episode Reward')
plt.title('PPO Training Progression with World Models + ICM')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ppo_training_curve.png', dpi=150)
plt.show()

# 2. CMA-ES vs PPO Comparison
methods = ['CMA-ES\n(Single)', 'CMA-ES\n(Iterative)', 'CMA-ES\n(Dream)', 'PPO']
scores = [-555, -114, -578, 145]
colors = ['#E24B4A', '#EF9F27', '#E24B4A', '#1D9E75']

plt.figure(figsize=(8, 5))
plt.bar(methods, scores, color=colors)
plt.ylabel('Best Mean Reward')
plt.title('Controller Optimization: CMA-ES vs PPO')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('cmaes_vs_ppo.png', dpi=150)
plt.show()