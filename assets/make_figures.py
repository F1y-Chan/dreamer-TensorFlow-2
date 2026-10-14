# -*- coding: utf-8 -*-
"""Generate figures for the Chinese README: architecture diagram + learning curves."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = r'D:\USTB\Code\dreamer-TensorFlow 2'
ASSETS = ROOT + r'\assets'

# ----------------------------------------------------------------------
# 1) Learning curves from scores/dreamer.json (paper reproduction data)
# ----------------------------------------------------------------------
with open(ROOT + r'\scores\dreamer.json', encoding='utf-8') as f:
    data = json.load(f)

tasks = ['dmc_walker_walk', 'dmc_cheetah_run', 'dmc_cartpole_swingup', 'dmc_cup_catch']
titles = ['Walker Walk', 'Cheetah Run', 'Cartpole Swingup', 'Cup Catch']

fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
for ax, task, title in zip(axes.flat, tasks, titles):
    runs = [r for r in data if r['task'] == task]
    if not runs:
        continue
    xs = np.array(runs[0]['xs'], dtype=float) / 1e6
    min_len = min(len(r['ys']) for r in runs)
    xs = xs[:min_len]
    ys = np.array([r['ys'][:min_len] for r in runs], dtype=float)
    for y in ys:
        ax.plot(xs, y, color='#377eb8', alpha=0.25, lw=1)
    mean = ys.mean(0)
    std = ys.std(0)
    ax.plot(xs, mean, color='#377eb8', lw=2.2, label='Dreamer (mean)')
    ax.fill_between(xs, mean - std, mean + std, color='#377eb8', alpha=0.15)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Environment steps (1M)', fontsize=9)
    ax.set_ylabel('Test return', fontsize=9)
    ax.set_ylim(0, 1000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
fig.suptitle('Dreamer learning curves on DeepMind Control tasks\n(3 seeds, paper reproduction data from scores/dreamer.json)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(ASSETS + r'\learning_curves.png', dpi=150)
print('learning_curves.png saved')
plt.close(fig)

# ----------------------------------------------------------------------
# 2) Architecture diagram (world model + behavior learning)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 5.6))
ax.axis('off')
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 5.6)

def panel(x, y, w, h, title, ec):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                       linewidth=2, edgecolor=ec, facecolor='none', mutation_aspect=0.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.28, title, ha='center', va='center',
            fontsize=12.5, fontweight='bold', color=ec)

def box(x, y, w, h, text, fc, ec, fs=10.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                       linewidth=1.5, edgecolor=ec, facecolor=fc, mutation_aspect=0.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, fontweight='bold')

def arrow(x1, y1, x2, y2, color='#333333', ls='-', lw=1.6):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=16,
                        linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(a)

# ===== Left panel: Learning the world model =====
panel(0.15, 0.9, 6.9, 4.4, '1.  Learn the World Model  (reconstruction + prediction)', '#377eb8')

box(0.55, 2.75, 1.7, 1.1, 'Encoder\n(CNN)', '#eaf1fb', '#377eb8')
box(2.75, 2.75, 2.1, 1.1, 'RSSM\nRecurrent State\nSpace Model', '#eaf1fb', '#377eb8')
box(5.35, 2.75, 1.5, 1.1, 'Decoder\n(CNN)', '#eaf1fb', '#377eb8')
box(5.35, 1.15, 1.5, 1.0, 'Reward\nPredictor', '#eaf1fb', '#377eb8')

arrow(2.3, 3.3, 2.7, 3.3)
arrow(4.9, 3.3, 5.3, 3.3)
arrow(6.1, 2.7, 6.1, 2.2)          # Decoder down to reward predictor

ax.text(0.55, 2.35, 'images $o_t$', fontsize=10, color='#555555', style='italic')
ax.text(5.2, 2.15, 'reconstruction $\\hat{o}_{t+1}$', fontsize=10, color='#555555', style='italic', ha='right')
ax.text(5.3, 1.95, 'reward $\\hat{r}_{t+1}$', fontsize=10, color='#555555', style='italic', ha='left')

# ===== Right panel: Learning behavior =====
panel(7.45, 0.9, 4.9, 4.4, '2.  Learn Behavior  (latent imagination)', '#e41a1c')

box(7.85, 2.75, 2.0, 1.1, 'Imagined latent\nstate sequence', '#fdeeee', '#e41a1c')
box(10.15, 2.75, 1.2, 1.1, 'Actor\n(policy)', '#fdeeee', '#e41a1c')
box(10.15, 1.15, 1.2, 1.0, 'Value\nfunction', '#fdeeee', '#e41a1c')
box(8.6, 1.15, 1.2, 1.0, 'Imagined\nreturns', '#fdeeee', '#e41a1c')

arrow(9.9, 3.3, 10.1, 3.3, color='#e41a1c')
arrow(8.85, 2.7, 8.85, 2.2, color='#e41a1c')   # imagined states -> returns
arrow(9.2, 1.7, 10.1, 1.7, color='#e41a1c')    # returns -> value
arrow(10.75, 2.7, 10.75, 2.2, color='#e41a1c', ls=(0, (4, 2)))  # value gradients up to actor

ax.text(10.75, 1.9, 'value\ngradients', fontsize=8.5, color='#e41a1c', ha='center')
ax.text(8.85, 1.9, 'returns\n$R_t$', fontsize=9, color='#e41a1c', ha='center')

fig.suptitle('Dreamer: Dream to Control (Hafner et al., 2019)', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(ASSETS + r'\dreamer_architecture.png', dpi=150, bbox_inches='tight')
print('dreamer_architecture.png saved')
plt.close(fig)
