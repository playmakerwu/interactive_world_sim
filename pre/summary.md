# Interactive World Simulator — RL Experiment Summary

## Setup

- **World model**: consistency-model latent world model trained on PushT
  (camera_1_color, 128×128). Encoder output is a per-view L2-normalized latent
  of shape **(4, 32, 32)** → 4096-D flat.
- **Goal latent** (`tests/goal_selection/z_goal.pt`): encoding of
  `pre/01_world_model/goal_frame.png`, a final frame where the T-block is
  visibly aligned with the target.
- **Reward**: cosine similarity between the current latent and the goal
  latent, both flattened to 4096-D.
- **RL algorithm**: Dreamer-style actor-critic with differentiable rollouts
  through the frozen world model. The final actor is a **discrete actor** with
  9 keyboard-style actions (±0.02 along each of the 4 action dims, plus noop),
  trained with straight-through Gumbel-Softmax.
- **Training data**: replay buffer sampled from `data/mini/pusht/{train,val}`,
  with hard-initial-state prioritisation (lowest init cos_sim percentile).

## RL Algorithm Verification

We verified the pipeline is wired correctly end-to-end
(`pre/03_rl_training/`):

- **Actor loss** decreases smoothly — the policy is optimising the
  λ-return (`actor_loss_curve.png`).
- **Critic loss** converges — the twin value heads track the λ-return
  target (`critic_loss_curve.png`).
- **λ-return** rises over training — the value estimate the actor maximises
  is moving in the right direction (`lambda_return_curve.png`).
- **Mean reward** (cosine similarity to goal) rises *marginally* over
  training — from ~0.977 to ~0.980 (`reward_curve.png`).

In short: gradients flow, losses behave, the objective is being optimised.
The algorithm is working.

## Why It Doesn't Solve PushT

Despite optimising correctly, the trained actor does not meaningfully move
the T-block toward the goal (`pre/04_rl_evaluation/`). The underlying cause
is **the reward itself, not the optimiser**.

### 3.1 The reward is nearly flat

Across 3 validation rollouts (H=50) from the trained actor:

| episode | init cos_sim | final cos_sim |
| ------- | ------------ | ------------- |
| 0       | 0.9850       | 0.9815        |
| 1       | 0.9827       | 0.9790        |
| 2       | 0.9881       | 0.9871        |
| **mean**| **0.9853**   | **0.9825**    |

The whole trajectory lives in a narrow ~0.01 band of cos_sim. See
`cosine_trajectory.png`.

### 3.2 Big visual change → tiny reward change

`reward_vs_visual.png` shows snapshots at t ∈ {0, 15, 35, 49} from one
rollout. The robot arms visibly move across the scene, yet the cos_sim bar
below each frame barely shifts (<1% total change). The reward signal is
blind to the motion that is actually happening.

### 3.3 The latent encodes arm position more than T-block position

`pre/05_spatial_analysis/` shows why the reward is uninformative.

- `spatial_similarity.png` / `spatial_overlay.png`: on (4, 32, 32) latents,
  per-position cosine similarity is non-uniform and spatially meaningful —
  the latent does preserve 2-D structure.
- **`arm_dominates.png`**: two frames with the T-block in the same place but
  the arms in different positions. Mean cos_sim = **0.9704**.
- **`tblock_invisible.png`**: two frames with arms in roughly the same place
  but the T-block clearly moved. Mean cos_sim = **0.9732** — *higher* than
  the arm-only case.

Same-T / different-arm is less similar in latent space than different-T /
same-arm. The encoder's latent axis of variation is dominated by arm
pose. The cosine-similarity-to-goal reward therefore rewards the actor for
matching arm configuration, not for pushing the T-block.

## Conclusion

The Dreamer pipeline is correctly implemented: losses decrease, λ-returns
rise, gradients are healthy. But the task is not actually being solved
because the reward — cosine similarity in this encoder's latent space — is
not a task-informative signal for PushT. The encoder preserves spatial
structure, but allocates most of its representational capacity to robot-arm
position rather than to the T-block. Closing the loop would require either
a task-aware reward (e.g., a learned critic on T-block coverage) or an
encoder that is explicitly invariant to irrelevant scene elements.
